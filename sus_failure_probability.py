import numpy as np
import casadi as ca
from multiprocessing import Pool

def limit_state_function(args):
    """Define the limit state function G(x). Failure occurs when G(x) <= 0."""
    m = 4662
    g = 32.172
    delta = 0.03491
    A0 = 0.4456e5
    A1 = -0.2398e2
    A2 = 0.1442e-1
    rho = 0.2203e-2
    S = 0.1560e4
    beta0 = 0.4
    beta_dot0 = 0.2
    sigma = 3
    B0 = 0.1552
    B1 = 0.12369
    B2 = 2.4203
    C0 = 0.7125
    C1 = 6.0877
    C2 = -9.0277
    alpha_star = 0.20944
    a = 6e-8
    b = -4e-11
    c = -np.log(25/30.6)*1e-12
    d = -8.02881e-8
    e = 6.28083e-11
    h_star = 1000
    eps = 1e-6

    def Smooth(x_, x0, x1):
        t = (x_ - x0) / (x1 - x0 + eps)
        return ca.if_else(x_ < x0, 0,
            ca.if_else(x_ > x1, 1, 6*t**5 - 15*t**4 + 10*t**3))

    def A_piecewise(x_, s_):
        A1 = -50 + a * (x_/s_)**3 + b * (x_/s_)**4
        A2 = 0.025 * ((x_/s_) - 2300)
        A3 = 50 - a * (4600 - (x_/s_))**3 - b * (4600 - (x_/s_))**4
        A4 = 50
        s1 = Smooth(x_, 480 * s_, 520 * s_)
        s2 = Smooth(x_, 4080 * s_, 4120 * s_)
        s3 = Smooth(x_, 4580 * s_, 4620 * s_)
        B12 = (1 - s1)*A1 + s1*A2
        B23 = (1 - s2)*A2 + s2*A3
        B34 = (1 - s3)*A3 + s3*A4
        return ca.if_else(x_ <= 500 * s_, B12,
                ca.if_else(x_ <= 4100 * s_, B23,
                ca.if_else(x_ <= 4600 * s_, B34, A4)))

    def B_piecewise(x_, s_):
        B1 = d * (x_/s_)**3 + e * (x_/s_)**4
        B2 = -51 * ca.exp(ca.fmin(-c * ((x_/s_) - 2300)**4, 30))
        B3 = d * (4600 - (x_/s_))**3 + e * (4600 - (x_/s_))**4
        B4 = 0
        s1 = Smooth(x_, 480 * s_, 520 * s_)
        s2 = Smooth(x_, 4080 * s_, 4120 * s_)
        s3 = Smooth(x_, 4580 * s_, 4620 * s_)
        B12 = (1 - s1)*B1 + s1*B2
        B23 = (1 - s2)*B2 + s2*B3
        B34 = (1 - s3)*B3 + s3*B4
        return ca.if_else(x_ <= 500 * s_, B12,
                ca.if_else(x_ <= 4100 * s_, B23,
                ca.if_else(x_ <= 4600 * s_, B34, B4)))

    def wind_x(x_, k_, s_):
        return k_ * A_piecewise(x_, s_)

    def wind_h(x_, h_, k_, s_):
        h_safe = ca.fmax(h_, 10.0)
        return k_ * h_safe / h_star * B_piecewise(x_, s_)

    def originalWindModel(x_, h_, k_, s_):
        return wind_x(x_, k_, s_), wind_h(x_, h_, k_, s_)

    def C_L(alpha_):
        return ca.if_else(alpha_ > alpha_star, C0 + C1 * alpha_,
                        C0 + C1 * alpha_ + C2 * (alpha_ - alpha_star)**2)
    def beta(t_):
        return ca.if_else(t_ < sigma, beta0 + beta_dot0 * t_, 1.0)

    def aircraft_ode(k_value):
        s_value = (1.0 / k_value) ** 2
        x_ = ca.MX.sym('x')
        h_ = ca.MX.sym('h')
        V_ = ca.MX.sym('V')
        gamma_ = ca.MX.sym('gamma')
        alpha_ = ca.MX.sym('alpha')
        u_ = ca.MX.sym('u')
        t_ = ca.MX.sym('t')

        T = beta(t_) * (A0 + A1 * V_ + A2 * V_**2)
        D = 0.5 * (B0 + B1 * alpha_ + B2 * alpha_**2) * rho * S * V_**2
        L = 0.5 * rho * S * C_L(alpha_) * V_**2

        Wx, Wh = originalWindModel(x_, h_, k_value, s_value)
        dWx_dx_fun = ca.Function("dWx_dx", [x_], [ca.gradient(Wx, x_)])
        dWh_dx_fun = ca.Function("dWh_dx", [x_, h_], [ca.gradient(Wh, x_)])
        dWh_dh_fun = ca.Function("dWh_dh", [x_, h_], [ca.gradient(Wh, h_)])

        V_safe = ca.fmax(V_, 1e-3)

        x_dot = V_ * ca.cos(gamma_) + Wx
        h_dot = V_ * ca.sin(gamma_) + Wh

        dWx_dx_val = dWx_dx_fun(x_)[0]
        dWh_dx_val = dWh_dx_fun(x_, h_)[0]
        dWh_dh_val = dWh_dh_fun(x_, h_)[0]

        Wx_dot = dWx_dx_val * x_dot
        Wh_dot = dWh_dx_val * x_dot + dWh_dh_val * h_dot

        V_dot = (
            T / m * ca.cos(alpha_ + delta)
            - D / m
            - g * ca.sin(gamma_)
            - (Wx_dot * ca.cos(gamma_) + Wh_dot * ca.sin(gamma_))
        )
        gamma_dot = (
            T / (m * V_safe) * ca.sin(alpha_ + delta)
            + L / (m * V_safe)
            - g / V_safe * ca.cos(gamma_)
            + (1 / V_safe) * (Wx_dot * ca.sin(gamma_) - Wh_dot * ca.cos(gamma_))
        )
        alpha_dot = u_

        y0 = ca.vertcat(x_, h_, V_, gamma_, alpha_)
        yk = ca.vertcat(x_dot, h_dot, V_dot, gamma_dot, alpha_dot)
        return ca.Function('f', [y0, u_, t_], [yk])

    def rk4_step(f, xk, uk, tk, dt):
        k1 = f(xk, uk, tk)
        k2 = f(xk + dt/2 * k1, uk, tk + dt/2)
        k3 = f(xk + dt/2 * k2, uk, tk + dt/2)
        k4 = f(xk + dt * k3, uk, tk + dt)
        return xk + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    def reconstruct_trajectory(u_opt, time_grid, k, s=None, x0=0, h0=600, V0=239.7, gamma0=-0.03925, alpha0=0.1283):
        # traj = {"x": [], "h": [], "V": [], "gamma": [], "alpha": [], "t": []}
        X = np.array([x0, h0, V0, gamma0, alpha0])
        if s is None:
            s = (1.0 / k) ** 2
        f = aircraft_ode(k)

        u_opt = np.array(u_opt[1:])
        dt_ls = np.diff(time_grid)
        h_min = h0
        for i, u in enumerate(u_opt):
            t = time_grid[i]
            dt = dt_ls[i]
            h_min = min(h_min, X[1]) 
            X = rk4_step(f, X, u, t, dt)
        return h_min
    
    u_opt, time_grid, k, s = args
    h_min = reconstruct_trajectory(u_opt, time_grid, k, s)
    return h_min


def subset_simulation(N, p0, u_opt, time_grid, k_mean, seed):
    """Perform subset simulation to estimate failure probability."""
    np.random.seed(seed)
    num_samples = N
    num_seeds = int(N * p0)  # Number of seeds for failure samples
    l = 0
    k = np.random.normal(k_mean, 0.08, N)  # Randomly sample k from a normal distribution
    s = (1.0 / k) ** 2

    G = np.zeros(num_samples)
    for i in range(num_samples):
        args = (u_opt, time_grid, k[i], s[i])
        G[i] = limit_state_function(args)
        
    threshold = np.percentile(G, 100 * p0)
    failure_samples = G <= threshold
    k = k[failure_samples]
    G = G[failure_samples]
    
    while threshold > 0.0:
        l += 1
        for i in range(num_samples-num_seeds):
            k_new = k[i] + np.random.normal(0.0, 0.08)
            s_new = (1.0 / k_new) ** 2
            args = (u_opt, time_grid, k_new, s_new)
            G_new = limit_state_function(args)
            
            if G_new <= threshold:
                k = np.append(k, k_new)
                G = np.append(G, G_new)
            else:
                k = np.append(k, k[i])
                G = np.append(G, G[i])
                
        threshold = np.percentile(G, 100 * p0)
        if threshold <= 0.0:
            return p0 ** (l-1) * np.mean(G <= 0.0)
        else:
            print(f"Iteration {l}, threshold: {threshold}")
            G = G[G <= threshold]
            k = k[G <= threshold]
    

if __name__ == "__main__":
    np.random.seed(63)
    
    # Load precomputed results
    res_no_adpt = np.load("res_no_adpt.npy", allow_pickle=True).item()
    res_adpt_u_3 = np.load("res_adpt_u_3.npy", allow_pickle=True).item()
    res_double_measure_Bayes = np.load("res_double_measure_Bayes.npy", allow_pickle=True).item()
    res_double_measure = np.load("res_double_measure.npy", allow_pickle=True).item()
    
    n_runs = 2
    
    def run_subset_simulation(args):
        """Wrapper function for subset_simulation to use with multiprocessing."""
        return subset_simulation(*args)
    
    for k in np.linspace(0.9, 1.1, 11):
        print(f"Running simulations for k_mean = {k}")
        
        # Prepare arguments for parallel execution
        args_no_adpt = [(1000, 0.1, res_no_adpt['u'], res_no_adpt['time_grid'], k, np.randint(0, 10000)) for _ in range(n_runs)]
        args_adpt_u_3 = [(1000, 0.1, res_adpt_u_3['u'], res_adpt_u_3['time_grid'], k, np.randint(0, 10000)) for _ in range(n_runs)]
        args_double_measure = [(1000, 0.1, res_double_measure['u'], res_double_measure['time_grid'], k, np.randint(0, 10000)) for _ in range(n_runs)]
        args_double_measure_Bayes = [(1000, 0.1, res_double_measure_Bayes['u'], res_double_measure_Bayes['time_grid'], k, np.randint(0, 10000)) for _ in range(n_runs)]
        
        # Use multiprocessing to run simulations in parallel
        with Pool() as pool:
            prob_no_adpt = pool.map(run_subset_simulation, args_no_adpt)
            prob_adpt_u_3 = pool.map(run_subset_simulation, args_adpt_u_3)
            prob_double_measure = pool.map(run_subset_simulation, args_double_measure)
            prob_double_measure_Bayes = pool.map(run_subset_simulation, args_double_measure_Bayes)
        
        # Print results
        print("Probability of failure without adaptation:", np.mean(prob_no_adpt))
        print("Probability of failure with adaptation u_3:", np.mean(prob_adpt_u_3))
        print("Probability of failure with double measure:", np.mean(prob_double_measure))
        print("Probability of failure with double measure Bayesian:", np.mean(prob_double_measure_Bayes))