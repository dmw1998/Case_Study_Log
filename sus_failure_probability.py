import numpy as np
import multiprocessing
from numba import jit
from tqdm import tqdm
import time

# Configuration - set SHOW_PROGRESS = False for maximum speed
SHOW_PROGRESS = True  # Change to False to disable all progress bars

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

    @jit(nopython=True)
    def Smooth(x_, x0, x1):
        eps = 1e-6
        t = (x_ - x0) / (x1 - x0 + eps)
        if x_ < x0:
            return 0.0
        elif x_ > x1:
            return 1.0
        else:
            return 6*t**5 - 15*t**4 + 10*t**3

    @jit(nopython=True)
    def A_piecewise(x_, s_):
        a = 6e-8
        b = -4e-11
        eps = 1e-6
        
        A1 = -50 + a * (x_/s_)**3 + b * (x_/s_)**4
        A2 = 0.025 * ((x_/s_) - 2300)
        A3 = 50 - a * (4600 - (x_/s_))**3 - b * (4600 - (x_/s_))**4
        A4 = 50.0
        s1 = Smooth(x_, 480 * s_, 520 * s_)
        s2 = Smooth(x_, 4080 * s_, 4120 * s_)
        s3 = Smooth(x_, 4580 * s_, 4620 * s_)
        B12 = (1 - s1)*A1 + s1*A2
        B23 = (1 - s2)*A2 + s2*A3
        B34 = (1 - s3)*A3 + s3*A4
        if x_ <= 500 * s_:
            return B12
        elif x_ <= 4100 * s_:
            return B23
        elif x_ <= 4600 * s_:
            return B34
        else:
            return A4

    @jit(nopython=True)
    def B_piecewise(x_, s_):
        c = -np.log(25/30.6)*1e-12
        d = -8.02881e-8
        e = 6.28083e-11
        eps = 1e-6
        
        B1 = d * (x_/s_)**3 + e * (x_/s_)**4
        B2 = -51 * np.exp(np.fmin(-c * ((x_/s_) - 2300)**4, 30))
        B3 = d * (4600 - (x_/s_))**3 + e * (4600 - (x_/s_))**4
        B4 = 0.0
        s1 = Smooth(x_, 480 * s_, 520 * s_)
        s2 = Smooth(x_, 4080 * s_, 4120 * s_)
        s3 = Smooth(x_, 4580 * s_, 4620 * s_)
        B12 = (1 - s1)*B1 + s1*B2
        B23 = (1 - s2)*B2 + s2*B3
        B34 = (1 - s3)*B3 + s3*B4
        if x_ <= 500 * s_:
            return B12
        elif x_ <= 4100 * s_:
            return B23
        elif x_ <= 4600 * s_:
            return B34
        else:
            return B4

    def wind_x(x_, k_, s_):
        return k_ * A_piecewise(x_, s_)

    def wind_h(x_, h_, k_, s_):
        h_safe = np.fmax(h_, 10.0)
        return k_ * h_safe / h_star * B_piecewise(x_, s_)

    def originalWindModel(x_, h_, k_, s_):
        return wind_x(x_, k_, s_), wind_h(x_, h_, k_, s_)

    def C_L(alpha_):
        if alpha_ > alpha_star:
            return C0 + C1 * alpha_
        else:
            return C0 + C1 * alpha_ + C2 * (alpha_ - alpha_star)**2
    def beta(t_):
        if t_ < sigma:
            return beta0 + beta_dot0 * t_
        else:
            return 1.0

    def aircraft_ode_func(state, u_, t_, k_value):
        """ODE function for aircraft dynamics - optimized version"""
        s_value = (1.0 / k_value) ** 2
        x_, h_, V_, gamma_, alpha_ = state
        
        T = beta(t_) * (A0 + A1 * V_ + A2 * V_**2)
        D = 0.5 * (B0 + B1 * alpha_ + B2 * alpha_**2) * rho * S * V_**2
        L = 0.5 * rho * S * C_L(alpha_) * V_**2

        # Use larger step for faster but still accurate gradient estimation
        eps_grad = 1e-6
        
        # Pre-compute wind values for gradient calculation
        Wx_c, Wh_c = originalWindModel(x_, h_, k_value, s_value)
        Wx_px, Wh_px = originalWindModel(x_ + eps_grad, h_, k_value, s_value)
        Wx_mx, Wh_mx = originalWindModel(x_ - eps_grad, h_, k_value, s_value)
        _, Wh_ph = originalWindModel(x_, h_ + eps_grad, k_value, s_value)
        _, Wh_mh = originalWindModel(x_, h_ - eps_grad, k_value, s_value)
        
        # Compute gradients
        Wx_dx = (Wx_px - Wx_mx) / (2 * eps_grad)
        Wh_dx = (Wh_px - Wh_mx) / (2 * eps_grad)
        Wh_dh = (Wh_ph - Wh_mh) / (2 * eps_grad)

        V_safe = np.fmax(V_, 1e-3)

        # Pre-compute trigonometric functions
        cos_gamma = np.cos(gamma_)
        sin_gamma = np.sin(gamma_)
        cos_alpha_delta = np.cos(alpha_ + delta)
        sin_alpha_delta = np.sin(alpha_ + delta)

        x_dot = V_ * cos_gamma + Wx_c
        h_dot = V_ * sin_gamma + Wh_c

        Wx_dot = Wx_dx * x_dot
        Wh_dot = Wh_dx * x_dot + Wh_dh * h_dot

        V_dot = (
            T / m * cos_alpha_delta
            - D / m
            - g * sin_gamma
            - (Wx_dot * cos_gamma + Wh_dot * sin_gamma)
        )
        gamma_dot = (
            T / (m * V_safe) * sin_alpha_delta
            + L / (m * V_safe)
            - g / V_safe * cos_gamma
            + (1 / V_safe) * (Wx_dot * sin_gamma - Wh_dot * cos_gamma)
        )
        alpha_dot = u_

        return np.array([x_dot, h_dot, V_dot, gamma_dot, alpha_dot])

    def rk4_step(state, u, t, dt, k_value):
        """RK4 integration step"""
        k1 = aircraft_ode_func(state, u, t, k_value)
        k2 = aircraft_ode_func(state + dt/2 * k1, u, t + dt/2, k_value)
        k3 = aircraft_ode_func(state + dt/2 * k2, u, t + dt/2, k_value)
        k4 = aircraft_ode_func(state + dt * k3, u, t + dt, k_value)
        return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    def reconstruct_trajectory(u_opt, time_grid, k, s=None, x0=0, h0=600, V0=239.7, gamma0=-0.03925, alpha0=0.1283):
        """Reconstruct trajectory and return minimum height"""
        state = np.array([x0, h0, V0, gamma0, alpha0])
        if s is None:
            s = (1.0 / k) ** 2

        u_opt = np.array(u_opt[1:])
        dt_ls = np.diff(time_grid)
        h_min = h0
        for i, u in enumerate(u_opt):
            t = time_grid[i]
            dt = dt_ls[i]
            h_min = min(h_min, state[1]) 
            state = rk4_step(state, u, t, dt, k)
        return h_min
    
    u_opt, time_grid, k, s = args
    h_min = reconstruct_trajectory(u_opt, time_grid, k, s)
    return h_min


def subset_simulation(N, p0, u_opt, time_grid, k_mean, seed):
    """Perform subset simulation to estimate failure probability - optimized version."""
    np.random.seed(seed)
    num_samples = N
    num_seeds = int(N * p0)
    l = 0
    
    # Pre-allocate arrays for better performance
    k = np.random.normal(k_mean, 0.08, N)
    s = (1.0 / k) ** 2

    # Initial samples with optional progress bar
    if SHOW_PROGRESS:
        desc = f"Initial samples (k={k_mean:.3f}, seed={seed})"
        G = np.array([limit_state_function((u_opt, time_grid, k[i], s[i])) 
                      for i in tqdm(range(num_samples), desc=desc, leave=False)])
    else:
        G = np.array([limit_state_function((u_opt, time_grid, k[i], s[i])) 
                      for i in range(num_samples)])
        
    threshold = np.percentile(G, 100 * p0)
    failure_indices = G <= threshold
    k = k[failure_indices][:num_seeds]
    G = G[failure_indices][:num_seeds]
    
    while threshold > 0.0:
        l += 1
        acc = 0
        new_samples = num_samples - num_seeds
        
        # Pre-allocate arrays for new samples
        k_new_arr = np.zeros(new_samples)
        G_new_arr = np.zeros(new_samples)
        accepted = np.zeros(new_samples, dtype=bool)
        
        # Iteration samples with optional progress bar
        if SHOW_PROGRESS:
            iter_desc = f"Iteration {l} (k={k_mean:.3f}, threshold={threshold:.2f})"
            iterator = tqdm(range(new_samples), desc=iter_desc, leave=False)
        else:
            iterator = range(new_samples)
            
        for i in iterator:
            k_new = 0.12 * k[i % len(k)] + np.sqrt(1 - 0.12 ** 2) * np.random.normal(k_mean, 0.08)
            s_new = (1.0 / k_new) ** 2
            G_new = limit_state_function((u_opt, time_grid, k_new, s_new))
            
            k_new_arr[i] = k_new
            G_new_arr[i] = G_new
            
            if G_new <= threshold:
                accepted[i] = True
                acc += 1
            else:
                # Use existing sample instead
                k_new_arr[i] = k[i % len(k)]
                G_new_arr[i] = G[i % len(G)]
        
        # Efficiently concatenate arrays
        k = np.concatenate([k, k_new_arr])
        G = np.concatenate([G, G_new_arr])
                
        threshold = np.percentile(G, 100 * p0)
        if threshold <= 0.0:
            return p0 ** (l-1) * np.mean(G <= 0.0)
        else:
            # Use boolean indexing for efficiency
            valid_indices = G <= threshold
            k = k[valid_indices][:num_seeds]
            G = G[valid_indices][:num_seeds]
    
    return p0 ** l
    
def run_subset_simulation(args):
    """Wrapper function for subset_simulation to use with multiprocessing."""
    return subset_simulation(*args)

if __name__ == "__main__":
    np.random.seed(66)
    
    # from mc_failure_probability import solve_wider_wind_ocp_pce_w_adaptive_mesh_double_measure
    
    # res_double_measure_1 = solve_wider_wind_ocp_pce_w_adaptive_mesh_double_measure(
    #     pce_order=2,
    #     k_mean=1.0,
    #     k_std=0.08,
    #     s_mean=1.0,
    #     s_std=0.15,
    #     k_obs=0.95,
    #     obs_noise=0.05,
    #     obs_time=30,
    #     du_tol=1e-3,
    #     max_iter=3,
    #     Bayesian_update=False,
    # )
    
    # res_double_measure_Bayes_1 = solve_wider_wind_ocp_pce_w_adaptive_mesh_double_measure(
    #     pce_order=2,
    #     k_mean=1.0,
    #     k_std=0.08,
    #     s_mean=1.0,
    #     s_std=0.15,
    #     k_obs=0.95,
    #     obs_noise=0.05,
    #     obs_time=30,
    #     du_tol=1e-3,
    #     max_iter=3,
    #     Bayesian_update=True,
    # )
    
    # res_double_measure_2 = solve_wider_wind_ocp_pce_w_adaptive_mesh_double_measure(
    #     pce_order=2,
    #     k_mean=1.0,
    #     k_std=0.08,
    #     s_mean=1.0,
    #     s_std=0.15,
    #     k_obs=1.05,
    #     obs_noise=0.05,
    #     obs_time=30,
    #     du_tol=1e-3,
    #     max_iter=3,
    #     Bayesian_update=False,
    # )
    
    # res_double_measure_Bayes_2 = solve_wider_wind_ocp_pce_w_adaptive_mesh_double_measure(
    #     pce_order=2,
    #     k_mean=1.0,
    #     k_std=0.08,
    #     s_mean=1.0,
    #     s_std=0.15,
    #     k_obs=1.05,
    #     obs_noise=0.05,
    #     obs_time=30,
    #     du_tol=1e-3,
    #     max_iter=3,
    #     Bayesian_update=True,
    # )

    # np.save("res_double_measure_1.npy", res_double_measure_1)
    # np.save("res_double_measure_Bayes_1.npy", res_double_measure_Bayes_1)
    # np.save("res_double_measure_2.npy", res_double_measure_2)
    # np.save("res_double_measure_Bayes_2.npy", res_double_measure_Bayes_2)
    
    # Load precomputed results
    res_no_adpt = np.load("res_no_adpt.npy", allow_pickle=True).item()
    res_adpt_u_3 = np.load("res_adpt_u_3.npy", allow_pickle=True).item()
    res_double_measure = np.load("res_double_measure.npy", allow_pickle=True).item()
    res_double_measure_Bayes = np.load("res_double_measure_Bayes.npy", allow_pickle=True).item()
    # res_double_measure_1 = np.load("res_double_measure_1.npy", allow_pickle=True).item()
    # res_double_measure_Bayes_1 = np.load("res_double_measure_Bayes_1.npy", allow_pickle=True).item()
    # res_double_measure_2 = np.load("res_double_measure_2.npy", allow_pickle=True).item()
    # res_double_measure_Bayes_2 = np.load("res_double_measure_Bayes_2.npy", allow_pickle=True).item()
    
    # Configuration - set SHOW_PROGRESS = False for maximum speed
    SHOW_PROGRESS = True  # Change to False to disable all progress bars
    n_runs = 10
    k_values = np.linspace(0.95, 1.05, 6)
    total_simulations = len(k_values) * 4 * n_runs  # 6 k values × 4 strategies × 10 runs
    
    if SHOW_PROGRESS:
        print(f"Starting {total_simulations} subset simulations...")
        print(f"Configuration: {n_runs} runs per strategy, 1000 samples per simulation")
        print(f"Expected time: 20-30 minutes with {multiprocessing.cpu_count()} CPU cores\n")
    
    start_time = time.time()
    completed_simulations = 0
    
    k_iterator = tqdm(k_values, desc="Overall Progress", unit="k_value") if SHOW_PROGRESS else k_values
    
    for i, k in enumerate(k_iterator):
        k_start_time = time.time()
        if SHOW_PROGRESS:
            print(f"\n{'='*60}")
            print(f"Running simulations for k_mean = {k:.3f} ({i+1}/{len(k_values)})")
            print(f"{'='*60}")
        
        # Prepare arguments for parallel execution
        args_no_adpt = [(1000, 0.1, res_no_adpt['u'], res_no_adpt['time_grid'], k, np.random.randint(0, 100000)) for _ in range(n_runs)]
        args_adpt_u_3 = [(1000, 0.1, res_adpt_u_3['u'], res_adpt_u_3['time_grid'], k, np.random.randint(0, 100000)) for _ in range(n_runs)]
        args_double_measure = [(1000, 0.1, res_double_measure['u'], res_double_measure['time_grid'], k, np.random.randint(0, 100000)) for _ in range(n_runs)]
        args_double_measure_Bayes = [(1000, 0.1, res_double_measure_Bayes['u'], res_double_measure_Bayes['time_grid'], k, np.random.randint(0, 100000)) for _ in range(n_runs)]
        
        # Use multiprocessing to run simulations in parallel
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            if SHOW_PROGRESS:
                print("Running no adaptation strategy...")
            prob_no_adpt = pool.map(run_subset_simulation, args_no_adpt)
            completed_simulations += n_runs
            
            if SHOW_PROGRESS:
                print("Running adaptation u_3 strategy...")
            prob_adpt_u_3 = pool.map(run_subset_simulation, args_adpt_u_3)
            completed_simulations += n_runs
            
            if SHOW_PROGRESS:
                print("Running double measure strategy...")
            prob_double_measure = pool.map(run_subset_simulation, args_double_measure)
            completed_simulations += n_runs
            
            if SHOW_PROGRESS:
                print("Running double measure Bayesian strategy...")
            prob_double_measure_Bayes = pool.map(run_subset_simulation, args_double_measure_Bayes)
            completed_simulations += n_runs
        
        # Print results for this k value
        if SHOW_PROGRESS:
            print(f"\nResults for k_mean = {k:.3f}:")
            print(f"  No adaptation:           {np.mean(prob_no_adpt):.6f} ± {np.std(prob_no_adpt):.6f}")
            print(f"  Adaptation u_3:          {np.mean(prob_adpt_u_3):.6f} ± {np.std(prob_adpt_u_3):.6f}")
            print(f"  Double measure:          {np.mean(prob_double_measure):.6f} ± {np.std(prob_double_measure):.6f}")
            print(f"  Double measure Bayesian: {np.mean(prob_double_measure_Bayes):.6f} ± {np.std(prob_double_measure_Bayes):.6f}")
            
            # Time estimates
            k_elapsed = time.time() - k_start_time
            total_elapsed = time.time() - start_time
            avg_time_per_k = total_elapsed / (i + 1)
            remaining_k = len(k_values) - (i + 1)
            eta = remaining_k * avg_time_per_k
            
            print(f"\nTiming:")
            print(f"  This k value: {k_elapsed:.1f}s")
            print(f"  Total elapsed: {total_elapsed/60:.1f}m")
            print(f"  Completed: {completed_simulations}/{total_simulations} simulations")
            print(f"  ETA: {eta/60:.1f}m remaining")
    
    total_time = time.time() - start_time
    if SHOW_PROGRESS:
        print(f"\n{'='*60}")
        print(f"All simulations completed!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"{'='*60}")
    else:
        print(f"Completed in {total_time/60:.1f} minutes")
    
    # np.save("prob_double_measure_Bayes.npy", prob_double_measure_Bayes)