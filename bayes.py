import casadi as ca
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

np.random.seed(42)

def solve_wider_wind_ocp_pce_w_adaptive_mesh_double_measure(pce_order, k_mean, k_std, s_mean, s_std, k_obs, obs_noise, obs_time, du_tol=1e-3, max_iter=3, Bayesian_update=True):
    
    def bayesian_update_pce(k_obs=k_obs, obs_noise=obs_noise, k_mean=k_mean, k_std=k_std):
        var_prior = k_std**2
        var_obs = obs_noise**2
        
        var_posterior = 1 / (1 / var_prior + 1 / var_obs)
        mean_posterior = var_posterior * (k_obs / var_obs + k_mean / var_prior)
        
        return mean_posterior, np.sqrt(var_posterior)
    
    # def split_time_grid(time_grid, obs_time):
    #     time_grid = np.unique(np.append(time_grid, obs_time))
    #     idx_obs = np.where(time_grid == obs_time)[0][0]
    #     grid_before = time_grid[:idx_obs + 1]
    #     grid_after = time_grid[idx_obs:]
    #     return grid_before, grid_after
    
    def refine_time_grid(u_guess, time_grid, du_tol=1e-2):
        refined = False
        u_guess = np.asarray(u_guess)
        time_grid = np.asarray(time_grid)
        du = np.abs(np.diff(np.insert(u_guess, 0, u_guess[0])))
        refined_time_grid = [time_grid[0]]  # Start with the first time point
        for i in range(1, len(du)):
            if du[i-1] > du_tol:
                refined = True
                t_mid = (time_grid[i-1] + time_grid[i]) / 2
                refined_time_grid.append(t_mid)
            refined_time_grid.append(time_grid[i])
        refined_time_grid.append(time_grid[-1])  # Ensure the last time point is included
        return refined, np.unique(refined_time_grid)
    
    # Time and discretization
    tf = 50        # final time [sec]
    N = 50         # number of control intervals
    # dt = tf / N    # time step
    time_grid = np.linspace(0, tf, N+1)  # time grid for adaptive mesh

    # Aircraft physical constants
    m = 4662                # mass [lb sec^2 / ft]
    g = 32.172              # gravity [ft/sec^2]
    delta = 0.03491         # thrust inclination angle [rad]

    # Thrust model coefficients: T = A0 + A1*V + A2*V^2
    A0 = 0.4456e5           # [lb]
    A1 = -0.2398e2          # [lb sec / ft]
    A2 = 0.1442e-1          # [lb sec^2 / ft^2]

    # Aerodynamic model
    rho = 0.2203e-2         # air density [lb sec^2 / ft^4]
    S = 0.1560e4            # reference surface area [ft^2]

    # Wind model 3 beta (smoothing) parameters
    beta0 = 0.4             # initial beta value (approximate)
    beta_dot0 = 0.2         # initial beta rate
    sigma = 3               # time to reach beta = 1 [sec]

    # C_D(alpha) = B0 + B1 * alpha + B2 * alpha**2, D = 0.5 * C_D(α) * ρ * S * V²
    B0 = 0.1552
    B1 = 0.12369            # [1/rad]
    B2 = 2.4203             # [1/rad^2]

    # Lift coefficient: C_L = C0 + C1 * alpha (+ C2 * alpha**2)
    C0 = 0.7125             # baseline lift coefficient
    C1 = 6.0877             # AOA lift slope [1/rad]

    # Lift/drag model optional extensions (if needed)
    C2 = -9.0277            # [rad^-2] — e.g., for moment or drag extension

    # Angle of attack & control constraints
    umax = 0.05236          # max control input (rate of change of alpha) [rad/sec]
    alphamax = 0.3          # max angle of attack [rad]
    alpha_star = 0.20944    # changing pt of AoA

    # Wind model x parameters (piecewise smooth wind)
    a = 6e-8                 # x transition midpoint [ft]
    b = -4e-11               # second transition point [ft]

    # Wind model h parameters (polynomial form)
    c = -np.log(25/30.6)*1e-12      # transition smoothing width [ft]
    d = -8.02881e-8         # polynomial coeff [sec^-1 ft^-2]
    e = 6.28083e-11         # polynomial coeff [sec^-1 ft^-3]

    # Cost function / target altitude
    hR = 1000               # reference altitude [ft]
    h_star = 1000           # used in some wind models

    # Auxiliary
    eps = 1e-6              # to avoid division by zero in V

    # Scaling factors (used if normalizing states)
    xscale = 10000          # [ft]
    hscale = 1000           # [ft]
    Vscale = 240            # [ft/sec]
    gammascale = 0.1        # [rad]
    alphascale = 0.3        # [rad]
    uscale = 0.05           # [rad/sec]

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

    def C_L(alpha_):
        return ca.if_else(alpha_ > alpha_star, C0 + C1 * alpha_,
                          C0 + C1 * alpha_ + C2 * (alpha_ - alpha_star)**2)
    def beta(t_):
        return ca.if_else(t_ < sigma, beta0 + beta_dot0 * t_, 1.0)

    def rk4_step(f, xk, uk, tk, dt, k_val, s_val):
        k1 = f(xk, uk, tk, k_val, s_val)
        k2 = f(xk + dt/2 * k1, uk, tk + dt/2, k_val, s_val)
        k3 = f(xk + dt/2 * k2, uk, tk + dt/2, k_val, s_val)
        k4 = f(xk + dt * k3, uk, tk + dt, k_val, s_val)
        return xk + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def solve_ocp(time_grid, pce_order=pce_order, k_mean=k_mean, k_std=k_std, s_mean=s_mean, s_std=s_std, X0=None):
        
        # PCE parameters
        k_dist = cp.Normal(k_mean, k_std)
        s_dist = cp.Normal(s_mean, s_std)
        joint_dist = cp.J(k_dist, s_dist)
        
        poly_basis = cp.expansion.stieltjes(pce_order, joint_dist)
        nodes, weights = cp.generate_quadrature(pce_order, joint_dist, rule='G')
        # print(poly_basis.shape)     # ((pce_order+1)(pce_order+2)/2, )
        # print(weights.shape)        # ((pce_order+1)^2, )
        # print(nodes.shape)          # (2, (pce_order+1)^2)
        
        M = nodes.shape[1]              # number of PCE samples
        
        weights_dm = ca.DM(weights)   # ((pce_order+1)^2， )
        
        Psi_mat = poly_basis(*nodes)    # (M_poly_basis, (pce_order+1)^2)
        Psi_dm = ca.DM(Psi_mat)         # (M_poly_basis, (pce_order+1)^2)
    
        N = len(time_grid)
        dt_list = np.diff(time_grid)  # Calculate time steps from the time grid
        
        # Opti instance and scaled variables
        opti = ca.Opti()
        x_s = opti.variable(M, N+1)
        h_s = opti.variable(M, N+1)
        V_s = opti.variable(M, N+1)
        gamma_s = opti.variable(M, N+1)
        alpha_s = opti.variable(M, N+1)
        X_pce = opti.variable(5*(pce_order+1), N+1)
        u_s = opti.variable(N)

        # Unscaled variables for dynamics
        x = x_s * xscale
        h = h_s * hscale
        V = V_s * Vscale
        gamma = gamma_s * gammascale
        alpha = alpha_s * alphascale
        u = u_s * uscale

        # Symbolic derivatives
        x_sym = ca.MX.sym("x")
        h_sym = ca.MX.sym("h")
        k_sym = ca.MX.sym("k")
        s_sym = ca.MX.sym("s")
        Wx_expr = wind_x(x_sym, k_sym, s_sym)
        Wh_expr = wind_h(x_sym, h_sym, k_sym, s_sym)
        dWx_dx_fun = ca.Function("dWx_dx", [x_sym, k_sym, s_sym], [ca.gradient(Wx_expr, x_sym)])
        dWh_dx_fun = ca.Function("dWh_dx", [x_sym, h_sym, k_sym, s_sym], [ca.gradient(Wh_expr, x_sym)])
        dWh_dh_fun = ca.Function("dWh_dh", [x_sym, h_sym, k_sym, s_sym], [ca.gradient(Wh_expr, h_sym)])
        
        def aircraft_ode(X, u_, t_, k_val, s_val):
            x_, h_, V_, gamma_, alpha_ = ca.vertsplit(X)

            T = beta(t_) * (A0 + A1 * V_ + A2 * V_**2)
            D = 0.5 * (B0 + B1 * alpha_ + B2 * alpha_**2) * rho * S * V_**2
            L = 0.5 * rho * S * C_L(alpha_) * V_**2

            Wx = wind_x(x_, k_val, s_val)
            Wh = wind_h(x_, h_, k_val, s_val)
            V_safe = ca.fmax(V_, 1e-3)

            x_dot = V_ * ca.cos(gamma_) + Wx
            h_dot = V_ * ca.sin(gamma_) + Wh

            dWx_dx_val = dWx_dx_fun(x_, k_val, s_val)[0]
            dWh_dx_val = dWh_dx_fun(x_, h_, k_val, s_val)[0]
            dWh_dh_val = dWh_dh_fun(x_, h_, k_val, s_val)[0]

            Wx_dot = dWx_dx_val * x_dot
            Wh_dot = dWh_dx_val * x_dot + dWh_dh_val * h_dot

            V_dot = T / m * ca.cos(alpha_ + delta) - D / m - g * ca.sin(gamma_) - (Wx_dot * ca.cos(gamma_) + Wh_dot * ca.sin(gamma_))
            gamma_dot = T / (m * V_safe) * ca.sin(alpha_ + delta) + L / (m * V_safe) - g / V_safe * ca.cos(gamma_) + (1 / V_safe) * (Wx_dot * ca.sin(gamma_) - Wh_dot * ca.cos(gamma_))
            alpha_dot = u_

            return ca.vertcat(x_dot, h_dot, V_dot, gamma_dot, alpha_dot)

        J = 0

        # Initial conditions
        if X0 is None:
            opti.subject_to(x_s[:, 0] == 0)
            opti.subject_to(h_s[:, 0] == 600 / hscale)
            opti.subject_to(V_s[:, 0] == 239.7 / Vscale)
            opti.subject_to(gamma_s[:, 0] == -0.03925 / gammascale)
            opti.subject_to(alpha_s[:, 0] == 0.1283 / alphascale)
            opti.subject_to(ca.vec(V_s) >= 1e-2 / Vscale)
            # opti.subject_to(gamma_s[:, -1] == -0.05236 / gammascale)
            opti.subject_to(gamma_s[:, -1] >= 0)
        else:
            opti.subject_to(x_s[:, 0] == X0["x"] / xscale)
            opti.subject_to(h_s[:, 0] == X0["h"] / hscale)
            opti.subject_to(V_s[:, 0] == X0["V"] / Vscale)
            opti.subject_to(gamma_s[:, 0] == X0["gamma"] / gammascale)
            opti.subject_to(alpha_s[:, 0] == X0["alpha"] / alphascale)
            opti.subject_to(ca.vec(V_s) >= 1e-2 / Vscale)
            # opti.subject_to(gamma_s[:, -1] == -0.05236 / gammascale)
            opti.subject_to(gamma_s[:, -1] >= 0)
        
        h_ddot_all = [[] for _ in range(N)]
        
        for j in range(M):
            k_val, s_val = nodes[:, j]
            
            i = 0
            dt = 0
            tk = time_grid[i]  # Current time from the time grid
                
            h_physical = h_s[:, i] * hscale
            h_mean = ca.dot(weights_dm, h_physical)

            h_pce_coeffs = ca.mtimes(Psi_dm, weights_dm * h_physical)
            h_high_coeffs = h_pce_coeffs[1:]
            h_std_term = ca.sqrt(ca.sumsqr(h_high_coeffs))

            deviation = hR - h_mean + h_std_term * 12
            scaled_deviation = deviation / hscale
            J += dt * scaled_deviation**6

            Xk = ca.vertcat(x[j, i], h[j, i], V[j, i], gamma[j, i], alpha[j, i])
            Uk = u[i]
            
            # h_ddot = d^2h/dt^2 = V_dot * sin(gamma) + V * cos(gamma) * gamma_dot + Wh_dot
            ode_out = aircraft_ode(Xk, Uk, tk, k_val, s_val)
            x_dot_k = ode_out[0]
            h_dot_k = ode_out[1]
            V_dot_k = ode_out[2]
            gamma_dot_k = ode_out[3]
            # alpha_dot_k = ode_out[4]
            
            dWh_dx_val = dWh_dx_fun(Xk[0], Xk[1], k_val, s_val)[0]
            dWh_dh_val = dWh_dh_fun(Xk[0], Xk[1], k_val, s_val)[0]
            Wh_dot_k = dWh_dx_val * x_dot_k + dWh_dh_val * h_dot_k
            
            h_ddot_k = V_dot_k * ca.sin(Xk[3]) + Xk[2] * ca.cos(Xk[3]) * gamma_dot_k + Wh_dot_k
            
            h_ddot_all[i].append(h_ddot_k)
            
            Xk_end = rk4_step(aircraft_ode, Xk, Uk, tk, dt, k_val, s_val)
            X_next = ca.vertcat(x[j, i+1], h[j, i+1], V[j, i+1], gamma[j, i+1], alpha[j, i+1])
            opti.subject_to(X_next == Xk_end)

            opti.subject_to(opti.bounded(-1, u_s[i], 1))

            alpha_i = alpha_s[:, i] * alphascale
            alpha_pce_coeffs = ca.mtimes(Psi_dm, weights_dm * alpha_i)
            alpha_high_coeffs = alpha_pce_coeffs[1:]
            alpha_mean = ca.dot(weights_dm, alpha_i)
            std_term = ca.sqrt(ca.sumsqr(alpha_high_coeffs))
            
            for i in range(1, N):
                dt = dt_list[i-1]  # Use the time step from the time grid
                tk = time_grid[i]  # Current time from the time grid
                
                h_physical = h_s[:, i] * hscale
                h_mean = ca.dot(weights_dm, h_physical)

                h_pce_coeffs = ca.mtimes(Psi_dm, weights_dm * h_physical)
                h_high_coeffs = h_pce_coeffs[1:]
                h_std_term = ca.sqrt(ca.sumsqr(h_high_coeffs))

                deviation = hR - h_mean + h_std_term * 12
                scaled_deviation = deviation / hscale
                J += dt * scaled_deviation**6

                Xk = ca.vertcat(x[j, i], h[j, i], V[j, i], gamma[j, i], alpha[j, i])
                Uk = u[i]
                
                # h_ddot = d^2h/dt^2 = V_dot * sin(gamma) + V * cos(gamma) * gamma_dot + Wh_dot
                ode_out = aircraft_ode(Xk, Uk, tk, k_val, s_val)
                x_dot_k = ode_out[0]
                h_dot_k = ode_out[1]
                V_dot_k = ode_out[2]
                gamma_dot_k = ode_out[3]
                # alpha_dot_k = ode_out[4]
                
                dWh_dx_val = dWh_dx_fun(Xk[0], Xk[1], k_val, s_val)[0]
                dWh_dh_val = dWh_dh_fun(Xk[0], Xk[1], k_val, s_val)[0]
                Wh_dot_k = dWh_dx_val * x_dot_k + dWh_dh_val * h_dot_k
                
                h_ddot_k = V_dot_k * ca.sin(Xk[3]) + Xk[2] * ca.cos(Xk[3]) * gamma_dot_k + Wh_dot_k
                
                h_ddot_all[i].append(h_ddot_k)
                
                Xk_end = rk4_step(aircraft_ode, Xk, Uk, tk, dt, k_val, s_val)
                X_next = ca.vertcat(x[j, i+1], h[j, i+1], V[j, i+1], gamma[j, i+1], alpha[j, i+1])
                opti.subject_to(X_next == Xk_end)

                opti.subject_to(opti.bounded(-1, u_s[i], 1))

                alpha_i = alpha_s[:, i] * alphascale
                alpha_pce_coeffs = ca.mtimes(Psi_dm, weights_dm * alpha_i)
                alpha_high_coeffs = alpha_pce_coeffs[1:]
                alpha_mean = ca.dot(weights_dm, alpha_i)
                std_term = ca.sqrt(ca.sumsqr(alpha_high_coeffs))

            opti.subject_to(alpha_mean + 3 * std_term <= alphamax)
            opti.subject_to(alpha_mean - 3 * std_term >= -alphamax)
                
        for i in range(N):
            h_ddot_vec = ca.vertcat(*h_ddot_all[i])
            h_ddot_pce_coeffs = ca.mtimes(Psi_dm, weights_dm * h_ddot_vec)
            h_ddot_mean = h_ddot_pce_coeffs[0]
            h_ddot_high_coeffs = h_ddot_pce_coeffs[1:]
            h_ddot_std_term = ca.sqrt(ca.sumsqr(h_ddot_high_coeffs))
            
            opti.subject_to(h_ddot_mean + 3 * h_ddot_std_term <= 10 * g)
            opti.subject_to(h_ddot_mean - 3 * h_ddot_std_term >= -2 * g)

        opti.minimize(J)

        # Initial guess
        for j in range(M):
            opti.set_initial(x_s[j, :], np.linspace(0, 1, N+1))
            opti.set_initial(h_s[j, :], 0.6)  # 600 ft / 1000
            opti.set_initial(V_s[j, :], 239.7 / Vscale)
            opti.set_initial(gamma_s[j, :], -0.01 / gammascale)
            opti.set_initial(alpha_s[j, :], 0.02 / alphascale)
        opti.set_initial(u_s, 0)

        # Solver
        opts = {
            "expand": True,
            "ipopt": {
                # "max_iter": 3000,
                "max_cpu_time": 120,
                "tol": 1e-6,
                "print_level": 0,
                "linear_solver": "mumps",
                "hessian_approximation": "limited-memory",
                # "bound_push": 1e-8,
                # "bound_frac": 1e-8
            }
        }
        opti.solver("ipopt", opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            opti.debug.show_infeasibilities()
            print(e)
            return {
                "x": opti.debug.value(x),
                "h": opti.debug.value(h),
                "V": opti.debug.value(V),
                "gamma": opti.debug.value(gamma),
                "alpha": opti.debug.value(alpha),
                "u": opti.debug.value(u),
                "J": opti.debug.value(J),
                "time_grid": time_grid,
                "dt_list": dt_list,
                "status": "failed"
            }

        return {
            "x": sol.value(x),
            "h": sol.value(h),
            "V": sol.value(V),
            "gamma": sol.value(gamma),
            "alpha": sol.value(alpha),
            "u": sol.value(u),
            "J": sol.value(J),
            "time_grid": time_grid,
            "dt_list": dt_list,
            "status": "success"
        }

    def solve_and_refine(time_grid=time_grid, max_iter=max_iter, Bayesian_update=Bayesian_update, obs_time=obs_time, du_tol=du_tol):
        refined = True
        iter_count = 0
        # print(f"Initial time grid: {len(time_grid)} points")

        while refined and iter_count <= max_iter:
            sol_1 = solve_ocp(time_grid)
            
            if Bayesian_update:
                k_posterior, k_std_posterior = bayesian_update_pce()
                s_mean = 1.0
                s_std = 0.15
            else:
                k_posterior = 1.1
                k_std_posterior = 0.08
                s_mean = 0.8
                s_std = 0.1
            
            idx_obs = np.argmin(np.abs(sol_1["time_grid"] - obs_time))
            # print(f"Observation at time {obs_time} sec, index {idx_obs}")
            # print(len(sol_1["time_grid"]), len(sol_1["u"]), sol_1["x"].shape, sol_1["h"].shape)
            
            X0 = {
                "x": sol_1["x"][:, idx_obs],
                "h": sol_1["h"][:, idx_obs],
                "V": sol_1["V"][:, idx_obs],
                "gamma": sol_1["gamma"][:, idx_obs],
                "alpha": sol_1["alpha"][:, idx_obs]
            }
            
            grid_2 = sol_1["time_grid"][idx_obs:]  # Time grid after observation
            
            sol_2 = solve_ocp(grid_2, k_mean=k_posterior, k_std=k_std_posterior, s_mean=s_mean, s_std=s_std, X0=X0)
            # print(f"Solution 1 status: {sol_1['status']}, Solution 2 status: {sol_2['status']}")
            # print(f"Solution 1 time grid: {len(sol_1['time_grid'])}, Solution 2 time grid: {len(sol_2['time_grid'])}")
            
            u_full = np.concatenate((sol_1["u"][:idx_obs], sol_2["u"]))
            # print(f"Full control u shape: {u_full.shape}")
            
            # Check if further refinement is needed
            refined, new_time_grid = refine_time_grid(u_full, time_grid, du_tol)
            if refined and iter_count < max_iter:
                time_grid = new_time_grid
                iter_count += 1
                print(f"Refining time grid, iteration {iter_count}")
            else:
                break
    
            
        return {
            "x": np.concatenate([sol_1["x"][:, :idx_obs], sol_2["x"]], axis=1),
            "h": np.concatenate([sol_1["h"][:, :idx_obs], sol_2["h"]], axis=1),
            "V": np.concatenate([sol_1["V"][:, :idx_obs], sol_2["V"]], axis=1),
            "gamma": np.concatenate([sol_1["gamma"][:, :idx_obs], sol_2["gamma"]], axis=1),
            "alpha": np.concatenate([sol_1["alpha"][:, :idx_obs], sol_2["alpha"]], axis=1),
            "u": u_full,
            # "J": sol_1["J"] + sol_2["J"],
            "time_grid": time_grid,
            "dt_list": np.diff(time_grid),
            "status": "success" if sol_1["status"] == "success" and sol_2["status"] == "success" else "failed"
        }
    
    return solve_and_refine()



def solve_wider_wind_ocp_pce_w_adaptive_mesh(pce_order, k_mean, k_std, s_mean, s_std, du_tol=1e-3, max_iter=3):
    
    def refine_time_grid(u_guess, time_grid, du_tol=1e-2):
        refined = False
        u_guess = np.asarray(u_guess)
        time_grid = np.asarray(time_grid)
        du = np.abs(np.diff(np.insert(u_guess, 0, u_guess[0])))
        refined_time_grid = [time_grid[0]]  # Start with the first time point
        for i in range(1, len(du)):
            if du[i-1] > du_tol:
                refined = True
                t_mid = (time_grid[i-1] + time_grid[i]) / 2
                refined_time_grid.append(t_mid)
            refined_time_grid.append(time_grid[i])
        refined_time_grid.append(time_grid[-1])  # Ensure the last time point is included
        return refined, np.unique(refined_time_grid)
    
    # Time and discretization
    tf = 50        # final time [sec]
    N = 50         # number of control intervals
    # dt = tf / N    # time step
    time_grid = np.linspace(0, tf, N+1)  # time grid for adaptive mesh

    # Aircraft physical constants
    m = 4662                # mass [lb sec^2 / ft]
    g = 32.172              # gravity [ft/sec^2]
    delta = 0.03491         # thrust inclination angle [rad]

    # Thrust model coefficients: T = A0 + A1*V + A2*V^2
    A0 = 0.4456e5           # [lb]
    A1 = -0.2398e2          # [lb sec / ft]
    A2 = 0.1442e-1          # [lb sec^2 / ft^2]

    # Aerodynamic model
    rho = 0.2203e-2         # air density [lb sec^2 / ft^4]
    S = 0.1560e4            # reference surface area [ft^2]

    # Wind model 3 beta (smoothing) parameters
    beta0 = 0.4             # initial beta value (approximate)
    beta_dot0 = 0.2         # initial beta rate
    sigma = 3               # time to reach beta = 1 [sec]

    # C_D(alpha) = B0 + B1 * alpha + B2 * alpha**2, D = 0.5 * C_D(α) * ρ * S * V²
    B0 = 0.1552
    B1 = 0.12369            # [1/rad]
    B2 = 2.4203             # [1/rad^2]

    # Lift coefficient: C_L = C0 + C1 * alpha (+ C2 * alpha**2)
    C0 = 0.7125             # baseline lift coefficient
    C1 = 6.0877             # AOA lift slope [1/rad]

    # Lift/drag model optional extensions (if needed)
    C2 = -9.0277            # [rad^-2] — e.g., for moment or drag extension

    # Angle of attack & control constraints
    umax = 0.05236          # max control input (rate of change of alpha) [rad/sec]
    alphamax = 0.3          # max angle of attack [rad]
    alpha_star = 0.20944    # changing pt of AoA

    # Wind model x parameters (piecewise smooth wind)
    a = 6e-8                 # x transition midpoint [ft]
    b = -4e-11               # second transition point [ft]

    # Wind model h parameters (polynomial form)
    c = -np.log(25/30.6)*1e-12      # transition smoothing width [ft]
    d = -8.02881e-8         # polynomial coeff [sec^-1 ft^-2]
    e = 6.28083e-11         # polynomial coeff [sec^-1 ft^-3]

    # Cost function / target altitude
    hR = 1000               # reference altitude [ft]
    h_star = 1000           # used in some wind models

    # Auxiliary
    eps = 1e-6              # to avoid division by zero in V

    # Scaling factors (used if normalizing states)
    xscale = 10000          # [ft]
    hscale = 1000           # [ft]
    Vscale = 240            # [ft/sec]
    gammascale = 0.1        # [rad]
    alphascale = 0.3        # [rad]
    uscale = 0.05           # [rad/sec]

    # PCE parameters
    k_dist = cp.Normal(k_mean, k_std)
    s_dist = cp.Normal(s_mean, s_std)
    joint_dist = cp.J(k_dist, s_dist)
    
    poly_basis = cp.expansion.stieltjes(pce_order, joint_dist)
    nodes, weights = cp.generate_quadrature(pce_order, joint_dist, rule='G')
    # print(poly_basis.shape)     # ((pce_order+1)(pce_order+2)/2, )
    # print(weights.shape)        # ((pce_order+1)^2, )
    # print(nodes.shape)          # (2, (pce_order+1)^2)
    
    M = nodes.shape[1]              # number of PCE samples
    
    weights_dm = ca.DM(weights)   # ((pce_order+1)^2， )
    
    Psi_mat = poly_basis(*nodes)    # (M_poly_basis, (pce_order+1)^2)
    Psi_dm = ca.DM(Psi_mat)         # (M_poly_basis, (pce_order+1)^2)

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

    def C_L(alpha_):
        return ca.if_else(alpha_ > alpha_star, C0 + C1 * alpha_,
                          C0 + C1 * alpha_ + C2 * (alpha_ - alpha_star)**2)
    def beta(t_):
        return ca.if_else(t_ < sigma, beta0 + beta_dot0 * t_, 1.0)

    def rk4_step(f, xk, uk, tk, dt, k_val, s_val):
        k1 = f(xk, uk, tk, k_val, s_val)
        k2 = f(xk + dt/2 * k1, uk, tk + dt/2, k_val, s_val)
        k3 = f(xk + dt/2 * k2, uk, tk + dt/2, k_val, s_val)
        k4 = f(xk + dt * k3, uk, tk + dt, k_val, s_val)
        return xk + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def solve_ocp(time_grid):
        N = len(time_grid)
        dt_list = np.diff(time_grid)  # Calculate time steps from the time grid
        
        # Opti instance and scaled variables
        opti = ca.Opti()
        x_s = opti.variable(M, N+1)
        h_s = opti.variable(M, N+1)
        V_s = opti.variable(M, N+1)
        gamma_s = opti.variable(M, N+1)
        alpha_s = opti.variable(M, N+1)
        X_pce = opti.variable(5*(pce_order+1), N+1)
        u_s = opti.variable(N)

        # Unscaled variables for dynamics
        x = x_s * xscale
        h = h_s * hscale
        V = V_s * Vscale
        gamma = gamma_s * gammascale
        alpha = alpha_s * alphascale
        u = u_s * uscale

        # Symbolic derivatives
        x_sym = ca.MX.sym("x")
        h_sym = ca.MX.sym("h")
        k_sym = ca.MX.sym("k")
        s_sym = ca.MX.sym("s")
        Wx_expr = wind_x(x_sym, k_sym, s_sym)
        Wh_expr = wind_h(x_sym, h_sym, k_sym, s_sym)
        dWx_dx_fun = ca.Function("dWx_dx", [x_sym, k_sym, s_sym], [ca.gradient(Wx_expr, x_sym)])
        dWh_dx_fun = ca.Function("dWh_dx", [x_sym, h_sym, k_sym, s_sym], [ca.gradient(Wh_expr, x_sym)])
        dWh_dh_fun = ca.Function("dWh_dh", [x_sym, h_sym, k_sym, s_sym], [ca.gradient(Wh_expr, h_sym)])
        
        def aircraft_ode(X, u_, t_, k_val, s_val):
            x_, h_, V_, gamma_, alpha_ = ca.vertsplit(X)

            T = beta(t_) * (A0 + A1 * V_ + A2 * V_**2)
            D = 0.5 * (B0 + B1 * alpha_ + B2 * alpha_**2) * rho * S * V_**2
            L = 0.5 * rho * S * C_L(alpha_) * V_**2

            Wx = wind_x(x_, k_val, s_val)
            Wh = wind_h(x_, h_, k_val, s_val)
            V_safe = ca.fmax(V_, 1e-3)

            x_dot = V_ * ca.cos(gamma_) + Wx
            h_dot = V_ * ca.sin(gamma_) + Wh

            dWx_dx_val = dWx_dx_fun(x_, k_val, s_val)[0]
            dWh_dx_val = dWh_dx_fun(x_, h_, k_val, s_val)[0]
            dWh_dh_val = dWh_dh_fun(x_, h_, k_val, s_val)[0]

            Wx_dot = dWx_dx_val * x_dot
            Wh_dot = dWh_dx_val * x_dot + dWh_dh_val * h_dot

            V_dot = T / m * ca.cos(alpha_ + delta) - D / m - g * ca.sin(gamma_) - (Wx_dot * ca.cos(gamma_) + Wh_dot * ca.sin(gamma_))
            gamma_dot = T / (m * V_safe) * ca.sin(alpha_ + delta) + L / (m * V_safe) - g / V_safe * ca.cos(gamma_) + (1 / V_safe) * (Wx_dot * ca.sin(gamma_) - Wh_dot * ca.cos(gamma_))
            alpha_dot = u_

            return ca.vertcat(x_dot, h_dot, V_dot, gamma_dot, alpha_dot)

        J = 0

        # Initial conditions
        opti.subject_to(x_s[:, 0] == 0)
        opti.subject_to(h_s[:, 0] == 600 / hscale)
        opti.subject_to(V_s[:, 0] == 239.7 / Vscale)
        opti.subject_to(gamma_s[:, 0] == -0.03925 / gammascale)
        opti.subject_to(alpha_s[:, 0] == 0.1283 / alphascale)
        opti.subject_to(ca.vec(V_s) >= 1e-2 / Vscale)
        # opti.subject_to(gamma_s[:, -1] == -0.05236 / gammascale)
        opti.subject_to(gamma_s[:, -1] >= 0)
        
        h_ddot_all = [[] for _ in range(N)]
        
        for j in range(M):
            k_val, s_val = nodes[:, j]
            
            i = 0
            dt = 0
            tk = time_grid[i]  # Current time from the time grid
                
            h_physical = h_s[:, i] * hscale
            h_mean = ca.dot(weights_dm, h_physical)

            h_pce_coeffs = ca.mtimes(Psi_dm, weights_dm * h_physical)
            h_high_coeffs = h_pce_coeffs[1:]
            h_std_term = ca.sqrt(ca.sumsqr(h_high_coeffs))

            deviation = hR - h_mean + h_std_term * 12
            scaled_deviation = deviation / hscale
            J += dt * scaled_deviation**6

            Xk = ca.vertcat(x[j, i], h[j, i], V[j, i], gamma[j, i], alpha[j, i])
            Uk = u[i]
            
            # h_ddot = d^2h/dt^2 = V_dot * sin(gamma) + V * cos(gamma) * gamma_dot + Wh_dot
            ode_out = aircraft_ode(Xk, Uk, tk, k_val, s_val)
            x_dot_k = ode_out[0]
            h_dot_k = ode_out[1]
            V_dot_k = ode_out[2]
            gamma_dot_k = ode_out[3]
            # alpha_dot_k = ode_out[4]
            
            dWh_dx_val = dWh_dx_fun(Xk[0], Xk[1], k_val, s_val)[0]
            dWh_dh_val = dWh_dh_fun(Xk[0], Xk[1], k_val, s_val)[0]
            Wh_dot_k = dWh_dx_val * x_dot_k + dWh_dh_val * h_dot_k
            
            h_ddot_k = V_dot_k * ca.sin(Xk[3]) + Xk[2] * ca.cos(Xk[3]) * gamma_dot_k + Wh_dot_k
            
            h_ddot_all[i].append(h_ddot_k)
            
            Xk_end = rk4_step(aircraft_ode, Xk, Uk, tk, dt, k_val, s_val)
            X_next = ca.vertcat(x[j, i+1], h[j, i+1], V[j, i+1], gamma[j, i+1], alpha[j, i+1])
            opti.subject_to(X_next == Xk_end)

            opti.subject_to(opti.bounded(-1, u_s[i], 1))

            alpha_i = alpha_s[:, i] * alphascale
            alpha_pce_coeffs = ca.mtimes(Psi_dm, weights_dm * alpha_i)
            alpha_high_coeffs = alpha_pce_coeffs[1:]
            alpha_mean = ca.dot(weights_dm, alpha_i)
            std_term = ca.sqrt(ca.sumsqr(alpha_high_coeffs))
            
            for i in range(1, N):
                dt = dt_list[i-1]  # Use the time step from the time grid
                tk = time_grid[i]  # Current time from the time grid
                
                h_physical = h_s[:, i] * hscale
                h_mean = ca.dot(weights_dm, h_physical)

                h_pce_coeffs = ca.mtimes(Psi_dm, weights_dm * h_physical)
                h_high_coeffs = h_pce_coeffs[1:]
                h_std_term = ca.sqrt(ca.sumsqr(h_high_coeffs))

                deviation = hR - h_mean + h_std_term * 12
                scaled_deviation = deviation / hscale
                J += dt * scaled_deviation**6

                Xk = ca.vertcat(x[j, i], h[j, i], V[j, i], gamma[j, i], alpha[j, i])
                Uk = u[i]
                
                # h_ddot = d^2h/dt^2 = V_dot * sin(gamma) + V * cos(gamma) * gamma_dot + Wh_dot
                ode_out = aircraft_ode(Xk, Uk, tk, k_val, s_val)
                x_dot_k = ode_out[0]
                h_dot_k = ode_out[1]
                V_dot_k = ode_out[2]
                gamma_dot_k = ode_out[3]
                # alpha_dot_k = ode_out[4]
                
                dWh_dx_val = dWh_dx_fun(Xk[0], Xk[1], k_val, s_val)[0]
                dWh_dh_val = dWh_dh_fun(Xk[0], Xk[1], k_val, s_val)[0]
                Wh_dot_k = dWh_dx_val * x_dot_k + dWh_dh_val * h_dot_k
                
                h_ddot_k = V_dot_k * ca.sin(Xk[3]) + Xk[2] * ca.cos(Xk[3]) * gamma_dot_k + Wh_dot_k
                
                h_ddot_all[i].append(h_ddot_k)
                
                Xk_end = rk4_step(aircraft_ode, Xk, Uk, tk, dt, k_val, s_val)
                X_next = ca.vertcat(x[j, i+1], h[j, i+1], V[j, i+1], gamma[j, i+1], alpha[j, i+1])
                opti.subject_to(X_next == Xk_end)

                opti.subject_to(opti.bounded(-1, u_s[i], 1))

                alpha_i = alpha_s[:, i] * alphascale
                alpha_pce_coeffs = ca.mtimes(Psi_dm, weights_dm * alpha_i)
                alpha_high_coeffs = alpha_pce_coeffs[1:]
                alpha_mean = ca.dot(weights_dm, alpha_i)
                std_term = ca.sqrt(ca.sumsqr(alpha_high_coeffs))

            opti.subject_to(alpha_mean + 3 * std_term <= alphamax)
            opti.subject_to(alpha_mean - 3 * std_term >= -alphamax)
                
        for i in range(N):
            h_ddot_vec = ca.vertcat(*h_ddot_all[i])
            h_ddot_pce_coeffs = ca.mtimes(Psi_dm, weights_dm * h_ddot_vec)
            h_ddot_mean = h_ddot_pce_coeffs[0]
            h_ddot_high_coeffs = h_ddot_pce_coeffs[1:]
            h_ddot_std_term = ca.sqrt(ca.sumsqr(h_ddot_high_coeffs))
            
            opti.subject_to(h_ddot_mean + 3 * h_ddot_std_term <= 10 * g)
            opti.subject_to(h_ddot_mean - 3 * h_ddot_std_term >= -2 * g)

        opti.minimize(J)

        # Initial guess
        for j in range(M):
            opti.set_initial(x_s[j, :], np.linspace(0, 1, N+1))
            opti.set_initial(h_s[j, :], 0.6)  # 600 ft / 1000
            opti.set_initial(V_s[j, :], 239.7 / Vscale)
            opti.set_initial(gamma_s[j, :], -0.01 / gammascale)
            opti.set_initial(alpha_s[j, :], 0.02 / alphascale)
        opti.set_initial(u_s, 0)

        # Solver
        opts = {
            "expand": True,
            "ipopt": {
                # "max_iter": 3000,
                "max_cpu_time": 120,
                "tol": 1e-6,
                "print_level": 0,
                "linear_solver": "mumps",
                "hessian_approximation": "limited-memory",
                # "bound_push": 1e-8,
                # "bound_frac": 1e-8
            }
        }
        opti.solver("ipopt", opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            opti.debug.show_infeasibilities()
            print(e)
            return {
                "x": opti.debug.value(x),
                "h": opti.debug.value(h),
                "V": opti.debug.value(V),
                "gamma": opti.debug.value(gamma),
                "alpha": opti.debug.value(alpha),
                "u": opti.debug.value(u),
                "J": opti.debug.value(J),
                "time_grid": time_grid,
                "dt_list": dt_list,
                "status": "failed"
            }

        return {
            "x": sol.value(x),
            "h": sol.value(h),
            "V": sol.value(V),
            "gamma": sol.value(gamma),
            "alpha": sol.value(alpha),
            "u": sol.value(u),
            "J": sol.value(J),
            "time_grid": time_grid,
            "dt_list": dt_list,
            "status": "success"
        }

    def solve_and_refine(time_grid, max_iter=3):
        refined = True
        iter_count = 0
        while refined and iter_count <= max_iter:
            sol = solve_ocp(time_grid)
            
            # Check if further refinement is needed
            refined, new_time_grid = refine_time_grid(sol["u"], time_grid, du_tol)
            if refined and iter_count < max_iter:
                time_grid = new_time_grid
                iter_count += 1
                print(f"Refining time grid, iteration {iter_count}")
            else:
                break
            
        return sol
    
    return solve_and_refine(time_grid, max_iter=max_iter)

res_double_measure_Bayes = solve_wider_wind_ocp_pce_w_adaptive_mesh_double_measure(
    pce_order=2,
    k_mean=1.0,
    k_std=0.08,
    s_mean=1.0,
    s_std=0.15,
    k_obs=1.05,
    obs_noise=0.05,
    obs_time=30,
    du_tol=1e-3,
    max_iter=3,
)

res_double_measure = solve_wider_wind_ocp_pce_w_adaptive_mesh_double_measure(
    pce_order=2,
    k_mean=1.0,
    k_std=0.08,
    s_mean=1.0,
    s_std=0.15,
    k_obs=1.05,
    obs_noise=0.05,
    obs_time=30,
    du_tol=1e-3,
    max_iter=3,
    Bayesian_update=False,
)

res_adpt_u = solve_wider_wind_ocp_pce_w_adaptive_mesh(pce_order=2, k_mean=1.0, k_std=0.08, s_mean=1.0, s_std=0.15, max_iter=3, du_tol=1e-3)

plt.figure(figsize=(18, 6))
plt.step(res_double_measure_Bayes["time_grid"], res_double_measure_Bayes["u"], color='tab:blue', label='Optimal u with Bayesian Update', linewidth=2)
plt.plot(res_double_measure_Bayes["time_grid"], -0.06*np.ones_like(res_double_measure_Bayes["time_grid"]), '.', color='tab:blue', markersize=3)
plt.step(res_double_measure["time_grid"], res_double_measure["u"], color='tab:orange', label='Optimal u without Bayesian Update', linewidth=2)
plt.plot(res_double_measure["time_grid"], -0.062*np.ones_like(res_double_measure["time_grid"]), '.', color='tab:orange', markersize=3)
plt.step(res_adpt_u["time_grid"], res_adpt_u["u"], color='tab:green', label='Adaptive Control u (No Update)', linewidth=2)
plt.plot(res_adpt_u["time_grid"], -0.064*np.ones_like(res_adpt_u["time_grid"]), '.', color='tab:green', markersize=3)
plt.xlabel('Time [s]')
plt.ylabel('u (control)')
# plt.title('Comparison of Adaptive Control $u$ with and without Bayesian Update')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def plot_x_h_pce_fixed_k_w_update(res, pce_order, k_fix, k_obs, obs_noise, obs_time, k_mean=1.0, k_std=0.08, s_mean=1.0, s_std=0.1, Bayesian_update=True):
    
    def bayesian_update_pce(k_obs=k_obs, obs_noise=obs_noise, k_mean=k_mean, k_std=k_std):
        var_prior = k_std**2
        var_obs = obs_noise**2
        
        var_posterior = 1 / (1 / var_prior + 1 / var_obs)
        mean_posterior = var_posterior * (k_obs / var_obs + k_mean / var_prior)
        
        return mean_posterior, np.sqrt(var_posterior)
    
    if Bayesian_update:
        k_posterior, k_std_posterior = bayesian_update_pce()
        s_mean = 1.0
        s_std = 0.15
    else:
        k_posterior = 1.1
        k_std_posterior = 0.08
        s_mean = 0.8
        s_std = 0.1
        
    idx_obs = np.argmin(np.abs(res["time_grid"] - obs_time))
    
    # sol_1
    k_dist = cp.Normal(k_mean, k_std)
    s_dist = cp.Normal(s_mean, s_std)
    joint_dist = cp.J(k_dist, s_dist)
    
    poly_basis = cp.expansion.stieltjes(pce_order, joint_dist)
    nodes, weights = cp.generate_quadrature(pce_order, joint_dist, rule='G')
    
    x_before = res['x'][:, :idx_obs]
    h_before = res['h'][:, :idx_obs]
    
    s_fix = (1/k_fix) ** 2
    
    coeffs_x_before = poly_basis(nodes[0], nodes[1]) @ (x_before * weights[:, None])
    coeffs_h_before = poly_basis(nodes[0], nodes[1]) @ (h_before * weights[:, None])
    pce_x_before = poly_basis(k_fix, s_fix) @ coeffs_x_before
    pce_h_before = poly_basis(k_fix, s_fix) @ coeffs_h_before
    
    # sol_2
    k_dist = cp.Normal(k_posterior, k_std_posterior)
    s_dist = cp.Normal(s_mean, s_std)
    joint_dist = cp.J(k_dist, s_dist)
    
    poly_basis = cp.expansion.stieltjes(pce_order, joint_dist)
    nodes, weights = cp.generate_quadrature(pce_order, joint_dist, rule='G')
    
    x_after = res['x'][:, idx_obs:]
    h_after = res['h'][:, idx_obs:]
    
    coeffs_x_after = poly_basis(nodes[0], nodes[1]) @ (x_after * weights[:, None])
    coeffs_h_after = poly_basis(nodes[0], nodes[1]) @ (h_after * weights[:, None])
    pce_x_after = poly_basis(k_fix, s_fix) @ coeffs_x_after
    pce_h_after = poly_basis(k_fix, s_fix) @ coeffs_h_after
    
    pce_x = np.concatenate([pce_x_before, pce_x_after])
    pce_h = np.concatenate([pce_h_before, pce_h_after])
    
    plt.plot(pce_x, pce_h, label='PCE Approximation for Order {}'.format(pce_order), linewidth=2)
    
def plot_x_h_pce_fixed_k(res, pce_order, k_fix, k_mean=1.0, k_std=0.08, s_mean=1.0, s_std=0.1):
    k_dist = cp.Normal(k_mean, k_std)
    s_dist = cp.Normal(s_mean, s_std)
    joint_dist = cp.J(k_dist, s_dist)
    
    poly_basis = cp.expansion.stieltjes(pce_order, joint_dist)
    nodes, weights = cp.generate_quadrature(pce_order, joint_dist, rule='G')
    
    x = res['x']
    h = res['h']
    
    s_fix = (1/k_fix) ** 2
    
    coeffs_x = poly_basis(nodes[0], nodes[1]) @ (x * weights[:, None])
    coeffs_h = poly_basis(nodes[0], nodes[1]) @ (h * weights[:, None])
    pce_x = poly_basis(k_fix, s_fix) @ coeffs_x
    pce_h = poly_basis(k_fix, s_fix) @ coeffs_h
    
    plt.plot(pce_x, pce_h, label='PCE Approximation for Order {}'.format(pce_order), linewidth=2)
    
plt.figure(figsize=(18, 6))
plot_x_h_pce_fixed_k_w_update(res_double_measure_Bayes, Bayesian_update=True, pce_order=2, k_fix=1.0, k_mean=1.0, k_std=0.08, k_obs=1.05, obs_noise=0.05, obs_time=30, s_mean=1.0, s_std=0.15)
plot_x_h_pce_fixed_k_w_update(res_double_measure, Bayesian_update=False, pce_order=2, k_fix=1.0, k_mean=1.0, k_std=0.08, k_obs=1.05, obs_noise=0.05, obs_time=30, s_mean=1.0, s_std=0.15)
plot_x_h_pce_fixed_k(res_adpt_u, pce_order=2, k_fix=1.0, k_mean=1.0, k_std=0.08, s_mean=1.0, s_std=0.15)
plt.xlabel('Horizontal Position $x$ [ft]')
plt.ylabel('Altitude $h$ [ft]')
# plt.title('Adaptive Mesh Variation with Fixed $k$ (Double Measure)')
plt.grid(True)
plt.legend(["Without Update", "Bayesian Update", "No Update"])
plt.show()