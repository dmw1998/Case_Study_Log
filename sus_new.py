import numpy as np
import multiprocessing
import os
import time
from tqdm import tqdm
from numba import jit

# === ENVIRONMENT SETUP ===
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# === CONFIGURATION ===
SHOW_PROGRESS = True
n_runs = 100
num_samples = 1000
p0 = 0.1
# k_values = np.linspace(0.95, 1.05, 6)
k_values = np.array([1.01, 1.03, 1.05])  # Adjusted for better coverage
# rhos = [0.12, 0.1, 0.08, 0.07, 0.07, 0.06]
rhos = [0.06, 0.06, 0.06]  # Adjusted for better coverage

# === STRATEGY CONFIGURATION ===
res_no_adpt = np.load("res_no_adpt.npy", allow_pickle=True).item()
res_adpt_u_3 = np.load("res_adpt_u_3.npy", allow_pickle=True).item()
res_double_measure = np.load("res_double_measure.npy", allow_pickle=True).item()
res_double_measure_Bayes = np.load("res_double_measure_Bayes.npy", allow_pickle=True).item()

strategy_configs = [
    ("No adaptation", res_no_adpt),
    ("Adaptation u_3", res_adpt_u_3),
    ("Double measure", res_double_measure),
    ("Double measure Bayesian", res_double_measure_Bayes),
]

# === LIMIT STATE FUNCTION ===
@jit(nopython=True, cache=False)
def Smooth(x_, x0, x1):
    eps = 1e-6
    t = (x_ - x0) / (x1 - x0 + eps)
    if x_ < x0:
        return 0.0
    elif x_ > x1:
        return 1.0
    else:
        return 6*t**5 - 15*t**4 + 10*t**3

@jit(nopython=True, cache=False)
def A_piecewise(x_, s_):
    a = 6e-8
    b = -4e-11
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

@jit(nopython=True, cache=False)
def B_piecewise(x_, s_):
    c = -np.log(25/30.6)*1e-12
    d = -8.02881e-8
    e = 6.28083e-11
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

def limit_state_function(args):
    u_opt, time_grid, k, s = args
    m, g, delta = 4662, 32.172, 0.03491
    A0, A1, A2 = 0.4456e5, -0.2398e2, 0.1442e-1
    rho, S = 0.2203e-2, 0.1560e4
    beta0, beta_dot0, sigma = 0.4, 0.2, 3
    B0, B1, B2 = 0.1552, 0.12369, 2.4203
    C0, C1, C2 = 0.7125, 6.0877, -9.0277
    alpha_star = 0.20944
    h_star = 1000

    def wind_x(x_, k_, s_): return k_ * A_piecewise(x_, s_)
    def wind_h(x_, h_, k_, s_): return k_ * np.fmax(h_, 10.0) / h_star * B_piecewise(x_, s_)
    def originalWindModel(x_, h_, k_, s_): return wind_x(x_, k_, s_), wind_h(x_, h_, k_, s_)

    def C_L(alpha_):
        return C0 + C1 * alpha_ + (C2 * (alpha_ - alpha_star)**2 if alpha_ <= alpha_star else 0)
    def beta(t_): return beta0 + beta_dot0 * t_ if t_ < sigma else 1.0

    def aircraft_ode_func(state, u_, t_, k_value):
        s_value = (1.0 / k_value) ** 2
        x_, h_, V_, gamma_, alpha_ = state
        T = beta(t_) * (A0 + A1 * V_ + A2 * V_**2)
        D = 0.5 * (B0 + B1 * alpha_ + B2 * alpha_**2) * rho * S * V_**2
        L = 0.5 * rho * S * C_L(alpha_) * V_**2
        eps_grad = 1e-6
        Wx_c, Wh_c = originalWindModel(x_, h_, k_value, s_value)
        Wx_px, Wh_px = originalWindModel(x_ + eps_grad, h_, k_value, s_value)
        Wx_mx, Wh_mx = originalWindModel(x_ - eps_grad, h_, k_value, s_value)
        _, Wh_ph = originalWindModel(x_, h_ + eps_grad, k_value, s_value)
        _, Wh_mh = originalWindModel(x_, h_ - eps_grad, k_value, s_value)
        Wx_dx = (Wx_px - Wx_mx) / (2 * eps_grad)
        Wh_dx = (Wh_px - Wh_mx) / (2 * eps_grad)
        Wh_dh = (Wh_ph - Wh_mh) / (2 * eps_grad)
        V_safe = np.fmax(V_, 1e-3)
        cos_gamma = np.cos(gamma_)
        sin_gamma = np.sin(gamma_)
        cos_alpha_delta = np.cos(alpha_ + delta)
        sin_alpha_delta = np.sin(alpha_ + delta)
        x_dot = V_ * cos_gamma + Wx_c
        h_dot = V_ * sin_gamma + Wh_c
        Wx_dot = Wx_dx * x_dot
        Wh_dot = Wh_dx * x_dot + Wh_dh * h_dot
        V_dot = T / m * cos_alpha_delta - D / m - g * sin_gamma - (Wx_dot * cos_gamma + Wh_dot * sin_gamma)
        gamma_dot = T / (m * V_safe) * sin_alpha_delta + L / (m * V_safe) - g / V_safe * cos_gamma + (Wx_dot * sin_gamma - Wh_dot * cos_gamma) / V_safe
        alpha_dot = u_
        return np.array([x_dot, h_dot, V_dot, gamma_dot, alpha_dot])

    def rk4_step(state, u, t, dt, k_value):
        k1 = aircraft_ode_func(state, u, t, k_value)
        k2 = aircraft_ode_func(state + dt/2 * k1, u, t + dt/2, k_value)
        k3 = aircraft_ode_func(state + dt/2 * k2, u, t + dt/2, k_value)
        k4 = aircraft_ode_func(state + dt * k3, u, t + dt, k_value)
        return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    def reconstruct_trajectory():
        state = np.array([0, 600, 239.7, -0.03925, 0.1283])
        dt_ls = np.diff(time_grid)
        h_min = 600
        for i, u in enumerate(u_opt[1:]):
            t = time_grid[i]
            dt = dt_ls[i]
            h_min = min(h_min, state[1])
            state = rk4_step(state, u, t, dt, k)
        return h_min

    return reconstruct_trajectory()

# === SUBSET SIMULATION CORE ===
def subset_simulation(N, p0, rho, u_opt, time_grid, k_mean, seed):
    np.random.seed(seed)
    num_seeds = int(N * p0)
    l = 0
    k = np.random.normal(k_mean, 0.08, N)
    s = (1.0 / k)**2
    G = np.fromiter((limit_state_function((u_opt, time_grid, k[i], s[i])) for i in range(N)), dtype=np.float64, count=N)
    threshold = np.percentile(G, 100 * p0)
    k = k[G <= threshold][:num_seeds]
    G = G[G <= threshold][:num_seeds]

    while threshold > 0.0:
        l += 1
        new_samples = N - num_seeds
        acc = 0
        for i in range(new_samples):
            k_new = rho * k[i] + np.sqrt(1 - rho**2) * np.random.normal(k_mean, 0.08)
            s_new = (1.0 / k_new)**2
            G_new = limit_state_function((u_opt, time_grid, k_new, s_new))
            if G_new <= threshold:
                k = np.append(k, k_new)
                G = np.append(G, G_new)
                acc += 1
            else:
                k = np.append(k, k[i])
                G = np.append(G, G[i])

        threshold = np.percentile(G, 100 * p0)
        # if acc / new_samples < 0.2 or acc / new_samples > 0.5:
        #     print(f"Threshold reached at iteration {l}: {threshold:.6f}, accepted {acc} new samples.")
        if threshold <= 0.0:
            return p0 ** (l-1) * np.mean(G <= 0.0)
        k = k[G <= threshold][:num_seeds]
        G = G[G <= threshold][:num_seeds]
        rho = rho * (1.5 + 3 * (k_mean - 0.94) + (l-1) * 0.1)

    return p0 ** l

def run_subset_simulation(args):
    return subset_simulation(*args)

# === MAIN LOOP ===
if __name__ == "__main__":
    np.random.seed(44)  # For reproducibility
    start_time = time.time()
    results_by_strategy = {}

    for i_k, k in enumerate(tqdm(k_values, desc="Overall Progress", unit="k_value")):
        print(f"\n{'='*60}")
        print(f"Running simulations for k_mean = {k:.3f} ({i_k+1}/{len(k_values)})")
        print(f"{'='*60}")

        for strategy_name, res in strategy_configs:
            print(f"\nRunning strategy: {strategy_name}")
            args_list = [
                (num_samples, p0, rhos[i_k], res['u'], res['time_grid'], k, np.random.randint(0, 100000))
                for _ in range(n_runs)
            ]

            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                results = list(tqdm(
                    pool.imap_unordered(run_subset_simulation, args_list),
                    total=n_runs,
                    desc=strategy_name,
                    leave=False
                ))

            mean_p = np.mean(results)
            std_p = np.std(results)
            print(f"  -> Result: {mean_p:.6e} ± {std_p:.1e}")
            results_by_strategy.setdefault(strategy_name, []).append((k, mean_p, std_p))

    total_time = time.time() - start_time
    print(f"\nAll simulations completed in {total_time / 60:.1f} minutes.")
    np.save("subset_simulation_results_3.npy", results_by_strategy)
