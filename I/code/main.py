######## Imports #############################################################
from pathlib import Path
import time
from timeit import repeat
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.sparse.linalg import svds

############ Question 2: objective and gradient #############################
# loop versions.
def phi(z):
    if z <= -1:
        return 0.0
    elif z <=0:
        return 0.5 * (1 + z)**2
    return 0.5 + z

def phi_prime(z):
    if z <= -1:
        return 0.0
    elif z <= 0:
        return 1 + z
    return 1.0

def f_lambda_loop(theta, X, y, lam):
    f = 0.0
    for i in range(X.shape[1]):
        z_i = (1 - 2 * y[i]) * (X[:, i] @ theta)
        f += phi(z_i)
    return f + (lam / 2) * (theta @ theta)

def grad_f_loop(theta, X, y, lam):
    grad = np.zeros_like(theta, dtype=float)
    for i in range(X.shape[1]):
        s_i = 1 -2 * y[i]
        z_i = s_i * (X[:, i] @ theta)
        grad += s_i * phi_prime(z_i) * X[:, i]
    return grad + lam * theta

# vectorized versions
def f_lambda(theta, X, y, lam):
    s = 1 - 2 * y
    z = s * (X.T @ theta)
    d = np.clip(1 + z,0,1)
    return np.sum(0.5 * d**2 + np.maximum(z, 0)) + (lam / 2) * (theta @ theta)      

# Matrix-vector form of the sum of individual gradient contributions.
def grad_f(theta, X, y, lam):
    s = 1 - 2 * y
    z = s * (X.T @ theta)
    d = np.clip(1 + z, 0, 1)
    return X @ (s * d) + lam * theta

######## Question 4: gradient descent implementation ###################################
# Fixed-step gradient descent from a reproducible initial point.
def algo(step, X, y, lam, results, time_limit=180):
    rng = np.random.default_rng(40)
    theta_0 = rng.standard_normal(X.shape[0])
    x = theta_0.copy()
    start = time.perf_counter()
    g = grad_f(x, X, y, lam)
    g_norm = np.linalg.norm(g)
    tolerance = 1e-3 * g_norm
    l1 = [f_lambda(x, X, y, lam)]
    l2 = [g_norm]
    print("initial:", g_norm)
    print("target:", tolerance)
    while True:
        if not np.isfinite(l1[-1]) or not np.isfinite(g_norm):
            stop_reason = "non finite value"
            break
        if g_norm <= tolerance:
            stop_reason = "gradient tolerance"
            break
        if time.perf_counter() - start >= time_limit:
            stop_reason = "time limit"
            break
        x = x - step * g
        g = grad_f(x, X, y, lam)
        g_norm = np.linalg.norm(g)
        l1.append(f_lambda(x, X, y, lam))
        l2.append(g_norm)

    elapsed = time.perf_counter() - start
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    np.savetxt(results / "q4_history.txt",np.column_stack((np.arange(len(l1)), l1, l2)),fmt=("%d", "%.18e", "%.18e"),header="iteration objective gradient_norm")
    np.savetxt(results / "q4_theta_final.txt", x)
    (results / "q4_stopping.txt").write_text(
        f"stop_reason: {stop_reason}\n"
        f"iterations: {len(l1) - 1}\n"
        f"elapsed_seconds: {elapsed:.6f}\n"
        f"time_limit_seconds: {time_limit}\n"
        f"step: {step:.18e}\n"
        f"lambda: {lam}\n"
        f"seed: 40\n"
        f"initial_gradient_norm: {l2[0]:.18e}\n"
        f"target_gradient_norm: {tolerance:.18e}\n"
        f"final_gradient_norm: {g_norm:.18e}\n",
        encoding="utf-8",
    )
    print("final:", g_norm)
    print(f"Q4: {stop_reason}, {len(l1) - 1} iterations, {elapsed:.2f}s")
    return x

############# Question 5: convergence plots ##########################################
def plot_convergence(results):
    results = Path(results)
    history = np.loadtxt(results / "q4_history.txt", ndmin=2)
    k, l1, l2 = history.T

    # We chose logarithmic vertical axes to show the decrease over several orders of
    # magnitude while keeping the behavior at later iterations visible.
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].semilogy(k, l1, label="objective")
    axes[0].set_ylabel(r"$f_\lambda(\theta_k)$")
    axes[0].set_title("Objective value")
    axes[1].semilogy(k, l2, label="gradient norm")
    axes[1].axhline(1e-3 * l2[0], color="tab:red", linestyle="--",
                    label="stopping tolerance")
    axes[1].set_ylabel(r"$\|\nabla f_\lambda(\theta_k)\|$")
    axes[1].set_xlabel("Iteration k")
    axes[1].set_title("Gradient norm")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(results / "q5_convergence.pdf")
    plt.close(fig)
    print(f"Q5: saved convergence plots in {results / 'q5_convergence.pdf'}")


####### Main #######################################################################
def main():
    directory = Path(__file__).resolve().parent.parent
    data = loadmat(directory / "data/mnist_train_test.mat", squeeze_me=True, struct_as_record=False)
    X = np.asarray(data["train"].X, dtype=float)
    y = np.asarray(data["train"].y, dtype=float).reshape(-1)
    lam = 0.005
    results = directory / "results"
    results.mkdir(parents=True, exist_ok=True)

    ########### Question 2: agreement checks and timings ###########################
    # Compare both implementations on exactly the same inputs.
    rng = np.random.default_rng(329)
    theta = rng.standard_normal(X.shape[0])
    f_loop = f_lambda_loop(theta, X, y, lam)
    f = f_lambda(theta, X, y, lam)
    grad_loop = grad_f_loop(theta, X, y, lam)
    grad = grad_f(theta, X, y, lam)

    # Compare the objective and every gradient component with both tolerances.
    rtol, atol = 1e-10, 1e-10
    np.testing.assert_allclose(f_loop, f, rtol=rtol, atol=atol)
    np.testing.assert_allclose(grad_loop, grad, rtol=rtol, atol=atol)
    q2_output = [
        "Question 2: objective and gradient comparison",
        f"Samples: {X.shape[1]}, parameters: {X.shape[0]}, lambda: {lam}, seed: 329",
        f"Agreement checks passed (rtol={rtol:g}, atol={atol:g}).",
        f"Objective (loop): {f_loop:.18e}",
        f"Objective (vectorized): {f:.18e}",
        f"Objective absolute difference: {abs(f_loop - f):.3e}",
        f"Gradient maximum absolute difference: {np.max(np.abs(grad_loop - grad)):.3e}",
        "Timings: best of 3 single evaluations on identical inputs, using timeit.repeat.",
    ]

    # Each timing is one evaluation, taking the best of three repetitions.
    # Objective and gradient timings use identical theta, X, y and lam.
    for name, loop, vectorized in (
        ("f_lambda", f_lambda_loop, f_lambda),
        ("grad_f", grad_f_loop, grad_f),
    ):
        loop_time = min(repeat(lambda: loop(theta, X, y, lam), number=1, repeat=3))
        vector_time = min(repeat(lambda: vectorized(theta, X, y, lam), number=1, repeat=3))
        q2_output.append(
            f"{name}: loop={loop_time:.6f}s, vectorized={vector_time:.6f}s, "
            f"speedup={loop_time / vector_time:.2f}x (best of 3)"
        )

    # Keep the agreement checks and runtime measurements with the other results.
    q2_report = "\n".join(q2_output) + "\n"
    (results / "q2_comparison.txt").write_text(q2_report, encoding="utf-8")
    print(q2_report, end="")

    ################ Question 3: gradient check #########################################
    # Taylor remainder along a reproducible random unit direction.
    rng = np.random.default_rng(1)
    theta = rng.standard_normal(X.shape[0])
    v = rng.standard_normal(X.shape[0])
    v /= np.linalg.norm(v)
    t = np.logspace(-8.0, 0.0, num=101)
    f0 = f_lambda(theta, X, y, lam)
    dir_deriv = v @ grad_f(theta, X, y, lam)
    # Difference between the actual value and its first-order approximation.
    err = np.array([abs(f_lambda(theta + h * v, X, y, lam) - f0 - h * dir_deriv) for h in t])
    # Save the step sizes and corresponding remainders as two columns.
    np.savetxt(results / "q3_gradient_check.txt", np.column_stack((t, err)), header="t absolute_taylor_remainder")

    fig, ax = plt.subplots()
    ax.loglog(t, err, label="Taylor remainder")
    # Add a t^2 reference at the last positive remainder. Its log-log slope
    # is 2. the vertical scale is chosen only for comparison with the curve.
    positive = np.flatnonzero(err > 0)
    if positive.size:
        j = positive[-1]
        ax.loglog(t, err[j] * (t / t[j])**2, "--", label="Slope 2 reference")
    ax.set_xlabel("t")
    ax.set_ylabel("Absolute Taylor remainder")
    ax.set_title("Q3: gradient check")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / "q3_gradient_check.pdf")
    plt.close(fig)
    print(f"Q3: saved plot and numerical data in {results}")
    # A slope near 2 is consistent with an O(t^2) remainder. This check
    # supports correctness without proving it. cancellation and roundoff
    # can dominate for very small t.

    ################ Question 4: step size and GD run ##########################################
    # L bounds the gradient's Lipschitz constant for the summed loss.
    # A fixed step of 1/L gives the standard descent guarantee.
    # Fix the SVD starting vector independently of the random GD initial point.
    sigma_max = svds(X, k=1, return_singular_vectors=False,v0=np.random.default_rng(40).standard_normal(min(X.shape)))[0]
    L = sigma_max**2 + lam
    print("L:", L)
    theta_final = algo(1 / L, X, y, lam, results)

    ################ Question 5: plot the recorded GD run ##########################################
    plot_convergence(results)


################ Script entry point ################################################
if __name__ == "__main__":
    main()
