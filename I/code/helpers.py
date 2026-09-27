######## Imports #############################################################
from pathlib import Path
import time
from timeit import repeat
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.sparse.linalg import svds




  
############ Question 2: objective and gradient #############################

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


# loop versions ---------------------------------------------

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


# vectorized versions --------------------------------------

def f_lambda(theta, X, y, lam):
    s = 1 - 2 * y
    z = s * (X.T @ theta)
    d = np.clip(1 + z,0,1)
    return np.sum(0.5 * d**2 + np.maximum(z, 0)) + (lam / 2) * (theta @ theta)      

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
    theta = theta_0.copy()

    start = time.perf_counter()

    g = grad_f(theta, X, y, lam)
    g_norm = np.linalg.norm(g)

    tolerance = 1e-3 * g_norm

    l1 = [f_lambda(theta, X, y, lam)]
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
        theta = theta - step * g
        g = grad_f(theta, X, y, lam)
        g_norm = np.linalg.norm(g)
        l1.append(f_lambda(theta, X, y, lam))
        l2.append(g_norm)

    elapsed = time.perf_counter() - start

    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)

    np.savetxt(results / "q4_history.txt",np.column_stack((np.arange(len(l1)), l1, l2)),fmt=("%d", "%.18e", "%.18e"),header="iteration objective gradient_norm")
    np.savetxt(results / "q4_theta_final.txt", theta)

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

    return theta



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



############# Question 7: prediction using GD output model parameters ################

def predict(X, theta_final):
    """
    Runs Prediction on given data using optimal parameters.

    Arguments:
        X           (np.array) : data of shape (D+1, M)
        theta_final (np.array) : optimal model parameter of shape (D+1,)
    
    Returns:
        pred_labels (np.array) : label predictions of shape (M,)
    """

    y_preds = X.T @ theta_final

    mask = y_preds > 0

    pred_labels = np.zeros_like(y_preds)

    pred_labels[mask] = 1


    return pred_labels



def classif_error_rate(y_targets, y_preds):
    """
    Computes the classification error rate.

    Arguments:
        y_targets   (np.array)  : true labels       -- shape (M, )
        y_preds     (np.array)  : predicted labels  -- shape (M, )
    
    Returns:
        error_rate  (float)     : classification error rate
    """

    #M = np.shape(y_targets)[0]

    return np.mean(y_targets != y_preds)