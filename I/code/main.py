######## Imports #############################################################
from pathlib import Path
import time
from timeit import repeat
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.sparse.linalg import svds


from helpers import phi, phi_prime, f_lambda, grad_f, f_lambda_loop, grad_f_loop, algo, plot_convergence, predict, classif_error_rate



def main():

    ### LOADING DATA -------------------------------------------------------------------
    directory = Path(__file__).resolve().parent.parent
    data = loadmat(directory / "data/mnist_train_test.mat", squeeze_me=True, struct_as_record=False)

    # Training Dataset
    X_train = np.asarray(data["train"].X, dtype=float)                # TRAIN SET INPUTS -- shape: (D+1, M_train) :  (785, 12665) 
    y_train = np.asarray(data["train"].y, dtype=float).reshape(-1)    # TRAIN SET LABELS -- shape: (M_train, )    :  (12665, )

    # Test Set Data
    X_test = np.asarray(data["test"].X, dtype=float)                  # TEST SET INPUTS
    y_test = np.asarray(data["test"].y, dtype=float).reshape(-1)      # TEST SET LABELS


    ###  Setup Results Directory ------------------------------------------------------
    results = directory / "results"
    results.mkdir(parents=True, exist_ok=True)

    ### CONSTANTS ---------------------------------------------------------------------
    lam = 0.005                     # ridge parameter value
    rtol, atol = 1e-10, 1e-10       # relative/absolute tolerance



    ################ QUESTION 2 --- agreement checks and timings ######################
    #
    #           Compare both implementations on exactly the same inputs.

    rng   = np.random.default_rng(329)
    theta = rng.standard_normal(X_train.shape[0])

    # Implementation using 'for' loop
    f_loop    = f_lambda_loop(theta, X_train, y_train, lam)
    grad_loop = grad_f_loop(theta, X_train, y_train, lam)

    # Implementation using vectorization
    f     = f_lambda(theta, X_train, y_train, lam)
    grad  = grad_f(theta, X_train, y_train, lam)

    # Compare the objective and every gradient component with both tolerances.
    np.testing.assert_allclose(f_loop, f, rtol=rtol, atol=atol)
    np.testing.assert_allclose(grad_loop, grad, rtol=rtol, atol=atol)

    q2_output = [
            "Question 2: objective and gradient comparison",
            f"Samples: {X_train.shape[1]}, parameters: {X_train.shape[0]}, lambda: {lam}, seed: 329",
            f"Agreement checks passed (rtol={rtol:g}, atol={atol:g}).",
            f"Objective (loop): {f_loop:.18e}",
            f"Objective (vectorized): {f:.18e}",
            f"Objective absolute difference: {abs(f_loop - f):.3e}",
            f"Gradient maximum absolute difference: {np.max(np.abs(grad_loop - grad)):.3e}",
            "Timings: 30 trials; best of 3 single evaluations per implementation per trial, using timeit.repeat on identical inputs.",
        ]
    
    # Each timing is one evaluation, taking the best of three repetitions.
    # Objective and gradient timings use identical theta, X_train, y_train and lam.
    timings = []
    for trial in range(1, 31):
        row = [trial]
        for loop, vectorized in (
            (f_lambda_loop, f_lambda),
            (grad_f_loop, grad_f),
        ):
            loop_time = min(repeat(lambda: loop(theta, X_train, y_train, lam), number=1, repeat=3))
            vector_time = min(repeat(lambda: vectorized(theta, X_train, y_train, lam), number=1, repeat=3))
            row.extend([loop_time, vector_time, loop_time / vector_time])
        timings.append(row)

    timings = np.asarray(timings)
    np.savetxt(
        results / "q2_repeated_timings.csv", timings, delimiter=",",
        header=("trial,objective_loop_seconds,objective_vector_seconds,"
                "objective_speedup,gradient_loop_seconds,"
                "gradient_vector_seconds,gradient_speedup"),
        comments="",
    )
    for name, column in (("f_lambda", 3), ("grad_f", 6)):
        speedups = timings[:, column]
        q2_output.append(
            f"{name}: minimum speedup={speedups.min():.2f}x, "
            f"median speedup={np.median(speedups):.2f}x across 30 trials"
        )

    # Keep the agreement checks and runtime measurements with the other results.
    q2_report = "\n".join(q2_output) + "\n"
    (results / "q2_comparison.txt").write_text(q2_report, encoding="utf-8")
    print(q2_report, end="")



    ################ QUESTION 3: gradient check #######################################
    #
    #       Taylor remainder along a reproducible random unit direction.

    rng     = np.random.default_rng(1)
    theta   = rng.standard_normal(X_train.shape[0])

    v  =  rng.standard_normal(X_train.shape[0])
    v  /= np.linalg.norm(v)
    t  =  np.logspace(-8.0, 0.0, num=101)
    f0 =  f_lambda(theta, X_train, y_train, lam)

    dir_deriv = v @ grad_f(theta, X_train, y_train, lam)

    err = np.array([abs(f_lambda(theta + h * v, X_train, y_train, lam) - f0 - h * dir_deriv) for h in t])

    np.savetxt(results / "q3_gradient_check.txt", np.column_stack((t, err)), header="t absolute_taylor_remainder")


    # PLOTTING SECTION  ---------------------------------------------------
    
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

    # ----------------------------------------------------------------------


    print(f"Q3: saved plot and numerical data in {results}")

    # A slope near 2 is consistent with an O(t^2) remainder. This check
    # supports correctness without proving it. cancellation and roundoff
    # can dominate for very small t.


    ################ Question 4: step size and GD run #################################
    #
    #   L bounds the gradient's Lipschitz constant for the summed loss.
    #   A fixed step of 1/L gives the standard descent guarantee.
    #   Fix the SVD starting vector independently of the random GD initial point.

    sigma_max = svds(X_train, k=1, return_singular_vectors=False,v0=np.random.default_rng(40).standard_normal(min(X_train.shape)))[0]
    
    L = sigma_max**2 + lam
    print("L:", L)
    step_size = 1/L

    theta_final = algo(step_size, X_train, y_train, lam, results)


    ################ Question 5: plot the recorded GD run #############################    
    
    plot_convergence(results)


    ################ Question 7: use optimal model parameters to make prediction on test data and compute train/test classif. error rates ##########################################

    # Model Evaluation on Training Set
    y_pred_train = predict(X_train, theta_final)
    train_classif_error_rate = classif_error_rate(y_train, y_pred_train)

    # Model Evaluation on Test Set
    y_pred_test = predict(X_test, theta_final)
    test_classif_error_rate = classif_error_rate(y_test, y_pred_test)

    # Record of Error Rates saved in 'results/q7_classif_error_rates.txt' file
    (results / "q7_classif_error_rates.txt").write_text(
        f"classif. error rate (train-set): {train_classif_error_rate}\n"
        f"classif. error rate (test-set):  {test_classif_error_rate}\n",
        encoding="utf-8",
    )

    print("Q7 -- classif. error rate")
    print(f" -- train-set : {train_classif_error_rate}")
    print(f" -- test-set  : {test_classif_error_rate}\n")






################ Script entry point ###################################################
if __name__ == "__main__":
    main()
