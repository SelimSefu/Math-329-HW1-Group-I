# Questions 2, 3, 4 and 5

Run `python I/code/main.py` from the repository root (or use the script's
absolute path from any working directory). Requires NumPy, SciPy and Matplotlib,
and `I/data/mnist_train_test.mat`.

If using the workspace-local dependencies installed during verification, run
the following from the repository root:

```powershell
python -c "import sys, runpy; sys.path.append('.python-deps'); runpy.run_path('I/code/main.py', run_name='__main__')"
```

Question 2 prints and saves to `results/q2_comparison.txt` the loop/vectorized
agreement checks with relative and absolute
tolerances of `1e-10`, plus separate objective and gradient timings on identical
inputs (best of three single evaluations).

Question 3 saves `results/q3_gradient_check.pdf` and
`results/q3_gradient_check.txt` relative to this directory. The text file contains
two columns: step size `t` and absolute first-order Taylor remainder. The script
uses seed 1 for the point and unit direction. Compare the straight portion of
the plotted curve with the slope-2 reference; very small steps may show roundoff.

Question 4 runs fixed-step gradient descent with step `1/L`, where
`L = sigma_max(X)**2 + lambda`, and a random initial point generated with seed 40.
It stops at a gradient norm at most `1e-3` times the initial norm or after 180
seconds. The time budget covers initialization and iterations, excluding the
SVD and file saving. Time is checked between iterations, so a running iteration
can finish just after the limit. A non-finite objective or gradient norm is
recorded as a numerical failure, not convergence.

- `results/q4_history.txt`: iteration, objective value, gradient norm, including
  iteration 0 and the final iterate.
- `results/q4_stopping.txt`: stopping reason, iteration count, elapsed time,
  step size, seed, regularization parameter and gradient tolerance details.
- `results/q4_theta_final.txt`: final parameter vector.

Each run replaces these output files. The fixed initial point is reproducible;
the number of iterations reached under the time limit depends on the machine.

Question 5 reads `results/q4_history.txt` and saves two separate panels in
`results/q5_convergence.pdf`: objective value and gradient norm versus iteration,
including iteration 0. Both vertical axes are logarithmic because the quantities
decrease over several orders of magnitude; this also makes the later behavior
visible. The objective plot shows the value itself, not the optimality gap.
The gradient plot includes the stopping tolerance as a dashed line.
