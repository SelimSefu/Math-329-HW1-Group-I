from pathlib import Path
from timeit import repeat

import numpy as np
from scipy.io import loadmat


def objective_gradient_loop(theta, X, y, lam):
    value = 0.0
    gradient = np.zeros_like(theta)

    for i in range(X.shape[1]):
        x_i = X[:, i]
        s_i = 1 - 2 * y[i]
        z = s_i * np.dot(x_i, theta)

        if z <= -1:
            loss = 0.0
            derivative = 0.0
        elif z < 0:
            loss = 0.5 * (1 + z) ** 2
            derivative = 1 + z
        else:
            loss = 0.5 + z
            derivative = 1.0

        value += loss
        gradient += s_i * derivative * x_i

    value += 0.5 * lam * np.dot(theta, theta)
    gradient += lam * theta
    return value, gradient


def objective_gradient_vectorized(theta, X, y, lam):
    s = 1 - 2 * y
    z = s * (X.T @ theta)

    derivative = np.clip(1 + z, 0, 1)
    loss = 0.5 * derivative**2 + np.maximum(z, 0)

    value = np.sum(loss) + 0.5 * lam * np.dot(theta, theta)
    gradient = X @ (s * derivative) + lam * theta
    return value, gradient


def main():
    data_path = Path("../data/mnist_train_test.mat")
    if not data_path.is_file():
        print("Place mnist_train_test.mat in ../data/ and run from code/.")
        return

    data = loadmat(data_path, squeeze_me=True, struct_as_record=False)
    train = data["train"]
    X = np.asarray(train.X, dtype=float)
    y = np.asarray(train.y, dtype=float).reshape(-1)

    lam = 0.005
    rng = np.random.default_rng(329)
    theta = rng.standard_normal(X.shape[0])

    value_loop, gradient_loop = objective_gradient_loop(theta, X, y, lam)
    value_vector, gradient_vector = objective_gradient_vectorized(theta, X, y, lam)

    rtol, atol = 1e-10, 1e-10
    np.testing.assert_allclose(value_loop, value_vector, rtol=rtol, atol=atol)
    np.testing.assert_allclose(gradient_loop, gradient_vector, rtol=rtol, atol=atol)

    print(f"Agreement checks passed (rtol={rtol:g}, atol={atol:g}).")
    print(f"Objective absolute difference: {abs(value_loop - value_vector):.3e}")
    print(
        "Gradient maximum absolute difference: "
        f"{np.max(np.abs(gradient_loop - gradient_vector)):.3e}"
    )

    loop_time = min(repeat(
        lambda: objective_gradient_loop(theta, X, y, lam),
        number=1, repeat=3,
    ))
    vector_time = min(repeat(
        lambda: objective_gradient_vectorized(theta, X, y, lam),
        number=1, repeat=3,
    ))

    print("Objective and gradient together, best of 3 runs:")
    print(f"Loop:       {loop_time:.6f} seconds")
    print(f"Vectorized: {vector_time:.6f} seconds")
    print(f"Speedup:    {loop_time / vector_time:.2f}x")


if __name__ == "__main__":
    main()
