import numpy as np

def order_parameter_r(thetas: np.ndarray) -> float:
    """Kuramoto-style order parameter magnitude r in [0,1].
    thetas: array of phases (radians)."""
    thetas = np.asarray(thetas, dtype=float)
    z = np.exp(1j * thetas)
    r = np.abs(np.mean(z))
    return float(r)

def phase_variance(thetas: np.ndarray) -> float:
    """Circular variance proxy using order parameter r: Var ≈ 1 - r."""
    r = order_parameter_r(thetas)
    return float(1.0 - r)

def lyapunov_energy(thetas: np.ndarray, delta_vec: np.ndarray | None = None,
                    alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> float:
    """ℒ = α·Var(θ) + β·r² + γ·||Δ||²."""
    var_theta = phase_variance(thetas)
    r = order_parameter_r(thetas)
    r2 = r*r
    delta_norm2 = 0.0 if delta_vec is None else float(np.sum(np.square(delta_vec)))
    L = alpha * var_theta + beta * r2 + gamma * delta_norm2
    return float(L)

def step_phases(thetas: np.ndarray, omega: np.ndarray, K: float, dt: float) -> np.ndarray:
    """One Euler step of a Kuramoto-like update for demonstration.
    dθ_i/dt = ω_i + (K/N) * sum_j sin(θ_j - θ_i)
    """
    thetas = np.asarray(thetas, dtype=float)
    omega = np.asarray(omega, dtype=float)
    N = thetas.size
    # Coupling term
    sin_diff = np.sin(thetas[None, :] - thetas[:, None])  # i,j
    coupling = (K / N) * np.sum(sin_diff, axis=1)
    dtheta = omega + coupling
    return (thetas + dt * dtheta) % (2*np.pi)
