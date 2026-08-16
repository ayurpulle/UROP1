"""Photonic single-parameter sensing (x-quadrature displacement) with a GKP probe and GKP Bell ancilla.

Protocol summary
----------------
1) Prepare a finite-energy square GKP on mode 0 (probe).
2) Create a GKP Bell pair on modes 1–2 using two GKPs, Hadamard, and CNOT.
3) Imprint the unknown displacement along q on the probe: Xgate(theta).
4) Interfere modes 0 and 1 on a 50:50 beamsplitter.
5) Homodyne: MeasureX on one output, MeasureP on the other.
6) Feed-forward: apply X and Z displacements to mode 2 with gain g = 1/sqrt(2).
7) Build an estimator for theta from s1 (q outcome). For the chosen BS convention,
   theta_hat = sqrt(2) * s1. Aggregate across many shots to reduce variance.

The corrected quantum state lives in mode 2 after feed-forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import (
    GKP,
    Rgate,
    CXgate,
    BSgate,
    MeasureX,
    MeasureP,
    Xgate,
    Zgate,
)

import scipy.stats as stats
import matplotlib.pyplot as plt

SQRT2 = math.sqrt(2.0)
np.random.seed(42)

@dataclass
class SingleShotResult:
    s1_q: float
    s2_p: float
    theta_hat: float

def build_gkp_bell_on_modes(q1, q2, epsilon: float = 0.05) -> None:
    """Create a GKP-encoded Bell state on modes (q1,q2)."""
    GKP(epsilon=epsilon) | q1
    GKP(epsilon=epsilon) | q2
    BSgate(math.pi / 4.0, 0.0) | (q1, q2)
    #Rgate(math.pi / 2.0) | q1  # GKP Hadamard
    #CXgate(1.0) | (q1, q2)  # GKP CNOT (SUM)
    

def single_shot(theta: float, epsilon: float = 1.0,
                gain: float = 1.0 / SQRT2, backend: str = "bosonic",
                cutoff_dim: int = 20) -> Tuple[SingleShotResult, sf.State]:
    """
    Run one sensing round for an unknown x-displacement theta.
    
    Args:
        theta: True displacement along q to imprint on the probe (in position units).
        epsilon: Finite-energy GKP parameter.
        gain: Feed-forward gain; 1/sqrt(2) matches a 50:50 BS convention.
        backend: Strawberry Fields backend; "bosonic" recommended for GKP.
        cutoff_dim: Fock cutoff when using the "fock" backend.
    
    Returns:
        Tuple of SingleShotResult with homodyne outcomes and the per-shot estimate, and the full state.
    """
    prog = sf.Program(3)
    with prog.context as q:
        # Probe GKP on mode 0
        GKP(epsilon=epsilon) | q[0]
        
        # GKP Bell on modes 1–2
        build_gkp_bell_on_modes(q[1], q[2], epsilon)
        
        # Parameter imprint: displacement along q on the probe
        Xgate(theta) | q[0]
        
        # Interaction and readout
        BSgate(math.pi / 4.0, 0.0) | (q[0], q[1])
        MeasureX | q[0]  # s1
        MeasureP | q[1]  # s2
        
        # Feed-forward onto mode 2 (error correction / noise removal)
        Xgate(gain * q[0].par) | q[2]
        Zgate(gain * q[1].par) | q[2]
    
    eng_opts = {}
    if backend == "fock" and cutoff_dim is not None:
        eng_opts = {"backend_options": {"cutoff_dim": cutoff_dim}}
    eng = sf.Engine(backend, **eng_opts)
    res = eng.run(prog)
    
    # samples shape: (shots, 2). Order follows measurement order (q0 then p1)
    s1, s2 = [float(x) for x in res.samples[0]]
    theta_hat = SQRT2 * s1  # mapping for the chosen beamsplitter convention
    return SingleShotResult(s1_q=s1, s2_p=s2, theta_hat=theta_hat), res

@dataclass
class BatchResult:
    theta_hats: np.ndarray
    s1_all: np.ndarray
    s2_all: np.ndarray
    theta_mean: float
    theta_std: float

def run_many_shots(theta: float, num_shots: int = 100,
                   epsilon: float = 1.0,
                   gain: float = 1.0 / SQRT2, backend: str = "bosonic",
                   #cutoff_dim: int = 20,
                   prior_mu: float = 0.0, prior_std: float = 1.0,
                   lik_std: float = 0.5) -> BatchResult:
    """Collect multiple shots and compute a mean estimate and sample std."""
    s1_list: List[float] = []
    s2_list: List[float] = []
    th_list: List[float] = []
    
    for _ in range(num_shots):
        out, _ = single_shot(theta, epsilon=epsilon, gain=gain,
                             backend=backend)
        s1_list.append(out.s1_q)
        s2_list.append(out.s2_p)
        th_list.append(out.theta_hat)
    
    s1_all = np.array(s1_list)
    s2_all = np.array(s2_list)
    theta_hats = np.array(th_list)
    
    # Bayesian updates
    post_mu = prior_mu
    post_std = prior_std
    scale = SQRT2  # E[s1] = theta * scale
    
    for s1 in s1_all:
        prior_precision = 1.0 / post_std**2
        lik_precision = (1.0 / lik_std**2) / scale**2  # Effective precision for theta
        post_precision = prior_precision + lik_precision
        post_mu = (post_mu * prior_precision + s1 * lik_precision * (1.0 / scale)) / post_precision  # Corrected update
        post_std = np.sqrt(1.0 / post_precision)
    
    theta_mean = post_mu
    theta_std = post_std
    
    return BatchResult(
        theta_hats=theta_hats,
        s1_all=s1_all,
        s2_all=s2_all,
        theta_mean=theta_mean,
        theta_std=theta_std,
    )

if __name__ == "__main__":
    # Demo run parameters
    true_theta = 0.1
    shots = 10
    eps = 0.05
    
    batch = run_many_shots(true_theta, num_shots=shots, epsilon=eps,
                           backend="bosonic")
    print(f"true theta: {true_theta}")
    print(f"mean estimate: {batch.theta_mean:.4f}  (std over shots: {batch.theta_std:.4f})")
    print(f"first 5 s1: {batch.s1_all[:5]}")
    print(f"first 5 theta_hats: {batch.theta_hats[:5]}")
    
    # Run a single shot to get the state for plotting
    _, single_res = single_shot(true_theta, epsilon=eps, gain= 1.0 / SQRT2, backend="bosonic")
    output_state = single_res.state  # res.state is the full multi-mode state
    
    # Compute Wigner for mode 2
    xvec = np.linspace(-5, 5, 100)
    pvec = np.linspace(-5, 5, 100)
    W = output_state.wigner(mode=2, xvec=xvec, pvec=pvec)
    
    # Plot
    plt.contourf(xvec, pvec, W, 100)
    plt.colorbar()
    plt.xlabel('q')
    plt.ylabel('p')
    plt.title('Wigner Function of Corrected Mode 2')
    plt.savefig('wigner_mode2.png')
    plt.show()


