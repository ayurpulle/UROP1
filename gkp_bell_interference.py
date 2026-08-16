"""
Photonic GKP error-correction primitive in Strawberry Fields.

This script:
 1) Prepares a data GKP qubit |psi> in mode 0.
 2) Creates a Bell pair |Phi+> = (|00> + |11>)/sqrt(2) on modes 1-2 using
    H (Fourier) on mode 1 followed by a CNOT (SUM/CX) onto mode 2.
 3) Interferes the data mode (0) and ancilla mode (1) on a 50:50 beamsplitter.
 4) Homodyne measures q on one output and p on the other, with gain factors
    matching the 50:50 convention, and applies feed-forward displacements to
    mode 2 to obtain the corrected state |psi'>.

Notes
-----
- Uses the bosonic backend which supports native GKP preparation.
- The CNOT in the GKP encoding is implemented by the CV SUM gate `CXgate(1.0)`.
- The Hadamard on a GKP qubit is a phase-space rotation by pi/2: `Rgate(np.pi/2)`.

"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import (
    GKP,
    Rgate,
    CXgate,
    BSgate,
    S2gate,
    MeasureX,
    MeasureP,
    Xgate,
    Zgate,
    Sgate,
)


def run_gkp_ec(
    epsilon: float = 0.1,
    gain: float = 1,
    backend: str = "bosonic",
    cutoff_dim: int | None = None,
    psi_rotation: float | None = None,
    r_tmsv: float = 10.0,
):
    """
    Execute the photonic GKP error-correction/teleportation step.

    Args:
        epsilon: Finite-energy parameter for GKP states (smaller -> closer to ideal).
        gain: Feed-forward gain applied to homodyne results (1/sqrt(2) for 50:50 BS).
        backend: Strawberry Fields backend ("bosonic" recommended for GKP).
        cutoff_dim: Fock cutoff when using the "fock" backend.
        psi_rotation: Optional rotation (radians) to define the input |psi> from |0>.
        use_tmsv: If True, use a two-mode squeezed vacuum (TMSV/EPR) pair as the
            entanglement resource instead of a GKP Bell pair built with CX.
        r_tmsv: Two-mode squeezing strength for the EPR pair.

    Returns:
        result: Engine run result; the corrected state is in mode 2.
    """

    # Program with 3 modes: [data, ancilla_A, ancilla_B]
    prog = sf.Program(3)

    with prog.context as q:

        theta = 0.1
        # 1) Prepare data GKP |psi> in mode 0
        GKP(epsilon=epsilon) | q[0]


        # 2) Prepare entanglement resource on modes 1-2
        Sgate(r_tmsv, 0.0)  | q[1]
        Sgate(-r_tmsv, 0.0) | q[2]
        BSgate(np.pi/4, 0.0) | (q[1], q[2])

        Xgate(theta) | q[0]

        # 3) Interfere data (0) with ancilla_A (1) on a 50:50 beamsplitter
        BSgate(np.pi / 4.0, 0.0) | (q[0], q[1])

        # 4) Homodyne measurements q and p, then feed-forward onto mode 2
        MeasureX | q[0]  # s1
        MeasureP | q[1]  # s2

        # Apply corrections to mode 2. Gain accounts for the 50:50 BS.
        Xgate(gain * q[0].par) | q[2]
        Zgate(gain * q[1].par) | q[2]

    eng_opts = {}
    if backend == "fock":
        if cutoff_dim is None:
            cutoff_dim = 50
        eng_opts = {"backend_options": {"cutoff_dim": cutoff_dim}}

    eng = sf.Engine(backend, **eng_opts)
    result = eng.run(prog)
    return result


def plot_wigner(state, ax, title, mode=0, xlim=(-10, 10), plim=(-10, 10), resolution=100):
    """
    Plot Wigner function for a given state.
    """
    x = np.linspace(xlim[0], xlim[1], resolution)
    p = np.linspace(plim[0], plim[1], resolution)
    X, P = np.meshgrid(x, p)
    
    # Calculate Wigner function
    W = state.wigner(mode, x, p)
    
    # Plot
    levels = np.linspace(-0.5, 0.5, 20)
    contour = ax.contourf(X, P, W, levels=levels, cmap='RdBu_r')
    ax.contour(X, P, W, levels=[0], colors='black', linewidths=0.5)
    
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return contour

def analyze_wigner_functions(epsilon=0.1, gain= 1 / np.sqrt(2.0), backend="bosonic"):
    """
    Plot Wigner functions of mode 0 (probe after interference) and mode 2 (corrected state).
    """
    # Run the GKP error correction protocol
    result = run_gkp_ec(epsilon=epsilon, gain=gain, backend=backend)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get the final state
    state = result.state
    
    # Plot Wigner function for mode 0 (probe after interference and measurement)
    # Note: After measurement, this mode may not have a well-defined Wigner function
    # We'll plot the Wigner function of mode 2 (corrected state) instead
    
    # Plot Wigner function for mode 2 (corrected state)
    contour1 = plot_wigner(state, ax1, 'Mode 2 (Corrected State)', mode=2,
                       xlim=(-10, 10), plim=(-10, 10), resolution=100)
    
    # For comparison, let's also create a reference GKP state to see the correction effect
    prog_ref = sf.Program(1)
    with prog_ref.context as q:
        GKP(epsilon=epsilon) | q[0]
    
    eng_opts = {}
    if backend == "fock":
        eng_opts = {"backend_options": {"cutoff_dim": 50}}
    
    eng = sf.Engine(backend, **eng_opts)
    ref_result = eng.run(prog_ref)
    
    contour2 = plot_wigner(ref_result.state, ax2, 'Reference GKP State', 
                          xlim=(-3, 3), plim=(-3, 3), resolution=100)
    
    # Add colorbars
    plt.colorbar(contour1, ax=ax1, label='Wigner function')
    plt.colorbar(contour2, ax=ax2, label='Wigner function')
    
    plt.tight_layout()
    plt.show()
    
    return fig, (ax1, ax2)



if __name__ == "__main__":
    # Example run
    res = run_gkp_ec(epsilon=0.08, gain=1.0 / np.sqrt(2.0), backend="bosonic")
    analyze_wigner_functions(epsilon=0.08, gain=1.0 / np.sqrt(2.0), backend="bosonic")
    # The corrected state is in mode 2 after it goes through the circuit
    print("Measurement samples (s1, s2):", res.samples)
    print("Corrected mode is mode 2 after it goes through the circuit; you can now analyze res.state as needed.")


