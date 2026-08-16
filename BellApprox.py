'''
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import (
    GKP,
    Sgate,
    BSgate,
    MeasureX,
    MeasureP,
    Xgate,
    Zgate,
    S2gate,
    CXgate,  # For controlled displacement
    Rgate,   # For GKP Bell preparation
    Ket,     # For direct Bell state preparation
)

import scipy.stats as stats
import matplotlib.pyplot as plt


import scipy.stats as stats
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class SingleShotResult:
    s1_q: float
    s2_p: float
    theta_hat: float

SQRT2 = np.sqrt(2.0)

def build_fock_bell_direct(q1, q2) -> None:
    """
    Directly prepare a Fock basis Bell state |00⟩ + |11⟩.
    This creates (|0,0⟩ + |1,1⟩)/√2 in the photon number basis.
    """
    # Direct Bell state preparation using Ket operation
    # Bell state: (|00⟩ + |11⟩)/√2
    # In SF, this is represented as amplitudes [c00, c01, c10, c11]
    bell_amplitudes = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])  # |00⟩ + |11⟩
    Ket(bell_amplitudes) | (q1, q2)

def single_shot_fock_bell(theta: float, epsilon: float = 0.2, 
                         gain: float = 1.0 / SQRT2, backend: str = "fock", cutoff_dim: int = 10):
    """
    Direct Fock Bell pair protocol:
    1) GKP probe on mode 0
    2) Direct Fock Bell pair |00⟩ + |11⟩ on modes 1-2
    3) Unconditional displacement on probe
    4) Beamsplitter interaction between probe and one Bell mode
    5) Homodyne measurements and feed-forward
    """
    prog = sf.Program(3)
    with prog.context as q:
        
        GKP(epsilon=epsilon) | q[0]
        build_fock_bell_direct(q[1], q[2])  # Direct Bell state
        BSgate(np.pi / 4.0, 0.0) | (q[0], q[1])
        MeasureX | q[0]  # s1
        MeasureP | q[1]  # s2
        Xgate(gain * q[0].par) | q[2]
        Zgate(gain * q[1].par) | q[2]
        
    eng_opts = {"backend_options": {"cutoff_dim": cutoff_dim}}
    eng = sf.Engine(backend, **eng_opts)
    res = eng.run(prog)
    
    # Extract measurements and return state
    s1, s2 = [float(x) for x in res.samples[0]]
    theta_hat = SQRT2 * s1  # Standard beamsplitter scaling
    return SingleShotResult(s1_q=s1, s2_p=s2, theta_hat=theta_hat), res.state

def single_shot_fock_bell_cx(theta: float, epsilon: float = 0.03, 
                            gain: float = 1.0 / SQRT2, backend: str = "fock", cutoff_dim: int = 10):
    """
    Direct Fock Bell pair with controlled displacement protocol:
    1) GKP probe on mode 0
    2) Direct Fock Bell pair |00⟩ + |11⟩ on modes 1-2
    3) Controlled displacement using CXgate
    4) Homodyne measurements (no beamsplitter, no feed-forward)
    """
    prog = sf.Program(3)
    with prog.context as q:
        
        GKP(epsilon=epsilon) | q[0]
        build_fock_bell_direct(q[1], q[2])  # Direct Bell state
        CXgate(theta).H | (q[0], q[1])  # Controlled displacement
        MeasureX | q[0]  # s1 (q̂ measurement)
        MeasureP | q[1]  # s2 (p̂ measurement)
        
    eng_opts = {"backend_options": {"cutoff_dim": cutoff_dim}}
    eng = sf.Engine(backend, **eng_opts)
    res = eng.run(prog)
    
    # Extract measurements and return state
    s1, s2 = [float(x) for x in res.samples[0]]
    theta_hat = SQRT2 * s1  # Standard scaling
    return SingleShotResult(s1_q=s1, s2_p=s2, theta_hat=theta_hat), res.state

if __name__ == "__main__":
    # Demo run parameters
    true_theta = 0.03
    eps = 0.2
    r_epr = 0.2
    
    
    # Run beamsplitter version with direct Fock Bell pair
    print("1. BEAMSPLITTER + DIRECT FOCK BELL:")
    result_fock_bs, state_fock_bs = single_shot_fock_bell(true_theta, epsilon=0.2, backend="fock", cutoff_dim=10)
    print(f"   s1: {result_fock_bs.s1_q:.4f}, s2: {result_fock_bs.s2_p:.4f}")
    print(f"   theta_hat: {result_fock_bs.theta_hat:.4f} (scaling: sqrt(2)*s1)")
    print(f"   Error: {abs(result_fock_bs.theta_hat - true_theta):.4f}")
    
    # Run controlled displacement version with direct Fock Bell pair
    print("\n2. CONTROLLED DISPLACEMENT + DIRECT FOCK BELL:")
    result_fock_cx, state_fock_cx = single_shot_fock_bell_cx(true_theta, epsilon=0.2, backend="fock", cutoff_dim=10)
    print(f"   s1: {result_fock_cx.s1_q:.4f}, s2: {result_fock_cx.s2_p:.4f}")
    print(f"   theta_hat: {result_fock_cx.theta_hat:.4f} (scaling: sqrt(2)*s1)")
    print(f"   Error: {abs(result_fock_cx.theta_hat - true_theta):.4f}")


        # Beamsplitter + Direct Fock Bell version
    W_fock_bs = state_fock_bs.wigner(mode=2, xvec=xvec, pvec=pvec)
    c1 = ax1.contourf(xvec, pvec, W_fock_bs, 100)
    ax1.set_xlabel('q')
    ax1.set_ylabel('p')
    ax1.set_title('Beamsplitter + Direct Fock Bell')
    plt.colorbar(c1, ax=ax1)
    
    # Controlled displacement + Direct Fock Bell version (mode 2 contains corrected state)
    W_fock_cx = state_fock_cx.wigner(mode=2, xvec=xvec, pvec=pvec)
    c2 = ax2.contourf(xvec, pvec, W_fock_cx, 100)
    ax2.set_xlabel('q')
    ax2.set_ylabel('p')
    ax2.set_title('Controlled Displacement + Direct Fock Bell')
    plt.colorbar(c2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('wigner_comparison_fock_bell_protocols.png', dpi=150)
    plt.show()
'''    

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import (
    GKP,
    BSgate,
    MeasureX,
    MeasureP,
    Xgate,
    Zgate,
    CXgate,
    Fouriergate,
    Ket, # Import Ket
)
from dataclasses import dataclass

@dataclass
class SingleShotResult:
    s1_q: float
    s2_p: float
    theta_hat: float

SQRT2 = np.sqrt(2.0)

# ... (SingleShotResult class and SQRT2 constant remain the same) ...

def get_initial_gkp_state(epsilon: float = 0.03, backend: str = "fock", cutoff_dim: int = 5):
    """
    Create the initial GKP state before any operations (without plotting).
    
    Args:
        epsilon: GKP finite energy parameter
        backend: Strawberry Fields backend 
        cutoff_dim: Fock space cutoff dimension
        
    Returns:
        Tuple of (state, wigner_data) where wigner_data is (xvec, pvec, W_gkp)
    """
    # Create a simple 1-mode program with just the GKP state
    prog = sf.Program(1)
    with prog.context as q:
        GKP(epsilon=epsilon) | q[0]
    
    # Run the program to get the initial state
    eng_opts = {}
    if backend == "fock" and cutoff_dim is not None:
        eng_opts = {"backend_options": {"cutoff_dim": cutoff_dim}}
    eng = sf.Engine(backend, **eng_opts)
    result = eng.run(prog)
    
    # Compute Wigner function
    xvec = np.linspace(-5, 5, 100)
    pvec = np.linspace(-5, 5, 100)
    W_gkp = result.state.wigner(mode=0, xvec=xvec, pvec=pvec)
    
    return result.state, (xvec, pvec, W_gkp)

def build_fock_bell_direct(q1, q2, cutoff_dim: int = 10) -> None:
    """
    Directly prepare a Fock basis Bell state |00⟩ + |11⟩.
    This creates (|0,0⟩ + |1,1⟩)/√2 in the photon number basis.
    
    For Strawberry Fields, a 2-mode state requires an N×N amplitude matrix,
    where N is the cutoff dimension.
    """
    # Initialize N×N amplitude matrix for 2-mode state
    bell_amplitudes = np.zeros((cutoff_dim, cutoff_dim), dtype=complex)
    
    # Set amplitudes for |00⟩ + |11⟩ 
    # bell_amplitudes[n1, n2] corresponds to amplitude of |n1,n2⟩
    bell_amplitudes[0, 0] = 1/np.sqrt(2)  # |0,0⟩ component
    if cutoff_dim > 1:
        bell_amplitudes[1, 1] = 1/np.sqrt(2)  # |1,1⟩ component
    
    # Apply the state
    Ket(bell_amplitudes) | (q1, q2)

def single_shot_direct_bs(theta: float, epsilon: float = 0.2, 
                           gain: float = 1.0 / SQRT2, backend: str = "fock", cutoff_dim: int = 5):
    """
    Protocol using a directly constructed initial state with Ket.
    """

    prog = sf.Program(3)
    with prog.context as q:

        GKP(epsilon=epsilon) | q[0]    
        build_fock_bell_direct(q[1], q[2], cutoff_dim)
        # The rest of the circuit is the same
        BSgate(np.pi / 4.0, 0.0) | (q[0], q[1])
        MeasureX | q[0]
        MeasureP | q[1]
        Xgate(gain * q[0].par) | q[2]
        Zgate(gain * q[1].par) | q[2]
        
    eng_opts = {"backend_options": {"cutoff_dim": cutoff_dim}}
    eng = sf.Engine(backend, **eng_opts)
    res = eng.run(prog)
    
    s1, s2 = [float(x) for x in res.samples[0]]
    theta_hat = SQRT2 * s1
    return SingleShotResult(s1_q=s1, s2_p=s2, theta_hat=theta_hat), res.state

def single_shot_direct_cx(theta: float, epsilon: float = 0.2, 
                            gain: float = 1.0 / SQRT2, backend: str = "fock", cutoff_dim: int = 5):
    """
    Protocol using a directly constructed initial state with Ket.
    """

    prog = sf.Program(3)
    with prog.context as q:

        GKP(epsilon=epsilon) | q[0]    
        build_fock_bell_direct(q[1], q[2], cutoff_dim)
        # The rest of the circuit is the same
        #Xgate(theta) | q[0]
        CXgate(theta).H | (q[0], q[1])  
        MeasureX | q[0]  # s1 (q̂ measurement)
        MeasureP | q[1]  # s2 (p̂ measurement)
        
    eng_opts = {"backend_options": {"cutoff_dim": cutoff_dim}}
    eng = sf.Engine(backend, **eng_opts)
    res = eng.run(prog)
    
    s1, s2 = [float(x) for x in res.samples[0]]
    theta_hat = SQRT2 * s1
    return SingleShotResult(s1_q=s1, s2_p=s2, theta_hat=theta_hat), res.state

if __name__ == "__main__":
    true_theta = 0.03
    
    # Get the initial GKP state before any operations
    print("CREATING INITIAL GKP STATE:")
    initial_gkp_state, (xvec_gkp, pvec_gkp, W_gkp_initial) = get_initial_gkp_state(epsilon=0.03, backend="fock", cutoff_dim=5)
    print("Initial GKP state created.\n")
    
    print("1. BEAMSPLITTER VERSION (Direct Fock Bell):")
    result_bs, state_bs = single_shot_direct_bs(
        true_theta, epsilon=0.03, backend="fock", cutoff_dim=5
    )
    print(f"   s1: {result_bs.s1_q:.4f}, s2: {result_bs.s2_p:.4f}")
    print(f"   theta_hat: {result_bs.theta_hat:.4f}")
    print(f"   Error: {abs(result_bs.theta_hat - true_theta):.4f}")
    
    print("\n2. CONTROLLED DISPLACEMENT VERSION (Direct Fock Bell):")
    result_cx, state_cx = single_shot_direct_cx(
        true_theta, epsilon=0.03, backend="fock", cutoff_dim=5
    )
    print(f"   s1: {result_cx.s1_q:.4f}, s2: {result_cx.s2_p:.4f}")
    print(f"   theta_hat: {result_cx.theta_hat:.4f}")
    print(f"   Error: {abs(result_cx.theta_hat - true_theta):.4f}")
    
    print(f"\nTrue theta: {true_theta}")
    
    # Create comprehensive 3-subplot figure
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Compute Wigner for mode 2 in both protocol cases
    xvec = np.linspace(-5, 5, 100)
    pvec = np.linspace(-5, 5, 100)
    
    # Subplot 1: Initial GKP state
    c1 = ax1.contourf(xvec_gkp, pvec_gkp, W_gkp_initial, 100)
    ax1.set_xlabel('q')
    ax1.set_ylabel('p')
    ax1.set_title('Initial GKP State (Mode 0)')
    plt.colorbar(c1, ax=ax1)
    
    # Subplot 2: Beamsplitter version (mode 2 contains corrected state)
    W_bs = state_bs.wigner(mode=2, xvec=xvec, pvec=pvec)
    c2 = ax2.contourf(xvec, pvec, W_bs, 100)
    ax2.set_xlabel('q')
    ax2.set_ylabel('p')
    ax2.set_title('Beamsplitter + Direct Fock Bell (Mode 2)')
    plt.colorbar(c2, ax=ax2)
    
    # Subplot 3: Controlled displacement version (mode 2 should contain the state)
    W_cx = state_cx.wigner(mode=2, xvec=xvec, pvec=pvec)
    c3 = ax3.contourf(xvec, pvec, W_cx, 100)
    ax3.set_xlabel('q')
    ax3.set_ylabel('p')
    ax3.set_title('Controlled Displacement + Direct Fock Bell (Mode 2)')
    plt.colorbar(c3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_wigner_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()