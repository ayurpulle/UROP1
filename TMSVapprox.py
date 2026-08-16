"""
Protocol (Knill-style controlled displacement): 

1) Prepare a finite-energy square GKP on mode 0 
2) Create TMSV pair on modes 1/2 using two squeezers ±r and a 50:50 BS.
3) Apply controlled displacement: displacement on GKP (mode 0) controlled by TMSV ancilla (mode 1).
4) Homodyne measurements: MeasureX on mode 0 (q̂), MeasureP on mode 1 (p̂).
5) The third mode of TMSV holds the teleported/corrected state |ψ'⟩.

This follows the Knill circuit where the displacement is controlled rather than unconditional.

Squeezing parameter eps is related to r by r = -10log_10(eps)?
"""


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

'No CX, isntead using BS and TMSV as a CV approximation for the Bell state from https://arxiv.org/pdf/2505.14775, which shows theoretical equivalent performance to CX.'

import scipy.stats as stats
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class SingleShotResult:
    s1_q: float
    s2_p: float
    theta_hat: float

SQRT2 = np.sqrt(2.0)
np.random.seed(42)

def get_initial_gkp_state(epsilon: float = 0.03, backend: str = "bosonic"):
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
    eng = sf.Engine(backend, **eng_opts)
    result = eng.run(prog)
    
    # Compute Wigner function
    xvec = np.linspace(-10, 10, 100)
    pvec = np.linspace(-10, 10, 100)
    W_gkp = result.state.wigner(mode=0, xvec=xvec, pvec=pvec)
    
    return result.state, (xvec, pvec, W_gkp)

def calculate_protocol_fidelity(initial_state, final_state):
    """
    Calculate fidelity between initial GKP (mode 0) and corrected GKP (mode 2).
    
    Args:
        initial_state: Initial GKP state (1-mode)
        final_state: Final state after protocol (3-mode, corrected state in mode 2)
        
    Returns:
        fidelity: Float between 0 and 1
        method: String indicating which method was used
    """
    try:
        # Method 1: Extract reduced states and use fidelity
        print("Trying reduced state fidelity...")
        
        # Get reduced state for mode 2 from final state
        final_reduced = final_state.reduced_dm(2)
        
        # For bosonic backend, try direct fidelity with proper syntax
        fidelity = initial_state.fidelity(final_reduced, mode=0)
        return fidelity, "reduced_state_fidelity"
        
        'why using reduced_dm instead of trace?'
    except Exception as e:
        print(f"Reduced state fidelity failed: {e}")
        try:
            # Method 2: Wigner function correlation (more reliable)
            print("Using Wigner correlation...")
            xvec = np.linspace(-6, 6, 60)
            pvec = np.linspace(-6, 6, 60)
            
            W_initial = np.real(initial_state.wigner(mode=0, xvec=xvec, pvec=pvec))
            W_final = np.real(final_state.wigner(mode=2, xvec=xvec, pvec=pvec))
            
            # Debug: Check if Wigner functions are reasonable
            print(f"   Initial Wigner range: [{W_initial.min():.4f}, {W_initial.max():.4f}]")
            print(f"   Final Wigner range: [{W_final.min():.4f}, {W_final.max():.4f}]")
            
            # Flatten arrays for correlation
            w1_flat = W_initial.flatten()
            w2_flat = W_final.flatten()
            'better method to isolate xvec and pvec, then flatten and do inner product?'
            # Remove any NaN or infinite values
            mask = np.isfinite(w1_flat) & np.isfinite(w2_flat)
            w1_clean = w1_flat[mask]
            w2_clean = w2_flat[mask]
            
            if len(w1_clean) == 0:
                print("   Warning: No valid Wigner data points")
                return 0.0, "wigner_correlation_failed"
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(w1_clean, w2_clean)[0, 1]
            
            if np.isnan(correlation):
                print("   Warning: Correlation is NaN, trying overlap method...")
                # Fallback to normalized overlap
                overlap = np.sum(w1_clean * w2_clean) / (np.linalg.norm(w1_clean) * np.linalg.norm(w2_clean))
                fidelity = (overlap + 1) / 2  # Map to [0,1]
            else:
                # Convert correlation to pseudo-fidelity
                fidelity = (correlation + 1) / 2  # Map [-1,1] to [0,1]
            
            fidelity = np.clip(fidelity, 0, 1)
            print(f"   Raw correlation: {correlation:.4f}")
            
            return fidelity, "wigner_correlation"
            
        except Exception as e2:
            print(f"Wigner overlap failed: {e2}")
            try:
                # Method 3: Simple correlation coefficient of Wigner functions
                print("Using Wigner correlation...")
                xvec = np.linspace(-5, 5, 50)
                pvec = np.linspace(-5, 5, 50)
                
                W_initial = np.real(initial_state.wigner(mode=0, xvec=xvec, pvec=pvec))
                W_final = np.real(final_state.wigner(mode=2, xvec=xvec, pvec=pvec))
                
                # Flatten for correlation calculation
                w1_flat = W_initial.flatten()
                w2_flat = W_final.flatten()
                
                # Calculate correlation coefficient
                correlation = np.corrcoef(w1_flat, w2_flat)[0, 1]
                
                # Convert correlation to pseudo-fidelity (0 to 1 range)
                fidelity = (correlation + 1) / 2  # Map [-1,1] to [0,1]
                fidelity = np.clip(fidelity, 0, 1)
                
                return fidelity, "wigner_correlation"
                
            except Exception as e3:
                print(f"All fidelity methods failed: {e3}")
                return None, "failed"

def build_epr_tmsv_on_modes(q1, q2, r: float = 3.0) -> None:
    """Create an EPR/TMSV state on modes (q1,q2) using two squeezers and a 50:50 BS."""
    Sgate(r, 0.0) | q1
    Sgate(-r, 0.0) | q2
    BSgate(np.pi / 4.0, 0.0) | (q1, q2)



def plot_initial_gkp_state(epsilon: float = 0.03, backend: str = "bosonic"):
    """
    Create and plot the initial GKP state before any operations.
    
    Args:
        epsilon: GKP finite energy parameter
        backend: Strawberry Fields backend 
        cutoff_dim: Fock space cutoff dimension
        
    Returns:
        The initial GKP state object
    """
    import matplotlib.pyplot as plt
    
    # Create a simple 1-mode program with just the GKP state
    prog = sf.Program(1)
    with prog.context as q:
        GKP(epsilon=epsilon) | q[0]
    
    # Run the program to get the initial state
    eng_opts = {}
    eng = sf.Engine(backend, **eng_opts)
    result = eng.run(prog)
    
    # Plot the Wigner function
    xvec = np.linspace(-10, 10, 100)
    pvec = np.linspace(-10, 10, 100)
    W_gkp = result.state.wigner(mode=0, xvec=xvec, pvec=pvec)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xvec, pvec, W_gkp, 100)
    plt.xlabel('q')
    plt.ylabel('p')
    plt.title(f'Initial GKP State (ε={epsilon})')
    plt.colorbar()
    plt.savefig(f'initial_gkp_state_eps_{epsilon}.png', dpi=150)
    plt.show()
    
    return result.state


def single_shot_beamsplitter(theta: float, epsilon: float = 0.1, r: float = 3.0,
                            gain: float = SQRT2, backend: str = "bosonic"):
    """
    Beamsplitter-based protocol (original approach):
    1) GKP probe on mode 0
    2) TMSV ancilla on modes 1-2
    3) Unconditional displacement on probe
    4) Beamsplitter interaction between probe and ancilla
    5) Homodyne measurements and feed-forward
    """
    prog = sf.Program(3)
    with prog.context as q:
        
        GKP(epsilon=epsilon) | q[0]
        build_epr_tmsv_on_modes(q[1], q[2], r)
        Xgate(theta) | q[0]
        BSgate(np.pi / 4.0, 0.0).H | (q[0], q[1])
        MeasureX | q[0]  # s1
        MeasureP | q[1]  # s2
        Xgate(gain * q[0].par) | q[2]
        Zgate(gain * q[1].par) | q[2]
        
    eng_opts = {}
    eng = sf.Engine(backend, **eng_opts)
    res = eng.run(prog)
    
    # Extract measurements and return state
    s1, s2 = [float(x) for x in res.samples[0]]
    theta_hat = SQRT2 * s1  # Standard beamsplitter scaling
    return SingleShotResult(s1_q=s1, s2_p=s2, theta_hat=theta_hat), res.state



if __name__ == "__main__":
    # Demo run parameters
    true_theta = 0.0
    eps = 0.01
    r_epr = 1.0

    # Get the initial GKP state before any operations
    print("CREATING INITIAL GKP STATE:")
    initial_gkp_state, (xvec_gkp, pvec_gkp, W_gkp_initial) = get_initial_gkp_state(epsilon=0.03, backend="bosonic")
    print("Initial GKP state created.\n")
    
    # Run beamsplitter version with TMSV
    print("1. BEAMSPLITTER + TMSV:")
    result_fock_bs, state_fock_bs = single_shot_beamsplitter(true_theta, epsilon=0.03, backend='bosonic')
    print(f"   s1: {result_fock_bs.s1_q:.4f}, s2: {result_fock_bs.s2_p:.4f}")
    print(f"   theta_hat: {result_fock_bs.theta_hat:.4f} (scaling: sqrt(2)*s1)")
    print(f"   Error: {abs(result_fock_bs.theta_hat - true_theta):.4f}")
    
    # Calculate fidelity between initial and final states
    print("\n2. FIDELITY ANALYSIS:")
    fidelity, method = calculate_protocol_fidelity(initial_gkp_state, state_fock_bs)
    if fidelity is not None:
        print(f"   Fidelity: {fidelity:.4f} (method: {method})")
        print(f"   Fidelity percentage: {fidelity*100:.2f}%")
        if fidelity > 0.9:
            print("   ✓ Excellent teleportation quality")
        elif fidelity > 0.8:
            print("   ✓ Good teleportation quality")
        elif fidelity > 0.6:
            print("   ~ Moderate teleportation quality")
        else:
            print("   ✗ Poor teleportation quality")
    else:
        print("   ✗ Fidelity calculation failed")
    
    print(f"\nTrue theta: {true_theta}")

    # Create 2-subplot figure: Initial GKP + Final TMSV result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Compute Wigner for mode 2 in the protocol result
    xvec = np.linspace(-10, 10, 100)
    pvec = np.linspace(-10, 10, 100)
    
    # Subplot 1: Initial GKP state
    c1 = ax1.contourf(xvec_gkp, pvec_gkp, W_gkp_initial, 100)
    ax1.set_xlabel('q')
    ax1.set_ylabel('p')
    ax1.set_title('Initial GKP State (Mode 0)')
    plt.colorbar(c1, ax=ax1)
    
    # Subplot 2: Final state after TMSV protocol
    W_bs = state_fock_bs.wigner(mode=2, xvec=xvec, pvec=pvec)
    c2 = ax2.contourf(xvec, pvec, W_bs, 100)
    ax2.set_xlabel('q')
    ax2.set_ylabel('p')
    if fidelity is not None:
        ax2.set_title(f'Beamsplitter + TMSV (Mode 2)\nFidelity: {fidelity:.3f}')
    else:
        ax2.set_title('Beamsplitter + TMSV (Mode 2)')
    plt.colorbar(c2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('tmsv_wigner_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    fidelity, method = calculate_protocol_fidelity(initial_gkp_state, state_fock_bs)
    print(f"Fidelity: {fidelity}")
    print(f"Method: {method}")
    print(f"Error: {abs(fidelity - 1)}")
     



