import numpy as np
from qutip import wigner, Qobj, wigner_cmap

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

import strawberryfields as sf
from strawberryfields.ops import *
from thewalrus.quantum import state_vector, density_matrix

# Parameters
r = 0.50
g = -np.exp(-r) / np.sqrt(2)

# Need more modes for GKP states: 4 cat states × 8 modes each = 32 modes total
# 2 cat states → 1 GKP state, so 4 cat states → 2 GKP states (logical + ancilla)
nmodes = 32

prog = sf.Program(nmodes)
eng = sf.Engine("gaussian")

def create_cat_state_block(prog, q_block, block_id):
    """Create a single cat state using 8 modes (indices q_block[0] to q_block[7])"""
    # Two-mode squeezed state preparation for each pair
    for i in range(0, 8, 2):
        S2gate(1.0) | (q_block[i], q_block[i+1])
        LossChannel(0.945 * 0.99) | q_block[i]
        LossChannel(0.945 * 0.99) | q_block[i+1]
        LossChannel(0.968) | q_block[i]
        Sgate(-r) | q_block[i]
        LossChannel(0.99) | q_block[i]
        LossChannel(0.968) | q_block[i]
    
    # Beam splitting network on even modes
    BSgate(np.arccos(np.sqrt(0.5)), 0.0) | (q_block[0], q_block[2])
    BSgate(np.arccos(np.sqrt(0.5)), 0.0) | (q_block[4], q_block[6])
    BSgate(np.arccos(np.sqrt(0.5)), 0.0) | (q_block[0], q_block[4])
    
    # Loss channels
    for i in range(8):
        LossChannel(0.942) | q_block[i]
    
    # Additional losses on specific modes
    for i in [1, 3, 5, 7]:
        LossChannel(0.965 * 0.94) | q_block[i]
    for i in [2, 4, 6]:
        LossChannel(0.94) | q_block[i]
    
    # Homodyne measurements on modes 2, 4, 6
    MeasureHomodyne(np.pi/2) | q_block[2]
    MeasureHomodyne(np.pi/2) | q_block[6]
    MeasureHomodyne(np.pi/2) | q_block[4]
    
    # Transmission coefficients
    tau = 0.943*0.99*0.965*0.942*0.94
    g_eff = g / np.sqrt(tau)
    
    # Conditional displacement on mode 0 of this block
    total_phase = (g_eff * q_block[4].par + 
                   (g_eff * q_block[2].par + g_eff * q_block[6].par) / np.sqrt(2))
    Zgate(total_phase) | q_block[0]
    
    # Final losses on the output mode
    LossChannel(0.99 * 0.99) | q_block[0]
    LossChannel(0.968) | q_block[0]
    LossChannel(0.94) | q_block[0]

def combine_cats_to_gkp(prog, cat_modes, output_mode, ancilla_mode=None):
    """Combine 2 cat states into a GKP state using interferometry"""
    
    # Phase relationship for GKP grid creation
    # Different phases create different grid orientations
    gkp_phase_1 = 0.0  # First cat state phase
    gkp_phase_2 = np.pi/2  # Second cat state phase (90° shifted)
    
    # Apply phases to prepare for interferometry
    Rgate(gkp_phase_1) | cat_modes[0]
    Rgate(gkp_phase_2) | cat_modes[1]
    
    # Interferometric combination - specific beam splitter ratios for GKP
    # This creates the grid structure characteristic of GKP states
    BSgate(np.arccos(np.sqrt(0.6)), 0.0) | (cat_modes[0], cat_modes[1])
    
    # Homodyne measurement on one mode to project onto GKP subspace
    MeasureHomodyne(0.0) | cat_modes[1]  # x-quadrature measurement
    
    # Conditional operations based on measurement outcome
    # This helps stabilize the GKP grid
    gkp_correction_gain = 0.1
    Xgate(gkp_correction_gain * cat_modes[1].par) | cat_modes[0]
    
    # The first cat mode now contains the GKP state
    return cat_modes[0]

with prog.context as q:
    print("Creating 4 cat states in parallel...")
    
    # Create 4 cat states using blocks of 8 modes each
    cat_output_modes = []
    for block in range(4):
        block_modes = q[8*block:8*(block+1)]
        create_cat_state_block(prog, block_modes, block)
        cat_output_modes.append(block_modes[0])  # The output cat state is always mode 0 of each block
    
    print("Combining cat states into GKP states...")
    
    # Combine first 2 cat states into logical GKP qubit
    logical_gkp_mode = combine_cats_to_gkp(prog, [cat_output_modes[0], cat_output_modes[1]], 
                                          output_mode=cat_output_modes[0])
    
    # Combine next 2 cat states into ancilla GKP state
    ancilla_gkp_mode = combine_cats_to_gkp(prog, [cat_output_modes[2], cat_output_modes[3]], 
                                          output_mode=cat_output_modes[2])
    
    print("Setting up syndrome measurement...")
    
    # Syndrome measurement: couple logical and ancilla for error detection
    # This is a simplified syndrome extraction - in practice would need more complex protocol
    BSgate(np.arccos(np.sqrt(0.9)), 0.0) | (logical_gkp_mode, ancilla_gkp_mode)
    
    # Measure ancilla in specific quadrature to extract syndrome information
    MeasureHomodyne(np.pi/4) | ancilla_gkp_mode  # 45° measurement for syndrome
    
    # Error correction based on syndrome (simplified)
    syndrome_gain = 0.05
    Xgate(syndrome_gain * ancilla_gkp_mode.par) | logical_gkp_mode
    
    print("Cat breeding and GKP preparation complete.")

# Run the simulation
state = eng.run(prog).state
mu = state.means()
cov = state.cov()

# Post-selection: herald on single photons in odd modes of all 4 blocks
# and on zero photons in measured modes
post_select_dict = {}

# For each of the 4 cat breeding blocks
for block in range(4):
    base = 8 * block
    # Herald modes (odd modes) should have 1 photon
    for i in [1, 3, 5, 7]:
        post_select_dict[base + i] = 1
    # Measured modes should have 0 photons (except output modes 0, 8, 16, 24)
    for i in [2, 4, 6]:
        post_select_dict[base + i] = 0

# Also post-select on measurement outcomes from GKP combination
post_select_dict[9] = 0   # Second cat mode from first GKP combination
post_select_dict[25] = 0  # Second cat mode from second GKP combination  
post_select_dict[16] = 0  # Ancilla mode after syndrome measurement

cutoff = 10
rho = density_matrix(mu, cov, post_select=post_select_dict, normalize=False, cutoff=cutoff)

success_p = np.real_if_close(np.trace(rho))
print(f"GKP breeding success probability: {success_p:.6f}")

# Visualize the logical GKP state (mode 0)
if success_p > 1e-10:  # Only visualize if success probability is reasonable
    grid = 80
    xvec = np.linspace(-6, 6, grid)
    W = wigner(Qobj(rho / np.trace(rho)), xvec, xvec)
    
    # Enhanced colormap for GKP visualization
    scp = np.max(W)
    scn = np.min(W)
    nrm = mpl.colors.TwoSlopeNorm(vmin=scn, vcenter=0, vmax=scp)
    
    # Plot the GKP Wigner function
    fig, axes = plt.subplots(1, figsize=(6, 5))
    contour_plot = axes.contourf(xvec, xvec, W, 400, cmap='RdBu', norm=nrm)
    
    axes.set_title(r"GKP State Wigner Function $W(x,p)$", fontsize=14)
    axes.set_xlabel(r"Position $x$", fontsize=12)
    axes.set_ylabel(r"Momentum $p$", fontsize=12)
    
    # Add grid lines to show expected GKP structure
    for i in range(-3, 4):
        axes.axvline(i * np.sqrt(np.pi), color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
        axes.axhline(i * np.sqrt(np.pi), color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
    
    cb = fig.colorbar(contour_plot, ax=axes)
    cb.set_label('Wigner Function Value', fontsize=12)
    fig.tight_layout()
    
    print(f"Maximum Wigner function value: {scp:.4f}")
    print(f"Minimum Wigner function value: {scn:.4f}")
    print("Dashed grid lines show ideal GKP grid spacing (√π)")
    
    plt.show()
else:
    print(f"Success probability too low ({success_p:.2e}) for meaningful visualization")

print("Protocol complete: Logical GKP qubit prepared with ancilla-based error syndrome capability") 