import strawberryfields as sf
import numpy as np
import matplotlib.pyplot as plt

# Initialize engine with Fock backend for 4 modes (more manageable)
eng = sf.Engine("fock", backend_options={"cutoff_dim": 12})

# Function to plot Wigner function
def plot_wigner_stage(state, mode, stage_name, xvec, pvec):
    wigner = state.wigner(mode, xvec, pvec)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 2D contour plot
    X, P = np.meshgrid(xvec, pvec)
    contour = ax1.contourf(X, P, wigner, levels=50, cmap='RdBu_r')
    ax1.set_xlabel('Position (x)')
    ax1.set_ylabel('Momentum (p)')
    ax1.set_title(f'{stage_name} - Mode {mode}')
    plt.colorbar(contour, ax=ax1, label='W(x,p)')
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, P, wigner, cmap='RdBu_r', alpha=0.8)
    ax2.set_xlabel('Position (x)')
    ax2.set_ylabel('Momentum (p)')
    ax2.set_zlabel('W(x,p)')
    ax2.set_title(f'{stage_name} - 3D View')
    
    plt.tight_layout()
    plt.show()
    
    print(f"{stage_name} - Mode {mode}:")
    print(f"  Max Wigner value: {np.max(wigner):.6f}")
    print(f"  Min Wigner value: {np.min(wigner):.6f}")
    print(f"  Value at origin: {wigner[50, 50]:.6f}\n")

# Define coordinate ranges for Wigner function
xvec = np.linspace(-5, 5, 100)
pvec = np.linspace(-5, 5, 100)

# Stage 1: Initial vacuum state
print("=== STAGE 1: Initial Vacuum State ===")
prog1 = sf.Program(4)
with prog1.context as q:
    sf.ops.Vacuum() | q[0]
    sf.ops.Vacuum() | q[1]
    sf.ops.Vacuum() | q[2]
    sf.ops.Vacuum() | q[3]

result1 = eng.run(prog1)
plot_wigner_stage(result1.state, 0, "Stage 1: Vacuum State", xvec, pvec)

# Stage 2: Create coherent states
print("=== STAGE 2: Coherent States ===")
prog2 = sf.Program(4)
with prog2.context as q:
    sf.ops.Coherent(1.5) | q[0]   # |α⟩ 
    sf.ops.Coherent(-1.5) | q[1]  # |-α⟩
    sf.ops.Vacuum() | q[2]
    sf.ops.Vacuum() | q[3]

result2 = eng.run(prog2)
plot_wigner_stage(result2.state, 0, "Stage 2: Coherent State (+α)", xvec, pvec)
plot_wigner_stage(result2.state, 1, "Stage 2: Coherent State (-α)", xvec, pvec)

# Stage 3: Create cat state using beam splitter
print("=== STAGE 3: Cat State Creation ===")
prog3 = sf.Program(4)
with prog3.context as q:
    sf.ops.Coherent(1.8) | q[0]   
    sf.ops.Coherent(-1.8) | q[1]  
    sf.ops.Vacuum() | q[2]
    sf.ops.Vacuum() | q[3]
    
    # Create cat state through 50:50 beam splitter
    sf.ops.BSgate(np.pi/4, 0) | (q[0], q[2])  # Creates superposition

result3 = eng.run(prog3)
plot_wigner_stage(result3.state, 0, "Stage 3: Cat State (Mode 0)", xvec, pvec)
plot_wigner_stage(result3.state, 2, "Stage 3: Cat State (Mode 2)", xvec, pvec)

# Stage 4: Apply squeezing to cat state
print("=== STAGE 4: Squeezed Cat State ===")
prog4 = sf.Program(4)
with prog4.context as q:
    sf.ops.Coherent(1.8) | q[0]   
    sf.ops.Coherent(-1.8) | q[1]  
    sf.ops.Vacuum() | q[2]
    sf.ops.Vacuum() | q[3]
    
    # Create cat state
    sf.ops.BSgate(np.pi/4, 0) | (q[0], q[2])
    
    # Apply squeezing to the cat state
    sf.ops.Sgate(0.5, 0) | q[0]      # Squeeze in x direction
    sf.ops.Sgate(0.5, np.pi/2) | q[2] # Squeeze in p direction

result4 = eng.run(prog4)
plot_wigner_stage(result4.state, 0, "Stage 4: Squeezed Cat (Mode 0)", xvec, pvec)
plot_wigner_stage(result4.state, 2, "Stage 4: Squeezed Cat (Mode 2)", xvec, pvec)

# Stage 5: Breed cats with strong squeezing
print("=== STAGE 5: Cat Breeding with Strong Squeezing ===")
prog5 = sf.Program(4)
with prog5.context as q:
    sf.ops.Coherent(2.0) | q[0]   
    sf.ops.Coherent(-2.0) | q[1]  
    sf.ops.Vacuum() | q[2]
    sf.ops.Vacuum() | q[3]
    
    # Create two cat states
    sf.ops.BSgate(np.pi/4, 0) | (q[0], q[2])  # First cat
    sf.ops.BSgate(np.pi/4, 0) | (q[1], q[3])  # Second cat
    
    # Strong squeezing to breed the cats
    sf.ops.Sgate(0.8, 0) | q[0]       # Strong x-squeezing
    sf.ops.Sgate(0.8, np.pi/2) | q[2] # Strong p-squeezing
    sf.ops.Sgate(0.6, np.pi/4) | q[1] # Diagonal squeezing
    sf.ops.Sgate(0.6, 3*np.pi/4) | q[3] # Anti-diagonal squeezing
    
    # Entangle the cat states
    sf.ops.BSgate(np.pi/6, np.pi/4) | (q[0], q[1])

result5 = eng.run(prog5)
plot_wigner_stage(result5.state, 0, "Stage 5: Bred Cat State (Mode 0)", xvec, pvec)
plot_wigner_stage(result5.state, 1, "Stage 5: Bred Cat State (Mode 1)", xvec, pvec)
plot_wigner_stage(result5.state, 2, "Stage 5: Bred Cat State (Mode 2)", xvec, pvec)
plot_wigner_stage(result5.state, 3, "Stage 5: Bred Cat State (Mode 3)", xvec, pvec)

# Stage 6: Final evolution with maximum non-classicality
print("=== STAGE 6: Maximum Non-Classical Cat Breeding ===")
prog6 = sf.Program(4)
with prog6.context as q:
    sf.ops.Coherent(2.2) | q[0]   
    sf.ops.Coherent(-2.2) | q[1]  
    sf.ops.Vacuum() | q[2]
    sf.ops.Vacuum() | q[3]
    
    # Initial cat creation
    sf.ops.BSgate(np.pi/4, 0) | (q[0], q[2])
    sf.ops.BSgate(np.pi/4, 0) | (q[1], q[3])
    
    # Maximum squeezing for strong non-classicality
    sf.ops.Sgate(1.0, 0) | q[0]        # Very strong x-squeezing
    sf.ops.Sgate(1.0, np.pi/2) | q[2]  # Very strong p-squeezing
    sf.ops.Sgate(0.8, np.pi/3) | q[1]  # Strong diagonal squeezing
    sf.ops.Sgate(0.8, 4*np.pi/3) | q[3] # Strong anti-diagonal
    
    # Complex entanglement pattern
    sf.ops.BSgate(np.pi/8, np.pi/3) | (q[0], q[1])
    sf.ops.BSgate(np.pi/8, 2*np.pi/3) | (q[2], q[3])
    sf.ops.BSgate(np.pi/12, np.pi/6) | (q[0], q[3])

result6 = eng.run(prog6)
plot_wigner_stage(result6.state, 0, "Stage 6: Ultimate Bred Cat (Mode 0)", xvec, pvec)
plot_wigner_stage(result6.state, 1, "Stage 6: Ultimate Bred Cat (Mode 1)", xvec, pvec)
plot_wigner_stage(result6.state, 2, "Stage 6: Ultimate Bred Cat (Mode 2)", xvec, pvec)
plot_wigner_stage(result6.state, 3, "Stage 6: Ultimate Bred Cat (Mode 3)", xvec, pvec)

print("=== SUMMARY ===")
print("Cat breeding evolution complete!")
print("You should observe:")
print("1. Vacuum → Coherent → Cat → Squeezed Cat → Bred Cat")
print("2. Progressive appearance of negative Wigner values")
print("3. Complex interference patterns indicating non-classicality")
print("4. Entanglement-induced correlations between modes")
print("5. Maximum non-classical features in the final stage")

