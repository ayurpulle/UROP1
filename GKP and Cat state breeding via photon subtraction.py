# References used in producing these calculations:
# 1. Dakna, B., Anselmi, F., & Paris, M. G. A. (1999). Generating Schrödinger cat-like states by optical homodyne detection. Physical Review A, 59(2), 1658-1661.
# 2. Ourjoumtsev, A., Tualle-Brouri, R., Laurat, J., & Grangier, P. (2006). Generating Optical Schrödinger Cats from Squeezed Vacuum. Science, 312(5770), 83-86.
# 3. Etesse, J., Bouillard, M., Kanseri, B., & Tualle-Brouri, R. (2015). Experimental demonstration of continuous-variable cloning with phase-conjugate inputs. Physical Review Letters, 114(19), 193602. (Related to cat state manipulations)
# 4. Sychev, D. V., Ulanov, A. E., Pushkina, A. A., Richards, M. W., Fedorov, I. A., & Lvovsky, A. I. (2017). Enlargement of optical Schrödinger's cat states. Nature Photonics, 11(6), 379-382.
# 5. Hastrup, J., & Andersen, U. L. (2016). Breeding the optical Schrödinger's cat state. arXiv preprint arXiv:1609.08425.
# 6. Vasconcelos, H. M., Sanz, L., & Glancy, S. (2010). All-optical generation of states for “Encoding a qubit in an oscillator”. Physical Review A, 82(6), 063845.
# 7. Hastrup, J., Neergaard-Nielsen, J. S., & Andersen, U. L. (2022). Protocol for Generating Optical Gottesman-Kitaev-Preskill States with Cavity QED. Physical Review Letters, 128(17), 170503.
# 8. Campagne-Ibarcq, P., et al. (2024). Logical states for fault-tolerant quantum computation with propagating light. Science, 383(6679), 274-279.
# 9. Hillmann, T. (2024). Optical Gottesman-Kitaev-Preskill Qubit Generation via Approximate Squeezed Schrödinger Cat State Breeding. arXiv preprint arXiv:2409.06902.

import strawberryfields as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Parameters
N = 10  # Cutoff dimension for Fock space
r = 1  # Squeezing parameter
k = 2  # Number of breeding iterations
s = 0.5  # Additional squeezing for cat states before GKP generation
r_p = -1  # Squeezing for p-projection approximation (negative for p-squeezing)

# Define grid for Wigner functions
xvec = np.linspace(-8, 8, 201)
pvec = np.linspace(-8, 8, 201)

# Create program for squeezed vacuum
prog_sqz = sf.Program(1)
with prog_sqz.context as q:
    sf.ops.Squeezed(r) | q[0]

# Engine with Fock backend
eng = sf.Engine("fock", backend_options={"cutoff_dim": N})

# Run to get squeezed state
result_sqz = eng.run(prog_sqz)
ket_sqz = result_sqz.state.ket()  # Ket vector (N,)

# Compute Wigner function for squeezed vacuum
W_sqz = result_sqz.state.wigner(0, xvec, pvec)

# Annihilation operator matrix
a_matrix = np.zeros((N, N), dtype=complex)
for i in range(1, N):
    a_matrix[i-1, i] = np.sqrt(i)

# Apply photon subtraction to get small cat ket
ket_small = a_matrix @ ket_sqz

# Normalize
norm = np.sqrt(np.vdot(ket_small, ket_small).real)
ket_small /= norm

# Reset engine
eng.reset()

# Create program for the small cat state
prog_small = sf.Program(1)
with prog_small.context as q:
    sf.ops.Ket(ket_small) | q[0]

# Run to get the state object
result_small = eng.run(prog_small)
state_small = result_small.state

# Compute the Wigner function for initial cat (iteration 0)
W_0 = state_small.wigner(0, xvec, pvec)

# Initialize list for all Wigner functions
all_W = [W_sqz, W_0]

# Set current ket to small cat for breeding
current_ket = ket_small

# Perform k iterations of breeding
for iter in range(1, k + 1):
    # Breed current ket with a new small cat
    ket_two = np.tensordot(current_ket, ket_small, axes=0)  # (N, N)

    # Reset engine for 2-mode program
    eng.reset()

    prog_breed = sf.Program(2)
    with prog_breed.context as q:
        sf.ops.Ket(ket_two) | q
        sf.ops.BSgate(np.pi / 4, 0) | (q[0], q[1])

    result_breed = eng.run(prog_breed)
    ket_bred_two = result_breed.state.ket()  # (N, N)

    # Condition on n=0 in mode 1
    ket_conditional = ket_bred_two[:, 0]
    norm_cond = np.sqrt(np.sum(np.abs(ket_conditional)**2).real)
    ket_conditional /= norm_cond

    # Reset engine for 1-mode program
    eng.reset()

    # Create program for the bred state
    prog_bred = sf.Program(1)
    with prog_bred.context as q:
        sf.ops.Ket(ket_conditional) | q[0]

    result_bred = eng.run(prog_bred)
    state_bred = result_bred.state

    # Compute the Wigner function for this iteration
    W_bred = state_bred.wigner(0, xvec, pvec)
    all_W.append(W_bred)

    # Update current ket
    current_ket = ket_conditional

# Now produce GKP from two copies of the final bred cat state
# First, apply additional squeezing to the final cat
eng.reset()
prog_sq_cat = sf.Program(1)
with prog_sq_cat.context as q:
    sf.ops.Ket(current_ket) | q[0]
    sf.ops.Sgate(s) | q[0]

result_sq_cat = eng.run(prog_sq_cat)
ket_sq_cat = result_sq_cat.state.ket()

# Prepare two squeezed cats
ket_two_gkp = np.tensordot(ket_sq_cat, ket_sq_cat, axes=0)

# Reset engine for 2-mode program
eng.reset()

prog_gkp = sf.Program(2)
with prog_gkp.context as q:
    sf.ops.Ket(ket_two_gkp) | q
    sf.ops.BSgate(np.pi / 4, 0) | (q[0], q[1])

result_gkp = eng.run(prog_gkp)
ket_gkp_two = result_gkp.state.ket()  # (N, N)

# Generate the projection state for approximate homodyne p=0 (squeezed in p)
eng.reset()
prog_proj = sf.Program(1)
with prog_proj.context as q:
    sf.ops.Squeezed(r_p) | q[0]

result_proj = eng.run(prog_proj)
ket_proj = result_proj.state.ket()

# Condition on the projection state in mode 1
ket_gkp = ket_gkp_two @ ket_proj.conj()
norm_gkp = np.sqrt(np.vdot(ket_gkp, ket_gkp).real)
ket_gkp /= norm_gkp

# Reset engine for 1-mode program
eng.reset()

# Create program for the GKP state
prog_gkp_state = sf.Program(1)
with prog_gkp_state.context as q:
    sf.ops.Ket(ket_gkp) | q[0]

result_gkp_state = eng.run(prog_gkp_state)
state_gkp = result_gkp_state.state

# Compute the Wigner function for GKP
W_gkp = state_gkp.wigner(0, xvec, pvec)
all_W.append(W_gkp)

# Determine global vmin and vmax for consistent color scale
global_min = min(np.min(w) for w in all_W)
global_max = max(np.max(w) for w in all_W)
max_abs = max(abs(global_min), abs(global_max))
vmin = -max_abs
vmax = max_abs

# Plot the Wigner function for squeezed vacuum
plt.figure(figsize=(6, 6))
plt.contourf(xvec, pvec, W_sqz, 100, cmap='RdBu_r', vmin=vmin, vmax=vmax)
plt.gca().set_aspect('equal')
plt.colorbar()
plt.xlabel('X (Position Quadrature)')
plt.ylabel('P (Momentum Quadrature)')
plt.title('Wigner Function of Squeezed Vacuum')
plt.savefig('wigner_squeezed_strawberry.png')
plt.show()

# Save the raw data for squeezed vacuum
np.savetxt('xvec_sqz_strawberry.txt', xvec)
np.savetxt('pvec_sqz_strawberry.txt', pvec)
np.savetxt('wigner_data_sqz_strawberry.txt', W_sqz)

# Plot the Wigner function for initial cat (iteration 0)
plt.figure(figsize=(6, 6))
plt.contourf(xvec, pvec, W_0, 100, cmap='RdBu_r', vmin=vmin, vmax=vmax)
plt.gca().set_aspect('equal')
plt.colorbar()
plt.xlabel('X (Position Quadrature)')
plt.ylabel('P (Momentum Quadrature)')
plt.title('Wigner Function of Initial Schrödinger Cat State \n(Iteration 0)')
plt.savefig('wigner_cat_iter0_strawberry.png')
plt.show()

# Save the raw data for initial cat
np.savetxt('xvec_iter0_strawberry.txt', xvec)
np.savetxt('pvec_iter0_strawberry.txt', pvec)
np.savetxt('wigner_data_iter0_strawberry.txt', W_0)

# Now plot for each breeding iteration
W_iter = all_W[2:-1]  # W for iterations 1 to k, excluding GKP
for iter in range(1, k + 1):
    W = W_iter[iter - 1]
    plt.figure(figsize=(6, 6))
    plt.contourf(xvec, pvec, W, 100, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.xlabel('X (Position Quadrature)')
    plt.ylabel('P (Momentum Quadrature)')
    plt.title(f'Wigner Function of Bred Schrödinger Cat State \n(Iteration {iter})')
    plt.savefig(f'wigner_cat_bred_iter{iter}_strawberry.png')
    plt.show()

    # Save the raw data for this iteration
    np.savetxt(f'xvec_bred_iter{iter}_strawberry.txt', xvec)
    np.savetxt(f'pvec_bred_iter{iter}_strawberry.txt', pvec)
    np.savetxt(f'wigner_data_bred_iter{iter}_strawberry.txt', W)

# Plot the Wigner function for GKP
plt.figure(figsize=(6, 6))
plt.contourf(xvec, pvec, W_gkp, 100, cmap='RdBu_r', vmin=vmin, vmax=vmax)
plt.gca().set_aspect('equal')
plt.colorbar()
plt.xlabel('X (Position Quadrature)')
plt.ylabel('P (Momentum Quadrature)')
plt.title('Wigner Function of Approximate GKP State')
plt.savefig('wigner_gkp_strawberry.png')
plt.show()

# Save the raw data for GKP
np.savetxt('xvec_gkp_strawberry.txt', xvec)
np.savetxt('pvec_gkp_strawberry.txt', pvec)
np.savetxt('wigner_data_gkp_strawberry.txt', W_gkp)

# Compute marginal probability distributions for GKP
dx = xvec[1] - xvec[0]
dp = pvec[1] - pvec[0]
Pr_x = np.sum(W_gkp, axis=1) * dp
Pr_p = np.sum(W_gkp, axis=0) * dx

# Plot marginal distributions
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(xvec, Pr_x)
axs[0].set_xlabel('X (Position Quadrature)')
axs[0].set_ylabel('Probability Density')
axs[0].set_title('Marginal Pr(X)')
axs[1].plot(pvec, Pr_p)
axs[1].set_xlabel('P (Momentum Quadrature)')
axs[1].set_ylabel('Probability Density')
axs[1].set_title('Marginal Pr(P)')
plt.suptitle('Marginal Probability Distributions of Approximate GKP State')
plt.savefig('marginal_dist_gkp_strawberry.png')
plt.show()

# Save the marginal data
np.savetxt('pr_x_gkp_strawberry.txt', Pr_x)
np.savetxt('pr_p_gkp_strawberry.txt', Pr_p)