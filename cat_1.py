# Import relevant libraries
import strawberryfields as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, colorbar

# Set the scale for phase space
sf.hbar = 1
scale = np.sqrt(sf.hbar * np.pi)

# Create a GKP |0> state
prog = sf.Program(1)

with prog.context as q:
    sf.ops.GKP() | q

eng = sf.Engine("bosonic")
gkp_0 = eng.run(prog).state

# Create a GKP |1> state
prog = sf.Program(1)

with prog.context as q:
    sf.ops.GKP() | q
    sf.ops.Xgate(np.sqrt(np.pi * sf.hbar)) | q

eng = sf.Engine("bosonic")
gkp_1 = eng.run(prog).state

# Get the phase space coordinates of the delta functions for the two states
q_coords_0 = gkp_0.means().real[:, 0]
p_coords_0 = gkp_0.means().real[:, 1]
q_coords_1 = gkp_1.means().real[:, 0]
p_coords_1 = gkp_1.means().real[:, 1]

# Determine whether the delta functions are positively or negatively weighted
delta_sign_0 = np.sign(gkp_0.weights().real)
delta_sign_1 = np.sign(gkp_1.weights().real)

# Plot the locations and signs of the deltas
fig,ax = plt.subplots(1, 2)
ax[0].scatter(q_coords_0 / scale,
           p_coords_0 / scale,
           c=delta_sign_0,
           cmap=plt.cm.RdBu, vmin=-1.5, vmax=1.5)
ax[1].scatter(q_coords_1 / scale,
           p_coords_1 / scale,
           c=delta_sign_0,
           cmap=plt.cm.RdBu, vmin=-1.5, vmax=1.5)
for i in range(2):
    ax[i].set_xlim(-3.5, 3.5)
    ax[i].set_ylim(-3.5, 3.5)
    ax[i].set_xlabel(r'$q$ (units of $\sqrt{\pi\hbar}$ )', fontsize=15)
    ax[i].set_aspect("equal")
ax[0].set_title(r'$|0\rangle_{\rm gkp}$ Wigner function', fontsize=15)
ax[1].set_title(r'$|1\rangle_{\rm gkp}$ Wigner function', fontsize=15)
ax[0].set_ylabel(r'$p$ (units of $\sqrt{\pi\hbar}$ )', fontsize=15)
fig.tight_layout()
plt.show()


# Choose some values of epsilon
epsilons = [0.05, 0.1, 0.2, 5]

# Pick a region of phase space to plot
quad = np.linspace(-3.5, 3.5, 200) * scale

wigners = []

for epsilon in epsilons:

    # Create a GKP |0> state
    prog = sf.Program(1)

    with prog.context as q:
        sf.ops.GKP(epsilon=epsilon) | q

    eng = sf.Engine("bosonic")
    gkp = eng.run(prog).state

    # Calculate the Wigner function
    wigner = gkp.wigner(mode=0, xvec=quad, pvec=quad)
    wigners.append(wigner)

# Plot the results
fig, axs = plt.subplots(1, 5, figsize=(16, 4), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.05]})
cmap = plt.cm.RdBu
cmax = np.real_if_close(np.amax(np.array(wigners)))
norm = colors.Normalize(vmin=-cmax, vmax=cmax)
cb1 = colorbar.ColorbarBase(axs[4], cmap=cmap, norm=norm, orientation="vertical")
for i in range(4):
    axs[i].contourf(
        quad / scale,
        quad / scale,
        wigners[i],
        levels=60,
        cmap=plt.cm.RdBu,
        vmin=-cmax,
        vmax=cmax,
    )
    axs[i].set_title(r'$\epsilon =$'+str(epsilons[i]), fontsize=15)
    axs[i].set_xlabel(r'$q$ (units of $\sqrt{\pi\hbar}$ )', fontsize=15)
axs[0].set_ylabel(r'$p$ (units of $\sqrt{\pi\hbar}$ )', fontsize=15)
cb1.set_label("Wigner function", fontsize=15)
fig.tight_layout()
plt.show()