import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import matplotlib.pyplot as plt

# Helper function to run and draw circuit
def run_and_draw(prog, title):
    """Runs the program and draws the circuit, then prints a description."""
    print(f"\n--- {title} ---")
    print("Circuit Design:")
    # Draw the circuit directly, which prints to console or can be saved.
    # For display in an environment like this, it's often best to print the string representation.
    circuit_str = prog.draw_circuit()
    print(circuit_str)
    print("\n")

    # For more complex states, we might want to run the engine and inspect properties
    # For simplicity, we'll just draw the circuit as requested.
    # eng = sf.Engine(backend="gaussian") # Using gaussian backend for these examples
    # results = eng.run(prog)
    # print("State created successfully (results not displayed for brevity).")


# --- 1. Vacuum State ---
# The vacuum state is the ground state of the electromagnetic field,
# with no photons present. All modes in Strawberry Fields programs
# start in the vacuum state by default.
print("Step 1: Creating the Vacuum State")
vacuum_prog = sf.Program(1) # One quantum mode
# No operations needed as it starts in vacuum
run_and_draw(vacuum_prog, "Vacuum State")


# --- 2. Squeezed Vacuum State ---
# A squeezed vacuum state is created by applying a squeezing operation
# to the vacuum state. This reduces the quantum noise in one quadrature
# at the expense of increasing it in the conjugate quadrature.
print("Step 2: Creating the Squeezed Vacuum State")
squeezed_vacuum_prog = sf.Program(1)
r_squeeze = 1.0 # Squeezing parameter (strength)
phi_squeeze = 0.0 # Squeezing angle (0.0 for position squeezing)
with squeezed_vacuum_prog.context as q:
    S2(r_squeeze, phi_squeeze) | q[0] # Apply squeezing gate to mode 0
run_and_draw(squeezed_vacuum_prog, "Squeezed Vacuum State")


# --- 3. Cat State (Approximate using Gaussian operations) ---
# A Cat state (e.g., |alpha> + |-alpha>) is a superposition of two
# macroscopically distinct coherent states. It is a non-Gaussian state.
# While a perfect Cat state requires non-linear operations or post-selection,
# we can illustrate a common method involving interference of coherent states.
# This circuit shows the preparation of two coherent states and their interference.
# To get a single-mode Cat state, one would typically perform a measurement
# on one output mode and post-select on a specific outcome (e.g., vacuum).
print("Step 3: Creating an Approximate Cat State (using interference)")
cat_prog = sf.Program(2) # Two modes for interference
alpha_coherent = 2.0 # Displacement parameter for coherent states
theta_bs = np.pi / 4 # Beam splitter angle (pi/4 for 50:50)
phi_bs = 0.0 # Beam splitter phase
with cat_prog.context as q:
    # Prepare two coherent states
    Dgate(alpha_coherent) | q[0] # Displace mode 0 to |alpha>
    Dgate(-alpha_coherent) | q[1] # Displace mode 1 to |-alpha>
    # Interfere them on a beam splitter
    BSgate(theta_bs, phi_bs) | (q[0], q[1])
    # After the beam splitter, if one mode is measured in vacuum,
    # the other mode is projected into an approximate Cat state.
    # This projection step is not explicitly shown as a gate here.
run_and_draw(cat_prog, "Approximate Cat State (Interference Method)")


# --- 4. Squeezed Cat State (Approximate) ---
# A Squeezed Cat state is a Cat state that has also undergone squeezing.
# We take the approximate Cat state preparation and add a squeezing gate
# to one of the output modes before a conceptual measurement.
print("Step 4: Creating an Approximate Squeezed Cat State")
squeezed_cat_prog = sf.Program(2)
r_squeeze_cat = 0.8 # Squeezing parameter for the Cat state
with squeezed_cat_prog.context as q:
    # Prepare two coherent states (same as Cat state)
    Dgate(alpha_coherent) | q[0]
    Dgate(-alpha_coherent) | q[1]
    # Interfere them on a beam splitter
    BSgate(theta_bs, phi_bs) | (q[0], q[1])
    # Apply squeezing to one of the output modes (where the Cat state would be formed)
    S2(r_squeeze_cat, phi_squeeze) | q[0] # Squeeze the potential Cat mode
run_and_draw(squeezed_cat_prog, "Approximate Squeezed Cat State")


# --- 5. GKP State (Approximate) ---
# A GKP (Gottesman-Kitaev-Preskill) state is a non-Gaussian state that
# approximates a "quantum comb" in phase space, forming a superposition
# of infinitely many displaced vacuum states. It's crucial for error correction
# and high-precision sensing.
# Creating a high-fidelity GKP state is extremely challenging and typically
# involves complex non-linear operations, feed-forward, and error correction.
# Here, we show a conceptual circuit that *starts* with a highly squeezed state,
# which is a fundamental building block. Further operations (e.g., conditional
# displacements based on measurements) would be required to build the full comb structure.
print("Step 5: Creating an Approximate GKP State (Conceptual Building Block)")
gkp_prog = sf.Program(1)
r_gkp_squeeze = 2.5 # Very strong squeezing for initial GKP component
with gkp_prog.context as q:
    S2(r_gkp_squeeze, 0.0) | q[0] # Apply very strong squeezing
    # To form a true GKP state, this squeezed state would then be
    # interfered with other states and undergo conditional displacements
    # or non-linear gates, often involving ancillary modes and measurements.
    # For example, one might apply a series of Dgates conditionally
    # based on measurements of an ancillary mode, effectively creating
    # a superposition of displaced squeezed states.
    # This simple circuit only represents the initial squeezing step.
run_and_draw(gkp_prog, "Approximate GKP State (Initial Squeezing Component)")

print("--- End of State Generation Examples ---")
