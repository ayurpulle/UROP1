import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

# This function remains the same
def create_gkp_state(eps: float = 0.01, cutoff_dim: int = 50):
    """
    Create a GKP state using Strawberry Fields.
    """
    prog = sf.Program(1)
    with prog.context as q:
        kwargs = dict(epsilon=eps, ampl_cutoff=1e-12, representation="real", shape="square")
        GKP(**kwargs) | q[0]
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = eng.run(prog)
    return result.state

# --- MODIFIED & NEW FUNCTIONS FOR ADAPTIVE SENSING ---

def adaptive_sensing_round(
    current_estimate: float, true_delta_x: float, threshold: float = 1.0, cutoff_dim: int = 50
) -> Tuple[float, bool]:
    """
    Implements a SINGLE round of an ADAPTIVE displacement sensing protocol.

    Args:
        current_estimate: The current best guess for the displacement. Used for feedback.
        true_delta_x: The actual "unknown" displacement (for simulation purposes).
        threshold: Post-selection threshold for the ancilla measurement.
        cutoff_dim: Cutoff dimension for the Fock space simulation.

    Returns:
        Tuple of (measurement_of_residual, success_flag)
    """
    prog = sf.Program(2)

    with prog.context as q:
        # Stage 1: Initialize states and apply feedback
        GKP(epsilon=0.01) | q[0]
        Vacuum() | q[1]
        
        # --- ADAPTIVE STEP 1: Apply feedback based on current estimate ---
        # We actively try to cancel the displacement we expect to see.
        Dgate(-current_estimate) | q[0]
        
        # The "unknown" physical displacement occurs
        Dgate(true_delta_x) | q[0]
        
        # The net displacement on the GKP state is now (true_delta_x - current_estimate)

        # Stage 2: Interaction to read out the RESIDUAL displacement
        BSgate(np.pi/4, 0) | (q[0], q[1])

        # Stage 3: Measure the ancilla's x-quadrature
        MeasureX | q[1]

    # Run the program
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = eng.run(prog)
    # This measurement now corresponds to the RESIDUAL error, not the total displacement
    x_measured_residual = result.samples[0][0]

    # Stage 4: Post-selection
    if np.abs(x_measured_residual) < threshold:
        return x_measured_residual, True
    else:
        return x_measured_residual, False

def run_adaptive_displacement_protocol(
    true_delta_x: float, num_rounds: int = 1000, learning_rate: float = 0.1
) -> Tuple[float, List[float], float]:
    """
    Runs the full adaptive sensing loop.

    Args:
        true_delta_x: The unknown physical displacement.
        num_rounds: The number of measurement rounds to perform.
        learning_rate: A factor (0 to 1) controlling how much each new measurement
                       updates the estimate. Smaller values lead to smoother convergence.

    Returns:
        Tuple of (final_estimate, estimate_history, success_rate)
    """
    # Start with no knowledge: our initial estimate is 0.
    estimated_delta_x = 0.0
    
    estimate_history = [estimated_delta_x]
    successful_rounds = 0
    threshold = np.sqrt(np.pi) / 2

    for _ in range(num_rounds):
        # Perform one round, passing in our current best guess
        residual_measurement, success = adaptive_sensing_round(estimated_delta_x, true_delta_x, threshold)
        
        if success:
            successful_rounds += 1
            
            # --- ADAPTIVE STEP 2: Update our estimate ---
            # 1. Convert the measured ancilla residual to a data qubit residual
            residual_on_gkp = residual_measurement * np.sqrt(2)
            
            # 2. Update our main estimate by adding a fraction of the measured error
            estimated_delta_x += learning_rate * residual_on_gkp
        
        # Store the new estimate for plotting, regardless of success, to see the full process
        estimate_history.append(estimated_delta_x)

    success_rate = successful_rounds / num_rounds
    return estimated_delta_x, estimate_history, success_rate


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # We can now test a larger displacement, where a simple strategy might fail
    unknown_displacement = 0.2

    print(f"Simulating an ADAPTIVE protocol to estimate delta_x = {unknown_displacement}")
    
    num_measurements = 100
    print(f"\nRunning {num_measurements} adaptive measurement rounds...")
    
    final_estimate, history, success_rate = run_adaptive_displacement_protocol(
        unknown_displacement, num_measurements, learning_rate=0.1
    )

    print(f"\nProtocol Results:")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Number of successful measurements: {int(success_rate * num_measurements)}")

    print("\n--- Final Parameter Estimation ---")
    print(f"Original displacement (delta_x): {unknown_displacement}")
    print(f"Final adaptive estimate:         {final_estimate:.4f}")
    
    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Adaptive Estimate')
    plt.axhline(y=unknown_displacement, color='r', linestyle='--', label='True Value')
    plt.xlabel("Measurement Round")
    plt.ylabel("Estimated Displacement")
    plt.title("Adaptive Sensing Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()