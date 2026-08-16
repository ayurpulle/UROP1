import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from typing import Tuple, List

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

# --- NEW FUNCTIONS FOR DISPLACEMENT SENSING ---

def displacement_sensing_interaction(delta_x: float, threshold: float = 1.0, cutoff_dim: int = 50) -> Tuple[float, bool]:
    """
    Implements one round of a displacement sensing protocol.

    Args:
        delta_x: The unknown physical displacement in the x-quadrature to be sensed.
        threshold: Threshold for post-selection success on the ancilla measurement.
        cutoff_dim: Cutoff dimension for the Fock space simulation.

    Returns:
        Tuple of (measurement_result, success_flag)
    """
    prog = sf.Program(2)  # 2 modes: GKP data mode and ancilla mode

    with prog.context as q:
        # Stage 1: Initialize GKP and apply the unknown displacement
        GKP(epsilon=0.01) | q[0]
        Vacuum() | q[1]
        
        # This is the physical process we are sensing
        Dgate(delta_x) | q[0]

        # Stage 2: Interaction to read out the displacement information
        # A 50-50 beamsplitter couples the data and ancilla modes.
        # This transfers information about the displacement on q[0] to q[1].
        BSgate(np.pi/180, 0) | (q[0], q[1])

        # Stage 3: Measure the ancilla to estimate the displacement
        # We measure the x-quadrature of the ancilla.
        MeasureX | q[1]

    # Run the program
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = eng.run(prog)
    x_measured = result.samples[0][0] # The x-quadrature measurement of mode 1

    # Stage 4: Post-selection
    # We only accept results where the ancilla measurement is small,
    # indicating the GKP state remained near a grid point.
    if np.abs(x_measured) < threshold:
        # The measurement itself is the valuable data for estimation
        return x_measured, True
    else:
        # Post-selection fails
        return x_measured, False

def run_displacement_sensing_protocol(delta_x: float, num_rounds: int = 1000) -> Tuple[List[float], float]:
    """
    Run multiple rounds of the displacement sensing protocol.

    Args:
        delta_x: The unknown physical displacement in the x-quadrature.
        num_rounds: The number of measurement rounds to perform.

    Returns:
        Tuple of (successful_measurements, success_rate)
    """
    successful_measurements = []
    # GKP states are defined on a grid with spacing sqrt(pi).
    # A good threshold is a fraction of that spacing.
    threshold = np.sqrt(np.pi) / 2 

    for _ in range(num_rounds):
        measurement, success = displacement_sensing_interaction(delta_x, threshold)
        if success:
            successful_measurements.append(measurement)

    success_rate = len(successful_measurements) / num_rounds
    return successful_measurements, success_rate


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # The "unknown" displacement we want to estimate
    unknown_displacement = 0.1

    print(f"Simulating a protocol to estimate an unknown displacement delta_x = {unknown_displacement}")
    
    # Run the protocol for many rounds to get good statistics
    num_measurements = 100
    print(f"\nRunning {num_measurements} measurement rounds...")
    measurements, success_rate = run_displacement_sensing_protocol(unknown_displacement, num_measurements)

    print(f"\nProtocol Results:")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Number of successful measurements: {len(measurements)}")

    if len(measurements) > 0:
        # The core of the estimation: the average of the ancilla measurements
        # is proportional to the original displacement.
        # For a 50-50 BS, the ancilla's average x is <x_a> = <x_d>/sqrt(2).
        # So, the estimated displacement is E[x_measured] * sqrt(2).
        estimated_mean = np.mean(measurements)
        estimated_displacement = estimated_mean * 10
        
        print("\n--- Parameter Estimation ---")
        print(f"Average of ancilla measurements (E[x_ancilla]): {estimated_mean:.4f}")
        print(f"Original displacement (delta_x):                 {unknown_displacement}")
        print(f"Estimated displacement (E[x_ancilla] * sqrt(2)): {estimated_displacement:.4f}")
        print("✓ Estimation complete.")
    else:
        print("\n⚠️ No successful measurements. Consider increasing num_rounds or the threshold.")