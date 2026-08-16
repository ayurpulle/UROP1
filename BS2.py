import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from typing import Tuple, List, Dict
from scipy.stats import norm
from scipy.optimize import minimize_scalar


class PhotonicGKPSensor:
    """
    Photonic implementation of GKP displacement sensing using beamsplitters
    instead of controlled operations.
    """
    
    def __init__(self, eps: float = 0.01, cutoff_dim: int = 50):
        self.eps = eps
        self.cutoff_dim = cutoff_dim
        self.ls = np.sqrt(2 * np.pi)  # Modular length for GKP
        
        # For adaptive Bayesian inference
        self.measurements = []
        self.phases = []
        self.posterior_samples = []
        
    def single_measurement_round(self, delta_x: float, phase: float, 
                               bs_angle: float = np.pi/90) -> Tuple[int, bool, Dict]:
        """
        Perform one round of measurement with phase encoding.
        
        Args:
            delta_x: Unknown displacement to sense
            phase: Adaptive phase for this measurement
            bs_angle: Beamsplitter angle (small for weak coupling)
            
        Returns:
            Tuple of (binary_outcome, success_flag, debug_info)
        """
        prog = sf.Program(2)
        
        with prog.context as q:
            # Step 1: Prepare GKP state
            print(f"\n--- Measurement Round ---")
            print(f"Step 1: Preparing GKP state with ε={self.eps}")
            GKP(epsilon=self.eps) | q[0]
            Vacuum() | q[1]
            
            # Step 2: Apply unknown displacement (this is what we're sensing)
            print(f"Step 2: Unknown displacement δx={delta_x:.3f} applied to GKP")
            Dgate(delta_x) | q[0]
            
            # Step 3: Apply phase shift for adaptive sensing
            # This encodes the measurement basis
            print(f"Step 3: Applying adaptive phase rotation φ={phase:.3f}")
            Rgate(phase) | q[0]  # Rotation in phase space
            
            # Step 4: Beamsplitter interaction (replaces controlled operation)
            print(f"Step 4: Beamsplitter coupling (angle={bs_angle:.4f})")
            BSgate(bs_angle, 0) | (q[0], q[1])
            
            # Step 5: Measure ancilla in x-quadrature
            MeasureX | q[1]
        
        # Run the circuit
        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff_dim})
        result = eng.run(prog)
        x_measured = result.samples[0][0]
        
        # Calculate the coupling factor for calibration
        coupling_factor = np.sin(bs_angle)
        print(f"Step 5: Ancilla x-measurement = {x_measured:.4f}")
        print(f"        BS coupling factor = sin({bs_angle:.4f}) = {coupling_factor:.4f}")
        print(f"        Inferred GKP displacement ≈ {x_measured/coupling_factor:.4f}")
        
        # Step 6: Convert to binary outcome based on sign
        # This implements pseudo-homodyne detection
        binary_outcome = 0 if x_measured > 0 else 1
        print(f"Step 6: Binary outcome = {binary_outcome} (based on sign)")
        
        # Step 7: Post-selection based on measurement magnitude
        # Accept if close to zero (near a GKP grid point)
        threshold = self.ls / 2  # Quarter of grid spacing
        success = abs(x_measured) < threshold
        print(f"Step 7: Post-selection {'SUCCESS' if success else 'FAIL'} (|x| < {threshold:.3f})")
        
        debug_info = {
            'x_measured': x_measured,
            'threshold': threshold,
            'phase_applied': phase,
            'coupling_factor': coupling_factor,
            'inferred_displacement': x_measured / coupling_factor
        }
        
        return binary_outcome, success, debug_info
    
    def compute_measurement_probability(self, delta_x: float, phase: float, 
                                      outcome: int, bs_angle: float = np.pi/90) -> float:
        """
        Calculate the probability of getting a specific binary outcome.
        
        This models the measurement statistics for Bayesian inference.
        """
        # For small BS angle θ, the beamsplitter coupling is:
        # x_ancilla ≈ sin(θ) * x_GKP
        # For θ = π/90 ≈ 0.0349 rad, sin(θ) ≈ 0.0349
        
        # The scaling factor for displacement transfer
        coupling_factor = np.sin(bs_angle)
        
        # Expected ancilla displacement for given delta_x and phase
        # After rotation by φ, the effective displacement measured is:
        expected_ancilla = coupling_factor * delta_x * np.cos(phase)
        
        # Model measurement statistics
        # For GKP, the measurement has periodic structure
        gkp_phase = self.ls * delta_x
        
        # Probability of measuring positive x (outcome 0)
        visibility = 0.9  # Reduced from 1 due to finite squeezing
        p_positive = 0.5 * (1 + visibility * np.cos(gkp_phase + phase))
        
        return p_positive if outcome == 0 else (1 - p_positive)
    
    def adaptive_phase_selection(self, iteration: int) -> float:
        """
        Select the next measurement phase adaptively based on current knowledge.
        """
        if iteration == 0:
            # First measurement: no prior information
            print("\nAdaptive phase selection: First measurement, using φ=0")
            return 0.0
        
        if len(self.posterior_samples) < 10:
            # Early stage: coarse search with uniform phases
            phase = (iteration * 2 * np.pi) / 10
            print(f"\nAdaptive phase selection: Coarse search, using φ={phase:.3f}")
            return phase
        
        # Later stage: refined search based on posterior
        # Estimate current best guess
        current_estimate = np.mean(self.posterior_samples[-100:])
        
        # Choose phase for maximum sensitivity
        # Phase that puts us at steepest part of cosine
        optimal_phase = -self.ls * current_estimate + np.pi/2
        
        print(f"\nAdaptive phase selection: Current estimate δx≈{current_estimate:.4f}")
        print(f"Selecting phase φ={optimal_phase:.3f} for maximum sensitivity")
        
        return optimal_phase
    
    def bayesian_update(self, delta_x_samples: np.ndarray, measurement: int, 
                       phase: float) -> np.ndarray:
        """
        Update posterior samples using Bayesian inference.
        """
        print(f"\nBayesian update with measurement={measurement}, phase={phase:.3f}")
        
        # Calculate likelihood for each sample
        likelihoods = np.array([
            self.compute_measurement_probability(dx, phase, measurement) 
            for dx in delta_x_samples
        ])
        
        # Resample according to likelihood (particle filter)
        weights = likelihoods / np.sum(likelihoods)
        indices = np.random.choice(len(delta_x_samples), size=len(delta_x_samples), 
                                 p=weights, replace=True)
        
        new_samples = delta_x_samples[indices]
        
        # Add small noise for diversity
        new_samples += np.random.normal(0, 0.001, size=len(new_samples))
        
        print(f"Posterior mean: {np.mean(new_samples):.4f}, std: {np.std(new_samples):.4f}")
        
        return new_samples
    
    def run_adaptive_protocol(self, true_delta_x: float, max_rounds: int = 100) -> Dict:
        """
        Run the complete adaptive sensing protocol.
        """
        print("="*60)
        print("STARTING ADAPTIVE GKP DISPLACEMENT SENSING PROTOCOL")
        print("="*60)
        print(f"True displacement to sense: δx = {true_delta_x}")
        print(f"Maximum rounds: {max_rounds}")
        
        # Initialize with uniform prior
        print("\nInitializing uniform prior over [-1, 1]")
        posterior_samples = np.random.uniform(-1, 1, size=1000)
        
        successful_measurements = 0
        total_attempts = 0
        
        # Main measurement loop
        for i in range(max_rounds):
            print(f"\n{'='*40}")
            print(f"ROUND {i+1}/{max_rounds}")
            print(f"{'='*40}")
            
            # Select phase adaptively
            phase = self.adaptive_phase_selection(i)
            
            # Perform measurement
            outcome, success, debug = self.single_measurement_round(true_delta_x, phase)
            total_attempts += 1
            
            if success:
                successful_measurements += 1
                
                # Update posterior
                posterior_samples = self.bayesian_update(posterior_samples, outcome, phase)
                self.posterior_samples.extend(posterior_samples[:10])  # Store subset
                
                # Store measurement data
                self.measurements.append(outcome)
                self.phases.append(phase)
                
                # Check convergence
                if np.std(posterior_samples) < 0.01:
                    print(f"\nCONVERGED after {i+1} rounds!")
                    break
        
        # Final results
        final_estimate = np.mean(posterior_samples)
        final_uncertainty = np.std(posterior_samples)
        success_rate = successful_measurements / total_attempts
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        return {
            'estimate': final_estimate,
            'uncertainty': final_uncertainty,
            'success_rate': success_rate,
            'measurements': successful_measurements,
            'total_attempts': total_attempts,
            'true_value': true_delta_x,
            'error': abs(final_estimate - true_delta_x)
        }


def analyze_results(results: Dict):
    """
    Analyze and display the sensing results.
    """
    print(f"\nTrue displacement:      δx = {results['true_value']:.4f}")
    print(f"Estimated displacement: δx = {results['estimate']:.4f} ± {results['uncertainty']:.4f}")
    print(f"Estimation error:       |error| = {results['error']:.4f}")
    print(f"\nMeasurement statistics:")
    print(f"  Success rate:         {results['success_rate']:.1%}")
    print(f"  Successful measurements: {results['measurements']}/{results['total_attempts']}")
    
    # Estimate SQL comparison (simplified)
    n_photons = 1 / (results['uncertainty']**2)  # Rough estimate
    sql_uncertainty = 1 / np.sqrt(n_photons)
    improvement = sql_uncertainty / results['uncertainty']
    
    print(f"\nSQL Comparison (approximate):")
    print(f"  Effective photon number: n ≈ {n_photons:.1f}")
    print(f"  SQL uncertainty:        σ_SQL ≈ {sql_uncertainty:.4f}")
    print(f"  GKP uncertainty:        σ_GKP = {results['uncertainty']:.4f}")
    print(f"  Improvement factor:     {improvement:.1f}x")


# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sensor
    sensor = PhotonicGKPSensor(eps=0.01, cutoff_dim=50)
    
    # Unknown displacement to sense
    true_displacement = 0.1
    
    # Run adaptive protocol with detailed output
    results = sensor.run_adaptive_protocol(true_displacement, max_rounds=100)
    
    # Analyze results
    analyze_results(results)