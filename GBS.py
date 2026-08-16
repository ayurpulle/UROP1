import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
from thewalrus import hafnian, tor
from thewalrus.quantum import density_matrix_element, state_vector
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import unitary_group

class GaussianBosonSampling:
    """
    Implementation of Gaussian Boson Sampling (GBS) using Strawberry Fields.
    
    The GBS device consists of:
    1. Squeezed vacuum states as input
    2. An N-mode interferometer
    3. Photon number resolving (PNR) detectors
    """
    
    def __init__(self, num_modes, cutoff_dim=10):
        """
        Initialize GBS circuit.
        
        Args:
            num_modes (int): Number of optical modes
            cutoff_dim (int): Fock space cutoff dimension
        """
        self.num_modes = num_modes
        self.cutoff_dim = cutoff_dim
        self.prog = sf.Program(num_modes)
        
    def create_circuit(self, squeezing_params, interferometer_params, 
                      displacement_params=None):
        """
        Create the GBS circuit with given parameters.
        
        Args:
            squeezing_params (array): Squeezing parameters for each mode
            interferometer_params (array): Parameters for the interferometer
            displacement_params (array, optional): Displacement parameters
        
        Returns:
            sf.Program: The quantum circuit
        """
        prog = sf.Program(self.num_modes)
        
        with prog.context as q:
            # Apply squeezing to each mode
            for i, r in enumerate(squeezing_params):
                Sgate(r) | q[i]
            
            # Apply displacements if provided
            if displacement_params is not None:
                for i, (alpha_r, alpha_i) in enumerate(displacement_params):
                    Dgate(alpha_r, alpha_i) | q[i]
            
            # Apply the interferometer using rectangular decomposition
            self._apply_interferometer(q, interferometer_params)
            
        return prog
    
    def _apply_interferometer(self, q, params):
        """
        Apply interferometer using rectangular decomposition.
        
        Args:
            q: Quantum registers
            params: Interferometer parameters (beam splitter angles and phases)
        """
        n = self.num_modes
        
        # Rectangular decomposition: alternating layers of beam splitters
        param_idx = 0
        
        # Odd layers
        for layer in range(n):
            for i in range(0, n-1, 2):
                if i + 1 < n and param_idx < len(params):
                    BSgate(params[param_idx], params[param_idx + 1]) | (q[i], q[i+1])
                    param_idx += 2
            
            # Even layers
            for i in range(1, n-1, 2):
                if i + 1 < n and param_idx < len(params):
                    BSgate(params[param_idx], params[param_idx + 1]) | (q[i], q[i+1])
                    param_idx += 2
    
    def run_sampling(self, circuit, num_samples=1000, backend='gaussian'):
        """
        Run GBS and collect samples.
        
        Args:
            circuit: The quantum circuit
            num_samples: Number of samples to collect
            backend: Strawberry Fields backend to use
        
        Returns:
            array: Photon number samples
        """
        eng = sf.Engine(backend)
        results = eng.run(circuit, shots=num_samples)
        
        samples = []
        for result in results:
            samples.append(result.samples)
        
        return np.array(samples)
    
    def calculate_probabilities(self, circuit, photon_pattern):
        """
        Calculate the probability of a specific photon number pattern.
        
        Args:
            circuit: The quantum circuit
            photon_pattern: List of photon numbers for each mode
        
        Returns:
            float: Probability of the pattern
        """
        # Run with Gaussian backend to get covariance matrix and means
        eng = sf.Engine('gaussian')
        state = eng.run(circuit).state
        
        # Get the covariance matrix and displacement vector
        cov = state.cov()
        mean = state.means()
        
        # Use thewalrus to calculate probability
        prob = density_matrix_element(mean, cov, photon_pattern, photon_pattern)
        
        return np.abs(prob)
    
    def prepare_gkp_state(self, target_state='zero', num_modes=3, max_photons=4):
        """
        Prepare approximate GKP (Gottesman-Kitaev-Preskill) states using GBS.
        Based on the method described in the project knowledge.
        
        Args:
            target_state: 'zero', 'one', 'plus', or 'minus'
            num_modes: Number of modes to use
            max_photons: Maximum total photon number
        
        Returns:
            dict: Optimized circuit parameters and fidelity
        """
        # Initialize random parameters
        num_params = 2 * num_modes * (num_modes - 1)  # For interferometer
        
        initial_params = {
            'squeezing': np.random.uniform(0, 1.5, num_modes),
            'interferometer': np.random.uniform(0, 2*np.pi, num_params),
            'postselection': self._generate_postselection_pattern(num_modes-1, max_photons)
        }
        
        # Define cost function (negative fidelity)
        def cost_function(params):
            squeez = params[:num_modes]
            inter = params[num_modes:]
            
            # Create circuit
            circuit = self.create_circuit(squeez, inter)
            
            # Calculate fidelity with target state
            # This is a simplified version - in practice you'd calculate
            # the actual fidelity with the GKP target state
            fidelity = self._calculate_gkp_fidelity(circuit, target_state, 
                                                   initial_params['postselection'])
            
            # Add probability weight to ensure reasonable success probability
            prob = self.calculate_probabilities(circuit, initial_params['postselection'])
            
            return -(fidelity + 0.1 * np.log(prob + 1e-10))
        
        # Optimize parameters
        x0 = np.concatenate([initial_params['squeezing'], 
                           initial_params['interferometer']])
        
        result = minimize(cost_function, x0, method='L-BFGS-B')
        
        # Extract optimized parameters
        opt_squeez = result.x[:num_modes]
        opt_inter = result.x[num_modes:]
        
        return {
            'squeezing': opt_squeez,
            'interferometer': opt_inter,
            'postselection': initial_params['postselection'],
            'fidelity': -result.fun
        }
    
    def _generate_postselection_pattern(self, n_measured, max_photons):
        """Generate a valid postselection pattern."""
        pattern = np.zeros(n_measured, dtype=int)
        remaining = max_photons
        
        for i in range(n_measured - 1):
            pattern[i] = np.random.randint(0, remaining + 1)
            remaining -= pattern[i]
        
        pattern[-1] = remaining
        return pattern
    
    def _calculate_gkp_fidelity(self, circuit, target_state, postselection):
        """
        Simplified GKP fidelity calculation.
        In practice, this would involve comparing with actual GKP states.
        """
        # This is a placeholder - actual implementation would calculate
        # fidelity with truncated GKP states
        return np.random.random()  # Placeholder
    
    def visualize_samples(self, samples):
        """
        Visualize the photon number distribution from samples.
        
        Args:
            samples: Array of photon number samples
        """
        fig, axes = plt.subplots(1, self.num_modes, figsize=(15, 4))
        
        if self.num_modes == 1:
            axes = [axes]
        
        for mode in range(self.num_modes):
            mode_samples = samples[:, mode]
            bins = np.arange(0, np.max(mode_samples) + 2) - 0.5
            
            axes[mode].hist(mode_samples, bins=bins, density=True, 
                          alpha=0.7, edgecolor='black')
            axes[mode].set_xlabel('Photon number')
            axes[mode].set_ylabel('Probability')
            axes[mode].set_title(f'Mode {mode}')
            axes[mode].set_xticks(range(int(np.max(mode_samples)) + 1))
        
        plt.tight_layout()
        plt.show()


# Example usage
def example_basic_gbs():
    """Basic GBS example with 4 modes."""
    # Initialize GBS with 4 modes
    gbs = GaussianBosonSampling(num_modes=4, cutoff_dim=10)
    
    # Set circuit parameters
    squeezing_params = [0.5, 0.3, 0.4, 0.6]  # Squeezing for each mode
    
    # Random unitary interferometer
    U = unitary_group.rvs(4)
    interferometer_params = []
    
    # Extract beam splitter parameters from unitary
    # (This is a simplified extraction - proper decomposition would be more complex)
    for i in range(4):
        for j in range(i+1, 4):
            theta = np.arccos(np.abs(U[i, i]))
            phi = np.angle(U[i, j])
            interferometer_params.extend([theta, phi])
    
    # Create and run circuit
    circuit = gbs.create_circuit(squeezing_params, interferometer_params)
    samples = gbs.run_sampling(circuit, num_samples=1000)
    
    # Visualize results
    print("Sample photon patterns (first 10):")
    print(samples[:10])
    
    gbs.visualize_samples(samples)
    
    # Calculate probability of specific pattern
    pattern = [1, 0, 1, 0]
    prob = gbs.calculate_probabilities(circuit, pattern)
    print(f"\nProbability of pattern {pattern}: {prob:.6f}")


def example_gkp_preparation():
    """Example of preparing GKP states using GBS."""
    # Initialize GBS for GKP state preparation
    gbs = GaussianBosonSampling(num_modes=3, cutoff_dim=15)
    
    # Attempt to prepare a GKP |0⟩ state
    print("Optimizing GBS circuit for GKP |0⟩ state preparation...")
    result = gbs.prepare_gkp_state(target_state='zero', num_modes=3, max_photons=4)
    
    print(f"\nOptimized squeezing parameters: {result['squeezing']}")
    print(f"Postselection pattern: {result['postselection']}")
    print(f"Estimated fidelity: {result['fidelity']:.4f}")
    
    # Create optimized circuit
    circuit = gbs.create_circuit(result['squeezing'], result['interferometer'])
    
    # Calculate success probability
    full_pattern = list(result['postselection']) + [0]  # Last mode is output
    prob = gbs.calculate_probabilities(circuit, full_pattern)
    print(f"Success probability: {prob:.6e}")


def example_with_displacement():
    """Example with displaced squeezed states."""
    gbs = GaussianBosonSampling(num_modes=2, cutoff_dim=15)
    
    # Parameters
    squeezing_params = [0.8, 0.6]
    displacement_params = [(0.5, 0.3), (0.2, -0.4)]  # (real, imag) parts
    interferometer_params = [np.pi/4, 0]  # Simple 50:50 beamsplitter
    
    # Create circuit with displacements
    circuit = gbs.create_circuit(squeezing_params, interferometer_params, 
                               displacement_params)
    
    # Run sampling
    samples = gbs.run_sampling(circuit, num_samples=1000, backend='fock')
    
    print("Displaced squeezed state GBS")
    print("Average photon numbers per mode:", np.mean(samples, axis=0))
    
    gbs.visualize_samples(samples)


if __name__ == "__main__":
    print("=== Basic GBS Example ===")
    example_basic_gbs()
    
    print("\n=== GKP State Preparation Example ===")
    example_gkp_preparation()
    
    print("\n=== Displaced Squeezed States Example ===")
    example_with_displacement()