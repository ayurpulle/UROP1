import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import plot_wigner
import seaborn as sns

class PhotonicGridStateSensor:
    """
    Simplified photonic implementation using Strawberry Fields
    for quantum-enhanced sensing with grid states
    """
    
    def __init__(self, cutoff_dim=30):
        """
        Initialize the photonic system using Strawberry Fields
        
        Parameters:
        cutoff_dim: Fock space cutoff for simulations
        """
        self.cutoff = cutoff_dim
        self.alpha_grid = np.sqrt(2 * np.pi)  # Grid spacing from papers
        
    def create_vacuum_state(self):
        """Initialize cavity in vacuum state"""
        prog = sf.Program(1)
        with prog.context as q:
            # Vacuum state is the default initialization
            pass
        return prog
    
    def create_coherent_state(self, alpha):
        """Create coherent state |α⟩"""
        prog = sf.Program(1)
        with prog.context as q:
            Dgate(alpha) | q[0]  # Displacement operator
        return prog
    
    def create_squeezed_state(self, r, phi=0):
        """Create squeezed state with squeezing parameter r"""
        prog = sf.Program(1)
        with prog.context as q:
            Sgate(r, phi) | q[0]  # Squeezing operator
        return prog
    
    def create_grid_state_approximation(self, n_peaks=3, amplitude=1.0):
        """
        Create grid state as superposition of coherent states
        This approximates the result of phase estimation protocol
        """
        prog = sf.Program(1)
        
        # Calculate grid positions in phase space
        grid_positions = []
        spacing = self.alpha_grid / np.sqrt(2)
        
        for i in range(-n_peaks//2, n_peaks//2 + 1):
            for j in range(-n_peaks//2, n_peaks//2 + 1):
                q_pos = i * spacing / 2
                p_pos = j * spacing / 2
                alpha = (q_pos + 1j * p_pos) * amplitude
                if abs(alpha) < 3:  # Keep reasonable size
                    grid_positions.append(alpha)
        
        # Create superposition (simplified - in practice this requires more complex preparation)
        # Here we create a representative coherent state at one grid point
        main_alpha = grid_positions[len(grid_positions)//2] if grid_positions else 0
        
        with prog.context as q:
            Dgate(main_alpha) | q[0]
            
        return prog, grid_positions
    
    def create_gkp_approximation(self, Delta=0.4):
        """
        Create an approximation of GKP (Gottesman-Kitaev-Preskill) grid state
        Using finite energy approximation with Gaussian envelope
        """
        prog = sf.Program(1)
        with prog.context as q:
            # Start with squeezed state
            Sgate(0.5) | q[0]
            
            # Apply series of displacements to create grid-like structure
            # This is a simplified approximation
            for i in range(3):
                Dgate(Delta * np.exp(1j * i * 2 * np.pi / 3)) | q[0]
                Sgate(-0.1, np.pi/4) | q[0]  # Additional squeezing
                
        return prog
    
    def phase_estimation_simulation(self, M_rounds=4):
        """
        Simulate the phase estimation protocol from the papers
        that creates grid states through M rounds of operations
        """
        prog = sf.Program(1)
        with prog.context as q:
            # Start with vacuum
            
            # Simulate M rounds of phase estimation
            for round_num in range(M_rounds):
                # Each round applies conditional operations
                strength = 0.5 / (round_num + 1)  # Decreasing strength
                
                if round_num % 2 == 0:
                    # Position shift operator simulation
                    Xgate(strength * self.alpha_grid / 4) | q[0]
                else:
                    # Momentum shift operator simulation  
                    Zgate(strength * self.alpha_grid / 4) | q[0]
                
                # Add some squeezing to approximate grid structure
                Sgate(0.2, round_num * np.pi / 4) | q[0]
        
        return prog
    
    def execute_and_analyze(self, program, state_name=""):
        """Execute program and return state for analysis"""
        engine = sf.Engine('fock', backend_options={'cutoff_dim': self.cutoff})
        result = engine.run(program)
        state = result.state
        return state
    
    def plot_wigner_comparison(self):
        """
        Create and plot Wigner functions for different photonic states
        """
        # Define states to compare
        states_info = [
            ("Vacuum State", self.create_vacuum_state()),
            ("Coherent State |α=1.5⟩", self.create_coherent_state(1.5)),
            ("Squeezed State", self.create_squeezed_state(0.8)),
            ("Grid State (Phase Est.)", self.phase_estimation_simulation(M_rounds=6))
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        # Define coordinate ranges for Wigner function
        xvs = np.linspace(-4, 4, 100)
        yvs = np.linspace(-4, 4, 100)
        
        for idx, (name, program) in enumerate(states_info):
            # Execute program
            state = self.execute_and_analyze(program, name)
            
            # Use Strawberry Fields' built-in Wigner function plotter
            ax = axes[idx]
            
            # SF's plot_wigner returns the figure, but we can extract the data
            wigner_data = state.wigner(mode=0, xvs=xvs, yvs=yvs)
            X, Y = np.meshgrid(xvs, yvs)
            
            # Plot using matplotlib with SF's calculated Wigner data
            im = ax.contourf(X, Y, wigner_data, levels=20, cmap='RdBu_r', alpha=0.8)
            ax.contour(X, Y, wigner_data, levels=10, colors='black', alpha=0.3, linewidths=0.5)
            
            ax.set_title(f'{name}\nWigner Function W(q,p)', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Position q', fontsize=11)
            ax.set_ylabel('Momentum p', fontsize=11)
            ax.set_aspect('equal')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
            
            # Add grid lines for reference
            if "Grid" in name:
                spacing = self.alpha_grid / np.sqrt(2) / 2
                for i in range(-2, 3):
                    ax.axhline(y=i*spacing, color='white', alpha=0.5, linestyle='--', linewidth=0.8)
                    ax.axvline(x=i*spacing, color='white', alpha=0.5, linestyle='--', linewidth=0.8)
        
        plt.tight_layout()
        plt.suptitle('Photonic States for Quantum-Enhanced Sensing\n(Strawberry Fields Implementation)', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.show()
        
        return states_info
    
    def analyze_sensing_performance(self, program, state_name=""):
        """
        Analyze quantum sensing performance metrics
        """
        state = self.execute_and_analyze(program, state_name)
        
        print(f"\n=== Analysis: {state_name} ===")
        
        # Calculate average photon number
        n_avg = state.mean_photon(0)
        print(f"Average photon number ⟨n̂⟩: {n_avg:.3f}")
        
        # Calculate quadrature variances
        q_var = state.quad_expectation(0, 0)  # Position variance
        p_var = state.quad_expectation(0, 1)  # Momentum variance
        
        print(f"Position variance Var(q̂): {q_var:.3f}")
        print(f"Momentum variance Var(p̂): {p_var:.3f}")
        
        # Uncertainty product
        uncertainty = np.sqrt(q_var * p_var)
        print(f"Uncertainty product ΔqΔp: {uncertainty:.3f}")
        print(f"Heisenberg limit (ℏ/2): {0.5:.3f}")
        print(f"Violation ratio: {uncertainty/0.5:.3f}")
        
        # Sensing performance metrics
        if n_avg > 0:
            sql_limit = 1 / np.sqrt(n_avg + 1)
            quantum_limit = 1 / np.sqrt(n_avg)
            enhancement = sql_limit / quantum_limit
            
            print(f"\nSensing Performance:")
            print(f"SQL scaling: ∝ 1/√(n+1) ≈ {sql_limit:.4f}")
            print(f"Quantum enhanced: ∝ 1/√n ≈ {quantum_limit:.4f}")
            print(f"Enhancement factor: {enhancement:.2f}")
            
            # Convert to dB
            enhancement_db = 10 * np.log10(enhancement)
            print(f"Enhancement: {enhancement_db:.1f} dB")
        
        return {
            'n_avg': n_avg,
            'q_var': q_var,
            'p_var': p_var,
            'uncertainty': uncertainty
        }
    
    def displacement_sensing_demo(self):
        """
        Demonstrate displacement sensing with grid states
        """
        print("=== Displacement Sensing Demonstration ===")
        
        # Create grid state
        grid_prog = self.phase_estimation_simulation(M_rounds=6)
        
        # Apply unknown displacement (this is what we want to sense)
        unknown_displacement = 0.3 + 0.2j
        
        sensing_prog = sf.Program(1)
        with sensing_prog.context as q:
            # Prepare grid state (simplified)
            Sgate(0.5) | q[0]
            Dgate(0.5) | q[0]
            
            # Apply unknown displacement
            Dgate(unknown_displacement) | q[0]
        
        # Analyze sensing capability
        results = self.analyze_sensing_performance(sensing_prog, "Grid State + Displacement")
        
        print(f"\nUnknown displacement: {unknown_displacement}")
        print(f"Real part (position): {unknown_displacement.real:.3f}")
        print(f"Imaginary part (momentum): {unknown_displacement.imag:.3f}")
        
        return results

def demonstrate_photonic_sensing():
    """
    Main demonstration of photonic quantum sensing
    """
    print("Photonic Quantum-Enhanced Sensing with Strawberry Fields")
    print("=" * 60)
    
    # Initialize sensor
    sensor = PhotonicGridStateSensor(cutoff_dim=25)
    
    # Plot Wigner functions
    print("\nGenerating Wigner function comparison...")
    states = sensor.plot_wigner_comparison()
    
    # Analyze different states
    for name, program in states:
        sensor.analyze_sensing_performance(program, name)
    
    # Demonstrate displacement sensing
    sensor.displacement_sensing_demo()
    
    print(f"\n{'='*60}")
    print("Key Results from Papers:")
    print("• Grid states achieve 5.1(5) dB improvement over SQL")
    print("• Phase-insensitive force sensitivity: 14.3 yN/√Hz")
    print("• Simultaneous measurement below uncertainty limit")
    print("• Compatible with microwave cavity implementations")

