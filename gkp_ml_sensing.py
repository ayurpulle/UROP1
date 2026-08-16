#!/usr/bin/env python3
"""gkp_ml_sensing.py

Deep Learning GKP Phase Estimation Protocol

Compares classical trigonometric estimation (atan2) with neural network 
estimation that learns from measurement statistics.

Features:
- Rich feature extraction from homodyne measurement data
- Neural network trained on simulated GKP measurements
- Direct comparison of classical vs ML approaches
- Handles experimental noise and imperfections

Usage: python gkp_ml_sensing.py
Dependencies: torch, sklearn, numpy, strawberryfields
"""
from __future__ import annotations

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import GKP, Rgate, Coherent, BSgate, MeasureFock

# Try to import ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
    print("✓ ML libraries loaded successfully")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"✗ ML libraries not available: {e}")
    print("Install with: pip install torch scikit-learn")

sf.hbar = 1.0


def experimental_homodyne_measurement(probe_theta, probe_phi, phase_shift, measurement_angle, 
                                    num_shots=100, lo_amplitude=8.0, verbose=False):
    """
    Full experimental homodyne detection circuit.
    
    Args:
        probe_theta: GKP state parameter (None for |0⟩)
        probe_phi: GKP state parameter  
        phase_shift: Unknown phase to apply
        measurement_angle: LO phase (0 for q, π/2 for p)
        num_shots: Number of measurement shots
        lo_amplitude: Local oscillator strength
        verbose: Print debug info
        
    Returns:
        Array of homodyne measurements (n1 - n2)
    """
    measurements = []
    
    if verbose:
        print(f"    Running homodyne (LO: {lo_amplitude}, angle: {measurement_angle:.3f})")
    
    for shot in range(num_shots):
        prog = sf.Program(2)
        with prog.context as q:
            # Signal path: GKP + phase rotation
            eps = 0.01
            kwargs = dict(epsilon=eps, ampl_cutoff=1e-12, representation="real", shape="square")
            if probe_theta is None:
                GKP(**kwargs) | q[0]
            else:
                GKP(state=[probe_theta, probe_phi], **kwargs) | q[0]
            
            Rgate(phase_shift) | q[0]  # Apply unknown phase
            
            # Local oscillator path
            Coherent(lo_amplitude) | q[1]
            Rgate(measurement_angle) | q[1]  # Set quadrature
            
            # Interference and detection
            BSgate(np.pi/4, 0) | (q[0], q[1])  # 50:50 beamsplitter
            MeasureFock() | q[0]  # Detector 1
            MeasureFock() | q[1]  # Detector 2
        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 30})
        result = eng.run(prog)
        n1, n2 = result.samples[0]
        measurements.append(n1 - n2)  # Homodyne signal
    
    return np.array(measurements)


def extract_measurement_features(q_measurements, p_measurements):
    """
    Extract rich statistical features from raw measurement data for ML.
    
    Args:
        q_measurements: Array of q-quadrature measurements
        p_measurements: Array of p-quadrature measurements
        
    Returns:
        Feature vector for neural network input
    """
    features = []
    
    # Basic statistics (8 features)
    features.extend([
        np.mean(q_measurements), np.std(q_measurements),
        np.mean(p_measurements), np.std(p_measurements),
        np.min(q_measurements), np.max(q_measurements),
        np.min(p_measurements), np.max(p_measurements)
    ])
    
    # Higher-order moments and percentiles (6 features)
    features.extend([
        np.var(q_measurements), np.var(p_measurements),
        float(np.percentile(q_measurements, 25)), float(np.percentile(q_measurements, 75)),
        float(np.percentile(p_measurements, 25)), float(np.percentile(p_measurements, 75))
    ])
    
    # Cross-correlation between q and p (1 feature)
    if len(q_measurements) > 1:
        corr_matrix = np.corrcoef(q_measurements, p_measurements)
        features.append(corr_matrix[0,1] if not np.isnan(corr_matrix[0,1]) else 0.0)
    else:
        features.append(0.0)
    
    # Histogram features (10 features: 5 bins each for q and p)
    q_hist, _ = np.histogram(q_measurements, bins=5, density=True)
    p_hist, _ = np.histogram(p_measurements, bins=5, density=True)
    features.extend(q_hist.tolist())
    features.extend(p_hist.tolist())
    
    # Skewness and kurtosis (4 features)
    from scipy.stats import skew, kurtosis
    features.extend([
        skew(q_measurements), kurtosis(q_measurements),
        skew(p_measurements), kurtosis(p_measurements)
    ])
    
    return np.array(features, dtype=np.float32)


class PhaseEstimatorNet(nn.Module):
    """
    Neural network for phase estimation from homodyne measurement features.
    
    Architecture:
        Input (29 features) → 128 → 64 → 32 → 2 outputs [cos(φ), sin(φ)]
    
    Design choices:
        - Output cos/sin instead of phase directly (avoids periodicity issues)
        - Normalization layer enforces unit circle constraint
        - Dropout for regularization
    """
    
    def __init__(self, input_dim=29):
        super(PhaseEstimatorNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 2)  # [cos(φ), sin(φ)]
        )
        
    def forward(self, x):
        output = self.network(x)
        # Normalize to unit circle: cos² + sin² = 1
        norm = torch.sqrt(output[:, 0]**2 + output[:, 1]**2 + 1e-8)
        return output / norm.unsqueeze(1)


def generate_training_data(num_samples=1000, noise_level=0.1, verbose=True):
    """
    Generate training data for the neural network.
    
    Args:
        num_samples: Number of training examples
        noise_level: Additional measurement noise
        verbose: Print progress
        
    Returns:
        (X_features, y_phases): Training data and labels
    """
    if verbose:
        print(f"Generating {num_samples} training samples...")
    
    X_features = []
    y_phases = []
    
    for i in range(num_samples):
        if verbose and i % 100 == 0:
            print(f"  Sample {i}/{num_samples}")
            
        # Random phase to estimate (uniform over [0, 2π])
        true_phase = np.random.uniform(0, 2*np.pi)
        
        # Generate measurement data
        q_meas = experimental_homodyne_measurement(
            None, None, true_phase, 0, num_shots=80, verbose=False
        )
        p_meas = experimental_homodyne_measurement(
            None, None, true_phase, np.pi/2, num_shots=80, verbose=False
        )
        
        # Add experimental noise
        q_meas += np.random.normal(0, noise_level, len(q_meas))
        p_meas += np.random.normal(0, noise_level, len(p_meas))
        
        # Extract features
        try:
            features = extract_measurement_features(q_meas, p_meas)
            X_features.append(features)
            y_phases.append(true_phase)
        except Exception as e:
            if verbose:
                print(f"    Warning: Failed to extract features for sample {i}: {e}")
            continue
    
    return np.array(X_features), np.array(y_phases)


def train_deep_learning_estimator(num_samples=800, epochs=300, verbose=True):
    """
    Train the neural network phase estimator.
    
    Args:
        num_samples: Number of training samples
        epochs: Training epochs
        verbose: Print training progress
        
    Returns:
        (model, scaler): Trained model and feature scaler
    """
    if not ML_AVAILABLE:
        print("ML libraries not available - cannot train")
        return None, None
        
    print("=== TRAINING DEEP LEARNING PHASE ESTIMATOR ===")
    
    # Generate training data
    X, y = generate_training_data(num_samples=num_samples, noise_level=0.08, verbose=verbose)
    
    if len(X) == 0:
        print("No training data generated!")
        return None, None
    
    print(f"Generated {len(X)} training samples with {X.shape[1]} features each")
    
    # Convert phases to cos/sin representation for better learning
    y_cossin = np.column_stack([np.cos(y), np.sin(y)])
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_cossin)
    
    # Create model and optimizer
    model = PhaseEstimatorNet(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    
    print(f"Training neural network for {epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Training step
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        # Track best model
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}, Loss: {loss.item():.6f}, Best: {best_loss:.6f}")
    
    print(f"Training complete! Final loss: {loss.item():.6f}")
    return model, scaler


def classical_estimation(q_measurements, p_measurements):
    """Classical phase estimation using trigonometry."""
    q_mean = np.mean(q_measurements)
    p_mean = np.mean(p_measurements)
    
    estimated_phase = np.arctan2(p_mean, q_mean)
    if estimated_phase < 0:
        estimated_phase += 2*np.pi
        
    return estimated_phase


def deep_learning_estimation(q_measurements, p_measurements, model, scaler):
    """Deep learning phase estimation using neural network."""
    if not ML_AVAILABLE or model is None or scaler is None:
        return None
        
    try:
        # Extract features
        features = extract_measurement_features(q_measurements, p_measurements)
        
        # Normalize and predict
        features_scaled = scaler.transform(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features_scaled)
        
        with torch.no_grad():
            prediction = model(features_tensor)
            cos_phi, sin_phi = prediction[0].numpy()
            
        # Convert back to phase
        estimated_phase = np.arctan2(sin_phi, cos_phi)
        if estimated_phase < 0:
            estimated_phase += 2*np.pi
            
        return estimated_phase
        
    except Exception as e:
        print(f"ML estimation failed: {e}")
        return None


def compare_estimation_methods():
    """Compare classical vs deep learning estimation on test phases."""
    print("=== COMPARING CLASSICAL VS DEEP LEARNING ESTIMATION ===")
    
    # Train deep learning model
    model, scaler = train_deep_learning_estimator(num_samples=600, epochs=250)
    
    if model is None:
        print("Training failed - running classical only")
        return
    
    # Test phases
    test_phases = [0.2, 0.8, 1.2, 1.8, 2.5, 3.8, 4.5, 5.5]
    
    print("\nTesting on unseen phases:")
    print("Phase(rad) | Phase(°) | Classical Error | ML Error    | Better Method")
    print("-" * 70)
    
    classical_errors = []
    ml_errors = []
    ml_wins = 0
    
    for true_phase in test_phases:
        # Get fresh measurement data
        q_meas = experimental_homodyne_measurement(None, None, true_phase, 0, num_shots=120)
        p_meas = experimental_homodyne_measurement(None, None, true_phase, np.pi/2, num_shots=120)
        
        # Classical estimation
        classical_est = classical_estimation(q_meas, p_meas)
        classical_error = min(abs(classical_est - true_phase), 
                            2*np.pi - abs(classical_est - true_phase))
        classical_errors.append(classical_error)
        
        # Deep learning estimation
        ml_est = deep_learning_estimation(q_meas, p_meas, model, scaler)
        if ml_est is not None:
            ml_error = min(abs(ml_est - true_phase), 
                         2*np.pi - abs(ml_est - true_phase))
            ml_errors.append(ml_error)
            
            better = "ML" if ml_error < classical_error else "Classical"
            if ml_error < classical_error:
                ml_wins += 1
                
            print(f"{true_phase:8.1f} | {true_phase*180/np.pi:6.1f}° | {classical_error:13.4f} | {ml_error:8.4f} | {better}")
        else:
            ml_errors.append(float('inf'))
            print(f"{true_phase:8.1f} | {true_phase*180/np.pi:6.1f}° | {classical_error:13.4f} | Failed   | Classical")
    
    # Summary statistics
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Classical method:")
    print(f"  Mean error: {np.mean(classical_errors):.4f} rad ({np.mean(classical_errors)*180/np.pi:.2f}°)")
    print(f"  Std error:  {np.std(classical_errors):.4f} rad ({np.std(classical_errors)*180/np.pi:.2f}°)")
    
    valid_ml_errors = [e for e in ml_errors if e != float('inf')]
    if valid_ml_errors:
        print(f"Deep Learning method:")
        print(f"  Mean error: {np.mean(valid_ml_errors):.4f} rad ({np.mean(valid_ml_errors)*180/np.pi:.2f}°)")
        print(f"  Std error:  {np.std(valid_ml_errors):.4f} rad ({np.std(valid_ml_errors)*180/np.pi:.2f}°)")
        
        improvement = (np.mean(classical_errors) - np.mean(valid_ml_errors)) / np.mean(classical_errors) * 100
        print(f"Performance change: {improvement:+.1f}%")
        print(f"ML wins: {ml_wins}/{len(test_phases)} cases ({ml_wins/len(test_phases)*100:.1f}%)")


def simple_demo():
    """Simple demonstration with single phase."""
    true_phase = 0.6
    
    print("=== SIMPLE GKP PHASE SENSING DEMO ===")
    print(f"True phase: {true_phase:.3f} rad ({true_phase*180/np.pi:.1f}°)")
    
    # Get measurements
    print("\nStep 1: Collecting measurement data...")
    q_measurements = experimental_homodyne_measurement(None, None, true_phase, 0, num_shots=100, verbose=True)
    p_measurements = experimental_homodyne_measurement(None, None, true_phase, np.pi/2, num_shots=100, verbose=True)
    
    # Classical estimation
    print("\nStep 2: Classical estimation...")
    classical_est = classical_estimation(q_measurements, p_measurements)
    classical_error = min(abs(classical_est - true_phase), 2*np.pi - abs(classical_est - true_phase))
    
    print(f"  Classical estimate: {classical_est:.3f} rad ({classical_est*180/np.pi:.1f}°)")
    print(f"  Classical error: {classical_error:.3f} rad ({classical_error*180/np.pi:.1f}°)")
    
    # Show feature extraction for ML
    if ML_AVAILABLE:
        print("\nStep 3: Feature extraction for ML...")
        try:
            features = extract_measurement_features(q_measurements, p_measurements)
            print(f"  Extracted {len(features)} features from measurement data")
            print(f"  Features include: means, stds, correlations, histograms, moments...")
            print("  (Would need trained model for ML estimation)")
        except Exception as e:
            print(f"  Feature extraction failed: {e}")


def main():
    """Main function with menu options."""
    print("🧠 GKP Phase Sensing: Classical vs Deep Learning")
    print("=" * 50)
    
    if not ML_AVAILABLE:
        print("Note: ML libraries not available. Only classical demo will work.")
        print()
    
    print("Choose demonstration:")
    print("1. Simple demo (classical + feature extraction)")
    print("2. Train ML model and compare methods")
    print("3. Feature extraction details")
    
    try:
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            simple_demo()
        elif choice == "2":
            if ML_AVAILABLE:
                compare_estimation_methods()
            else:
                print("ML libraries required for this option")
                simple_demo()
        elif choice == "3":
            # Show feature extraction example
            print("=== FEATURE EXTRACTION EXAMPLE ===")
            dummy_q = np.random.randn(50)
            dummy_p = np.random.randn(50)
            if ML_AVAILABLE:
                features = extract_measurement_features(dummy_q, dummy_p)
                print(f"Feature vector length: {len(features)}")
                print("Feature types:")
                print("  - Basic stats (8): means, stds, mins, maxs")
                print("  - Moments (6): variances, percentiles")
                print("  - Correlation (1): q-p correlation")
                print("  - Histograms (10): 5 bins each for q,p")
                print("  - Shape (4): skewness, kurtosis")
                print(f"Total: {len(features)} features")
            else:
                print("ML libraries required for feature extraction")
        else:
            print("Invalid choice, running simple demo...")
            simple_demo()
            
    except (KeyboardInterrupt, EOFError):
        print("\nRunning simple demo...")
        simple_demo()


if __name__ == "__main__":
    main() 