#!/usr/bin/env python3
"""idealisedgkp.py

Minimal GKP phase sensing with experimental homodyne detection circuit.

Protocol:
1. Prepare GKP |0⟩ state
2. Apply unknown phase R(φ)  
3. Full homodyne: Local Oscillator + Beamsplitter + Dual Photodetectors
4. Extract phase from measurement means

Simply run: python idealisedgkp.py
"""
from __future__ import annotations
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import GKP, Rgate, Coherent, BSgate, MeasureFock

sf.hbar = 1.0


def experimental_homodyne_measurement(probe_theta, probe_phi, phase_shift, measurement_angle, 
                                    num_shots=100, lo_amplitude=8.0):
    """
    Full experimental homodyne detection circuit:
    - Signal: GKP state + phase rotation
    - Local Oscillator: Strong coherent beam
    - 50:50 Beamsplitter interference
    - Dual photodetectors
    - Difference signal processing
    """
    measurements = []
    
    print(f"    Running homodyne circuit (LO amplitude: {lo_amplitude})")
    print(f"    Signal: GKP → R({phase_shift:.3f}) → BS")
    print(f"    LO: Coherent({lo_amplitude}) → R({measurement_angle:.3f}) → BS")
    
    for shot in range(num_shots):
        # Create 2-mode program: [0] = signal, [1] = local oscillator
        prog = sf.Program(2)
        with prog.context as q:
            # === SIGNAL PATH (Mode 0) ===
            eps = 0.01
            kwargs = dict(epsilon=eps, ampl_cutoff=1e-12, representation="real", shape="square")
            if probe_theta is None:
                GKP(**kwargs) | q[0]
            else:
                GKP(state=[probe_theta, probe_phi], **kwargs) | q[0]
            
            # Apply unknown phase shift to signal
            Rgate(phase_shift) | q[0]
            
            # === LOCAL OSCILLATOR PATH (Mode 1) ===
            Coherent(lo_amplitude) | q[1]
            Rgate(measurement_angle) | q[1]  # LO phase sets quadrature
            
            # === INTERFERENCE ===
            BSgate(np.pi/4, 0) | (q[0], q[1])  # 50:50 beamsplitter
            
            # === DUAL PHOTODETECTION ===
            MeasureFock() | q[0]  # Detector 1
            MeasureFock() | q[1]  # Detector 2
        
        # Run the experimental circuit
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 30})
        result = eng.run(prog)
        
        # Extract photon counts and compute homodyne signal
        n1, n2 = result.samples[0]
        homodyne_signal = n1 - n2  # Difference current
        measurements.append(homodyne_signal)
    
    return np.array(measurements)


def main():
    """Run single phase sensing test."""
    true_phase = 0.4  # The unknown phase we want to estimate
    
    print("=== EXPERIMENTAL GKP PHASE SENSING ===")
    print(f"True phase: {true_phase:.3f} rad ({true_phase*180/np.pi:.1f}°)")
    
    # Step 1: Measure q-quadrature (LO phase = 0)
    print("\nStep 1: q-quadrature measurement...")
    q_measurements = experimental_homodyne_measurement(
        None, None, true_phase, 0, num_shots=100
    )
    q_mean = np.mean(q_measurements)
    print(f"  q-quadrature mean: {q_mean:.3f}")
    
    # Step 2: Measure p-quadrature (LO phase = π/2)  
    print("\nStep 2: p-quadrature measurement...")
    p_measurements = experimental_homodyne_measurement(
        None, None, true_phase, np.pi/2, num_shots=100
    )
    p_mean = np.mean(p_measurements)
    print(f"  p-quadrature mean: {p_mean:.3f}")
    
    # Step 3: Extract phase from measurements
    print("\nStep 3: Phase estimation...")
    estimated_phase = np.arctan2(p_mean, q_mean)
    
    # Handle negative angles
    if estimated_phase < 0:
        estimated_phase += 2*np.pi
    
    # Calculate error
    estimation_error = abs(estimated_phase - true_phase)
    if estimation_error > np.pi:
        estimation_error = 2*np.pi - estimation_error
    
    print(f"  Estimated phase: {estimated_phase:.3f} rad ({estimated_phase*180/np.pi:.1f}°)")
    print(f"  Estimation error: {estimation_error:.3f} rad ({estimation_error*180/np.pi:.1f}°)")
    print(f"  Relative error: {estimation_error/true_phase*100:.1f}%")
    
    # Show how photon counts → phase
    print(f"  ⟨q⟩ = {q_mean:.3f} ∝ cos({true_phase:.3f}) = {np.cos(true_phase):.3f}")
    print(f"  ⟨p⟩ = {p_mean:.3f} ∝ sin({true_phase:.3f}) = {np.sin(true_phase):.3f}")
    print(f"  φ = atan2(⟨p⟩, ⟨q⟩) = atan2({p_mean:.3f}, {q_mean:.3f}) = {estimated_phase:.3f}")


if __name__ == "__main__":
    main()