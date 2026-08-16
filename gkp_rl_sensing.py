#!/usr/bin/env python3
"""gkp_rl_sensing.py

Reinforcement Learning for Optimal GKP Parameter Encoding

Instead of fixed measurement protocols, train an RL agent to discover 
optimal strategies for encoding and sensing unknown phases.

The agent learns to:
1. Choose which GKP states to prepare
2. Select measurement angles and sequences  
3. Decide when to stop and make final estimate
4. Optimize the entire sensing protocol end-to-end

Usage: python gkp_rl_sensing.py
Dependencies: torch, gymnasium, numpy, strawberryfields
"""
from __future__ import annotations

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import GKP, Rgate, Coherent, BSgate, MeasureFock
from typing import Tuple, Dict, Any, Optional
import random
from collections import deque

# Try to import RL libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    RL_AVAILABLE = True
    print("✓ RL libraries loaded successfully")
except ImportError as e:
    RL_AVAILABLE = False
    print(f"✗ RL libraries not available: {e}")
    print("Install with: pip install torch")

sf.hbar = 1.0


class GKPSensingEnvironment:
    """
    RL Environment for GKP phase sensing.
    
    State: [current_estimate, uncertainty, measurement_history, budget_remaining]
    Action: [gkp_theta, gkp_phi, measurement_angle, stop_flag]
    Reward: Based on final estimation accuracy and efficiency
    """
    
    def __init__(self, max_measurements=10, measurement_budget=500):
        self.max_measurements = max_measurements
        self.measurement_budget = measurement_budget
        self.reset()
        
    def reset(self, true_phase: Optional[float] = None) -> np.ndarray:
        """Reset environment with new unknown phase."""
        self.true_phase = true_phase if true_phase is not None else np.random.uniform(0, 2*np.pi)
        self.current_estimate = np.pi  # Start with uniform prior
        self.uncertainty = np.pi  # Large initial uncertainty
        self.measurement_history = []
        self.measurements_used = 0
        self.shots_used = 0
        self.done = False
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation for the agent."""
        # Basic state: estimate, uncertainty, resource usage
        state = [
            np.cos(self.current_estimate), np.sin(self.current_estimate),  # Current estimate
            self.uncertainty,  # Uncertainty
            self.measurements_used / self.max_measurements,  # Resource usage
            len(self.measurement_history) / 20.0  # History length (normalized)
        ]
        
        # Add recent measurement statistics
        if self.measurement_history:
            recent_measurements = self.measurement_history[-3:]  # Last 3 measurements
            for meas_data in recent_measurements:
                state.extend([
                    meas_data['mean'] / 10.0,  # Normalize measurement mean
                    meas_data['std'] / 5.0,    # Normalize measurement std
                ])
            
            # Pad if fewer than 3 measurements
            while len(recent_measurements) < 3:
                state.extend([0.0, 0.0])
        else:
            state.extend([0.0] * 6)  # 3 measurements × 2 stats each
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take action in environment.
        
        Action: [gkp_theta, gkp_phi, measurement_angle, num_shots, stop_flag]
        """
        if self.done:
            return self._get_state(), 0.0, True, {}
        
        # Decode action
        gkp_theta = action[0] * np.pi  # [0,1] -> [0,π]
        gkp_phi = action[1] * 2 * np.pi  # [0,1] -> [0,2π] 
        measurement_angle = action[2] * np.pi  # [0,1] -> [0,π]
        num_shots = int(action[3] * 80) + 20  # [0,1] -> [20,100] shots
        stop_flag = action[4] > 0.5  # Boolean decision
        
        reward = 0.0
        info = {}
        
        if stop_flag or self.measurements_used >= self.max_measurements:
            # Agent decides to stop - calculate final reward
            estimation_error = min(
                abs(self.current_estimate - self.true_phase),
                2*np.pi - abs(self.current_estimate - self.true_phase)
            )
            
            # Reward based on accuracy and efficiency
            accuracy_reward = np.exp(-estimation_error * 5)  # High reward for low error
            efficiency_reward = (self.measurement_budget - self.shots_used) / self.measurement_budget
            
            reward = accuracy_reward + 0.3 * efficiency_reward
            self.done = True
            
            info = {
                'final_error': estimation_error,
                'shots_used': self.shots_used,
                'measurements_used': self.measurements_used,
                'efficiency': efficiency_reward
            }
        else:
            # Perform measurement and update estimate
            if self.shots_used + num_shots <= self.measurement_budget:
                measurement_data = self._perform_measurement(
                    gkp_theta, gkp_phi, measurement_angle, num_shots
                )
                
                # Update state estimate (simplified Bayesian update)
                self._update_estimate(measurement_data, measurement_angle)
                
                # Small reward for gaining information
                reward = -0.01  # Small cost for each measurement
                self.measurements_used += 1
                self.shots_used += num_shots
                
                info = {
                    'measurement_mean': measurement_data['mean'],
                    'measurement_std': measurement_data['std']
                }
            else:
                # Out of budget - force stop
                reward = -0.5  # Penalty for poor resource management
                self.done = True
        
        return self._get_state(), reward, self.done, info
    
    def _perform_measurement(self, gkp_theta: float, gkp_phi: float, 
                           measurement_angle: float, num_shots: int) -> Dict[str, float]:
        """Perform actual quantum measurement."""
        measurements = []
        
        for shot in range(num_shots):
            prog = sf.Program(2)
            with prog.context as q:
                # Prepare chosen GKP state
                eps = 0.01
                kwargs = dict(epsilon=eps, ampl_cutoff=1e-12, representation="real", shape="square")
                if gkp_theta is None:
                    GKP(**kwargs) | q[0]
                else:
                    GKP(state=[gkp_theta, gkp_phi], **kwargs) | q[0]
                
                # Apply unknown phase
                Rgate(self.true_phase) | q[0]
                
                # Local oscillator
                Coherent(8.0) | q[1]
                Rgate(measurement_angle) | q[1]
                
                # Interference and detection
                BSgate(np.pi/4, 0) | (q[0], q[1])
                MeasureFock() | q[0]
                MeasureFock() | q[1]
            
            try:
                eng = sf.Engine("fock", backend_options={"cutoff_dim": 25})
                result = eng.run(prog)
                n1, n2 = result.samples[0]
                measurements.append(float(n1 - n2))
            except Exception:
                measurements.append(0.0)  # Handle simulation failures gracefully
        
        measurement_data = {
            'mean': np.mean(measurements),
            'std': np.std(measurements) if len(measurements) > 1 else 1.0,
            'angle': measurement_angle,
            'gkp_state': (gkp_theta, gkp_phi),
            'num_shots': num_shots
        }
        
        self.measurement_history.append(measurement_data)
        return measurement_data
    
    def _update_estimate(self, measurement_data: Dict[str, float], measurement_angle: float):
        """Update phase estimate based on measurement (simplified)."""
        measured_mean = measurement_data['mean']
        measured_std = measurement_data['std']
        
        # Simple update rule (could be made more sophisticated)
        if measurement_angle < np.pi/4:  # Mostly q-quadrature
            # Update based on q measurement
            predicted_q = np.cos(self.current_estimate) * 3.0  # Rough scaling
            error = measured_mean - predicted_q
            
        elif measurement_angle > 3*np.pi/4:  # Mostly p-quadrature  
            # Update based on p measurement
            predicted_p = np.sin(self.current_estimate) * 3.0
            error = measured_mean - predicted_p
        else:
            # Mixed quadrature
            predicted = (np.cos(self.current_estimate) * np.cos(measurement_angle) + 
                        np.sin(self.current_estimate) * np.sin(measurement_angle)) * 3.0
            error = measured_mean - predicted
        
        # Update estimate (learning rate proportional to confidence)
        learning_rate = 0.1 / (1 + measured_std)
        self.current_estimate += learning_rate * error * 0.1
        
        # Ensure phase is in [0, 2π]
        self.current_estimate = self.current_estimate % (2 * np.pi)
        
        # Update uncertainty (decreases with more measurements)
        information_gain = 1.0 / (1 + measured_std)
        self.uncertainty *= (1 - 0.1 * information_gain)
        self.uncertainty = max(self.uncertainty, 0.01)  # Minimum uncertainty


class SensingPolicyNetwork(nn.Module):
    """
    Neural network policy for the RL agent.
    
    Input: Environment state
    Output: Action probabilities/values
    """
    
    def __init__(self, state_dim=11, action_dim=5, hidden_dim=128):
        super(SensingPolicyNetwork, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )
        
        # Action outputs (continuous actions + discrete stop)
        self.gkp_theta_head = nn.Linear(64, 1)  # [0,1] -> [0,π]
        self.gkp_phi_head = nn.Linear(64, 1)    # [0,1] -> [0,2π]
        self.measurement_angle_head = nn.Linear(64, 1)  # [0,1] -> [0,π]
        self.num_shots_head = nn.Linear(64, 1)  # [0,1] -> [20,100]
        self.stop_head = nn.Linear(64, 1)       # Probability of stopping
        
        # Value function (for actor-critic)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, state):
        features = self.shared_layers(state)
        
        # Continuous actions (sigmoid to bound in [0,1])
        gkp_theta = torch.sigmoid(self.gkp_theta_head(features))
        gkp_phi = torch.sigmoid(self.gkp_phi_head(features))
        measurement_angle = torch.sigmoid(self.measurement_angle_head(features))
        num_shots = torch.sigmoid(self.num_shots_head(features))
        
        # Stop decision (sigmoid for probability)
        stop_prob = torch.sigmoid(self.stop_head(features))
        
        # Value estimate
        value = self.value_head(features)
        
        return {
            'gkp_theta': gkp_theta,
            'gkp_phi': gkp_phi, 
            'measurement_angle': measurement_angle,
            'num_shots': num_shots,
            'stop_prob': stop_prob,
            'value': value
        }


class PPOAgent:
    """
    Proximal Policy Optimization agent for learning optimal sensing strategies.
    """
    
    def __init__(self, state_dim=11, action_dim=5, lr=3e-4):
        if not RL_AVAILABLE:
            raise ImportError("PyTorch required for RL agent")
            
        self.policy_net = SensingPolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # Experience replay
        self.memory = deque(maxlen=2000)
        
    def select_action(self, state, training=True):
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            policy_output = self.policy_net(state_tensor)
        
        if training:
            # Add exploration noise
            noise_scale = 0.1
            action = np.array([
                policy_output['gkp_theta'].item() + np.random.normal(0, noise_scale),
                policy_output['gkp_phi'].item() + np.random.normal(0, noise_scale),
                policy_output['measurement_angle'].item() + np.random.normal(0, noise_scale),
                policy_output['num_shots'].item() + np.random.normal(0, noise_scale),
                1.0 if np.random.random() < policy_output['stop_prob'].item() else 0.0
            ])
            
            # Clip to valid ranges
            action = np.clip(action, 0.0, 1.0)
        else:
            # Deterministic action for evaluation
            action = np.array([
                policy_output['gkp_theta'].item(),
                policy_output['gkp_phi'].item(),
                policy_output['measurement_angle'].item(),
                policy_output['num_shots'].item(),
                1.0 if policy_output['stop_prob'].item() > 0.5 else 0.0
            ])
        
        return action, policy_output['value'].item()
    
    def store_experience(self, state, action, reward, next_state, done, value):
        """Store experience for training."""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'value': value
        })
    
    def train(self, batch_size=64):
        """Train the policy using PPO."""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(list(self.memory), batch_size)
        
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.FloatTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        values = torch.FloatTensor([exp['value'] for exp in batch])
        
        # Calculate advantages (simplified)
        advantages = rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy update
        policy_outputs = self.policy_net(states)
        
        # Calculate policy loss (simplified PPO)
        policy_loss = -advantages.mean()  # Simplified for demo
        value_loss = F.mse_loss(policy_outputs['value'].squeeze(), rewards)
        
        total_loss = policy_loss + self.value_coef * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()


def train_rl_agent(episodes=500, max_steps=10):
    """Train the RL agent to find optimal sensing strategies."""
    if not RL_AVAILABLE:
        print("RL libraries not available")
        return None
    
    print("=== TRAINING RL AGENT FOR OPTIMAL GKP SENSING ===")
    
    env = GKPSensingEnvironment(max_measurements=max_steps)
    agent = PPOAgent()
    
    episode_rewards = []
    episode_errors = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action, value = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done, value)
            
            state = next_state
            total_reward += reward
            
            if done:
                if 'final_error' in info:
                    episode_errors.append(info['final_error'])
                break
        
        episode_rewards.append(total_reward)
        
        # Train agent periodically
        if episode % 10 == 0 and episode > 0:
            loss = agent.train()
            
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if episode_rewards else 0
            avg_error = np.mean(episode_errors[-50:]) if episode_errors else 0
            print(f"Episode {episode:3d}: Avg Reward = {avg_reward:.3f}, Avg Error = {avg_error:.4f} rad")
    
    print("Training complete!")
    return agent


def evaluate_agent(agent, test_episodes=20):
    """Evaluate trained agent performance."""
    if agent is None:
        return
        
    print("\n=== EVALUATING TRAINED AGENT ===")
    
    env = GKPSensingEnvironment()
    test_errors = []
    test_efficiencies = []
    
    for episode in range(test_episodes):
        state = env.reset()
        
        for step in range(env.max_measurements):
            action, _ = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            
            if done:
                if 'final_error' in info:
                    test_errors.append(info['final_error'])
                    test_efficiencies.append(info['efficiency'])
                break
    
    print(f"Test Results over {test_episodes} episodes:")
    print(f"  Mean error: {np.mean(test_errors):.4f} ± {np.std(test_errors):.4f} rad")
    print(f"  Mean error: {np.mean(test_errors)*180/np.pi:.2f} ± {np.std(test_errors)*180/np.pi:.2f}°")
    print(f"  Mean efficiency: {np.mean(test_efficiencies):.3f}")
    
    return test_errors, test_efficiencies


def compare_with_classical():
    """Compare RL agent with classical fixed protocol."""
    print("\n=== COMPARING RL vs CLASSICAL PROTOCOLS ===")
    
    # Classical protocol (fixed)
    def classical_protocol(true_phase):
        env = GKPSensingEnvironment()
        env.reset(true_phase)
        
        # Fixed strategy: |0⟩ state, q then p measurements
        action_q = np.array([0.0, 0.0, 0.0, 0.8, 0.0])  # q-quadrature
        action_p = np.array([0.0, 0.0, 0.5, 0.8, 0.0])  # p-quadrature  
        action_stop = np.array([0.0, 0.0, 0.0, 0.0, 1.0])  # stop
        
        state, _, _, _ = env.step(action_q)
        state, _, _, _ = env.step(action_p)
        _, _, _, info = env.step(action_stop)
        
        return info.get('final_error', np.pi)
    
    # Test both methods
    test_phases = np.random.uniform(0, 2*np.pi, 50)
    
    classical_errors = [classical_protocol(phase) for phase in test_phases]
    
    print(f"Classical protocol:")
    print(f"  Mean error: {np.mean(classical_errors):.4f} rad ({np.mean(classical_errors)*180/np.pi:.2f}°)")
    
    if RL_AVAILABLE:
        print("Train an RL agent to compare with classical protocol!")
    else:
        print("Install PyTorch to train RL agent for comparison")


def main():
    """Main function with options."""
    print("🤖 RL-Based GKP Parameter Encoding Optimization")
    print("=" * 55)
    
    if not RL_AVAILABLE:
        print("Note: PyTorch not available. Only classical comparison will work.")
        print()
    
    print("Choose option:")
    print("1. Train RL agent to optimize sensing strategy")
    print("2. Compare classical vs RL approaches (conceptual)")
    print("3. Environment demo")
    
    try:
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            if RL_AVAILABLE:
                agent = train_rl_agent(episodes=200, max_steps=8)
                evaluate_agent(agent, test_episodes=20)
            else:
                print("PyTorch required for RL training")
        elif choice == "2":
            compare_with_classical()
        elif choice == "3":
            # Demo environment
            print("=== ENVIRONMENT DEMO ===")
            env = GKPSensingEnvironment(max_measurements=3)
            state = env.reset()
            print(f"Initial state: {state}")
            print(f"True phase: {env.true_phase:.3f} rad")
            
            # Random actions
            for i in range(3):
                action = np.random.rand(5)
                state, reward, done, info = env.step(action)
                print(f"Step {i+1}: reward={reward:.3f}, done={done}")
                if done:
                    break
        else:
            print("Invalid choice")
            
    except (KeyboardInterrupt, EOFError):
        print("\nDemo aborted")


if __name__ == "__main__":
    main() 