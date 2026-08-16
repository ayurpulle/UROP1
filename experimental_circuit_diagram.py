import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow
import numpy as np

# Create the figure and axis
fig, ax = plt.subplots(1, 1, figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.set_aspect('equal')

# Define colors
laser_color = '#FF6B6B'
opo_color = '#4ECDC4'
bs_color = '#45B7D1'
detector_color = '#96CEB4'
mirror_color = '#FECA57'
control_color = '#9B59B6'

# Helper function to draw components
def draw_laser(x, y, label):
    """Draw laser source"""
    rect = FancyBboxPatch((x-0.3, y-0.2), 0.6, 0.4, 
                         boxstyle="round,pad=0.05", 
                         facecolor=laser_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, 'LASER', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(x, y-0.5, label, ha='center', va='center', fontsize=7)

def draw_opo(x, y, label):
    """Draw Optical Parametric Oscillator"""
    rect = FancyBboxPatch((x-0.4, y-0.25), 0.8, 0.5, 
                         boxstyle="round,pad=0.05", 
                         facecolor=opo_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y+0.05, 'OPO', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x, y-0.15, '(KTP)', ha='center', va='center', fontsize=7)
    ax.text(x, y-0.6, label, ha='center', va='center', fontsize=8, fontweight='bold')

def draw_beamsplitter(x, y, ratio, angle=45):
    """Draw beam splitter"""
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Create the beam splitter as a line
    length = 0.4
    x1 = x - length/2 * np.cos(angle_rad)
    y1 = y - length/2 * np.sin(angle_rad)
    x2 = x + length/2 * np.cos(angle_rad)
    y2 = y + length/2 * np.sin(angle_rad)
    
    ax.plot([x1, x2], [y1, y2], color=bs_color, linewidth=4)
    ax.text(x+0.3, y+0.3, f'{ratio}', ha='center', va='center', fontsize=8, fontweight='bold')

def draw_detector(x, y, label, det_type='single'):
    """Draw photodetector"""
    if det_type == 'single':
        circle = Circle((x, y), 0.15, facecolor=detector_color, edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, 'D', ha='center', va='center', fontsize=8, fontweight='bold')
    else:  # homodyne
        rect = Rectangle((x-0.2, y-0.15), 0.4, 0.3, facecolor=detector_color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, 'HD', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(x, y-0.4, label, ha='center', va='center', fontsize=7)

def draw_mirror(x, y, angle=0):
    """Draw mirror"""
    angle_rad = np.radians(angle)
    length = 0.3
    x1 = x - length/2 * np.cos(angle_rad + np.pi/2)
    y1 = y - length/2 * np.sin(angle_rad + np.pi/2)
    x2 = x + length/2 * np.cos(angle_rad + np.pi/2)
    y2 = y + length/2 * np.sin(angle_rad + np.pi/2)
    
    ax.plot([x1, x2], [y1, y2], color=mirror_color, linewidth=6)

def draw_phase_modulator(x, y, label):
    """Draw phase modulator"""
    rect = Rectangle((x-0.2, y-0.15), 0.4, 0.3, facecolor='lightblue', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, 'φ', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(x, y-0.4, label, ha='center', va='center', fontsize=7)

def draw_beam(x1, y1, x2, y2, color='red', linewidth=2, linestyle='-'):
    """Draw optical beam"""
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, linestyle=linestyle)

def draw_control_box(x, y, w, h, label):
    """Draw control/processing box"""
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                         boxstyle="round,pad=0.05", 
                         facecolor=control_color, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Title
ax.text(10, 13.5, 'Experimental Circuit for GKP State Generation and Quantum Sensing', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# ============================================================================
# SECTION 1: SQUEEZED LIGHT GENERATION
# ============================================================================
ax.text(2, 12.5, 'Section 1: Squeezed Light Generation', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='blue')

# Pump laser
draw_laser(1, 11.5, '1064 nm\nCW Laser')

# OPO for squeezing
draw_opo(3.5, 11.5, 'Squeezed Vacuum\nr = 1.2 (3 dB)')

# Beam from laser to OPO
draw_beam(1.3, 11.5, 3.1, 11.5, color='red')

# ============================================================================
# SECTION 2: PHOTON SUBTRACTION FOR CAT STATES
# ============================================================================
ax.text(7, 12.5, 'Section 2: Cat State Generation', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='blue')

# Weak beam splitter for photon subtraction
draw_beamsplitter(6, 11.5, 'R=0.1')

# Single photon detector
draw_detector(6.7, 12.2, 'Single Photon\nDetector', 'single')

# Post-selection logic
draw_control_box(6.7, 10.5, 1.2, 0.6, 'Post-Select\nn=1')

# Beams
draw_beam(3.9, 11.5, 5.7, 11.5, color='green')  # Squeezed light
draw_beam(6.3, 11.8, 6.7, 12.05, color='orange')  # To detector
draw_beam(6.3, 11.2, 6.7, 10.8, color='purple')  # To post-selection

# Cat state output
ax.text(8.5, 11.5, 'Small Cat State\n|α⟩ + |-α⟩', ha='center', va='center', 
        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
draw_beam(6.3, 11.5, 8, 11.5, color='purple')

# ============================================================================
# SECTION 3: CAT STATE BREEDING
# ============================================================================
ax.text(11, 12.5, 'Section 3: Cat State Breeding', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='blue')

# State preparation for breeding
ax.text(9.5, 10.5, 'Cat State 1', ha='center', va='center', fontsize=9)
ax.text(9.5, 9.5, 'Cat State 2', ha='center', va='center', fontsize=9)

# 50:50 beam splitter for breeding
draw_beamsplitter(11, 10, '50:50')

# Vacuum detector for post-selection
draw_detector(12, 10.7, 'Vacuum\nDetector', 'single')

# Enhanced cat state output
ax.text(13, 9.3, 'Enhanced\nCat State', ha='center', va='center', 
        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

# Beams for breeding
draw_beam(9.8, 10.5, 10.7, 10.3, color='purple')
draw_beam(9.8, 9.5, 10.7, 9.7, color='purple')
draw_beam(11.3, 10.3, 12, 10.55, color='orange')  # To detector
draw_beam(11.3, 9.7, 13, 9.5, color='darkgreen')  # Enhanced cat

# ============================================================================
# SECTION 4: GKP STATE GENERATION
# ============================================================================
ax.text(15.5, 12.5, 'Section 4: GKP Generation', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='blue')

# Additional squeezing stage
draw_opo(14.5, 9.3, 'Additional\nSqueezing')

# p-squeezed reference
draw_opo(15, 11, 'p-Squeezed\nReference')

# Final beam splitter for GKP
draw_beamsplitter(16, 10, '50:50')

# Homodyne detector for projection
draw_detector(17, 10.7, 'Homodyne\nDetector', 'homodyne')

# GKP state output
ax.text(17.5, 9.3, 'GKP Probe\nState', ha='center', va='center', 
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.4", facecolor='gold', alpha=0.8))

# Beams for GKP generation
draw_beam(13.3, 9.5, 14.1, 9.3, color='darkgreen')
draw_beam(14.9, 9.3, 15.7, 9.7, color='blue')
draw_beam(15.3, 11, 15.7, 10.3, color='cyan')
draw_beam(16.3, 10.3, 17, 10.55, color='magenta')
draw_beam(16.3, 9.7, 17.5, 9.5, color='gold')

# ============================================================================
# SECTION 5: QUANTUM SENSING PROTOCOL
# ============================================================================
ax.text(10, 7.5, 'Section 5: Quantum Sensing Protocol', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='red')

# Unknown parameter interaction
draw_phase_modulator(10, 6.5, 'Unknown\nDisplacement\nδx')

# Adaptive homodyne detection
draw_detector(12, 6.5, 'Adaptive\nHomodyne', 'homodyne')

# Local oscillator with phase control
draw_laser(14, 5.5, 'Local\nOscillator')
draw_phase_modulator(13, 5.5, 'Phase\nControl θ')

# Beam splitter for homodyne
draw_beamsplitter(11.5, 6.5, '50:50')

# Data processing and control
draw_control_box(15.5, 6.5, 2, 1, 'Bayesian\nParameter\nEstimation')
draw_control_box(15.5, 4.5, 2, 1, 'Fisher Info\nOptimization')

# Sensing beams
draw_beam(17.5, 9, 17.5, 7, color='gold')  # GKP down
draw_beam(17.5, 7, 10, 7, color='gold')    # To sensing
draw_beam(10, 7, 10, 6.8, color='gold')    # To modulator
draw_beam(10, 6.2, 11.2, 6.3, color='red')  # After displacement
draw_beam(11.8, 6.7, 12, 6.65, color='red')  # To detector

# LO beam
draw_beam(14, 5.8, 13, 5.8, color='orange')
draw_beam(12.7, 5.5, 11.8, 6.2, color='orange')

# Control signals (dashed lines)
draw_beam(14.5, 6.5, 13, 5.8, color='purple', linestyle='--', linewidth=1)
draw_beam(14.5, 4.5, 13.3, 5.2, color='purple', linestyle='--', linewidth=1)

# ============================================================================
# SECTION 6: MEASUREMENT AND FEEDBACK
# ============================================================================
ax.text(10, 3.5, 'Section 6: Real-time Control and Data Processing', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='red')

# Data flow diagram
draw_control_box(7, 2.5, 2.5, 0.8, 'Measurement\nData Stream')
draw_control_box(10, 2.5, 2.5, 0.8, 'Bayesian\nUpdate')
draw_control_box(13, 2.5, 2.5, 0.8, 'Phase\nOptimization')
draw_control_box(16, 2.5, 2.5, 0.8, 'Convergence\nCheck')

# Data flow arrows
ax.annotate('', xy=(8.7, 2.5), xytext=(8.3, 2.5), 
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(11.7, 2.5), xytext=(11.3, 2.5), 
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(14.7, 2.5), xytext=(14.3, 2.5), 
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Feedback loop
ax.annotate('', xy=(13, 1.8), xytext=(16, 1.8), 
            arrowprops=dict(arrowstyle='->', lw=2, color='blue', linestyle='--'))
ax.text(14.5, 1.5, 'Feedback Loop', ha='center', va='center', fontsize=9, color='blue')

# ============================================================================
# LEGEND AND SPECIFICATIONS
# ============================================================================
ax.text(1, 1.5, 'Key Specifications:', ha='left', va='center', fontsize=11, fontweight='bold')
ax.text(1, 1.1, '• Squeezing: 3-6 dB', ha='left', va='center', fontsize=9)
ax.text(1, 0.8, '• Detection Efficiency: >90%', ha='left', va='center', fontsize=9)
ax.text(1, 0.5, '• Phase Stability: <1 mrad RMS', ha='left', va='center', fontsize=9)
ax.text(1, 0.2, '• Success Rate: ~1% (post-selection)', ha='left', va='center', fontsize=9)

ax.text(18, 1.5, 'Performance:', ha='right', va='center', fontsize=11, fontweight='bold')
ax.text(18, 1.1, '• Precision: 3-5x beyond SQL', ha='right', va='center', fontsize=9)
ax.text(18, 0.8, '• Measurement Efficiency: 10x', ha='right', va='center', fontsize=9)
ax.text(18, 0.5, '• Sensitivity: 10⁻¹⁵ m/√Hz', ha='right', va='center', fontsize=9)
ax.text(18, 0.2, '• Bandwidth: 1 kHz (limited by feedback)', ha='right', va='center', fontsize=9)

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig('GKP_quantum_sensing_circuit.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show() 