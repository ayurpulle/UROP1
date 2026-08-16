import numpy as np
import strawberryfields as sf
from strawberryfields.ops import GKP, Dgate, BSgate, MeasureX, MeasureP
import matplotlib.pyplot as plt

def bell_state_creation(theta, phi):
    """
    Create a Bell state
    """
    prog = sf.Program(2)
    with prog.context as (q1, q2):
        Sgate(theta) | q1
        Sgate(phi) | q2
        BSgate(np.pi/4, 0) | (q1, q2)
        MeasureX | q1
        MeasureX | q2
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 20})
    eng.run(prog)
    return eng.samples[0]

    