"""
01 — Basic quantum teleportation on Braket
-------------------------------------------
Prepare a state on q0, teleport it to q2 using a Bell pair shared on (q1, q2),
then verify by inverting the preparation on q2. If teleportation worked,
P(q2 = |0>) should be ~1.0.

Deferred-measurement form: uses CNOT/CZ instead of mid-circuit measurement
+ classical feed-forward. Mathematically equivalent, works everywhere in Braket.
"""

import numpy as np
from braket.circuits import Circuit
from braket.devices import LocalSimulator


def teleport_circuit(theta: float, phi: float) -> Circuit:
    c = Circuit()
    # 1. prepare |psi> on q0
    c.ry(0, theta).rz(0, phi)
    # 2. Bell pair on (q1, q2)
    c.h(1).cnot(1, 2)
    # 3. Bell-basis measurement on (q0, q1) (deferred)
    c.cnot(0, 1).h(0)
    # 4. corrections on q2
    c.cnot(1, 2).cz(0, 2)
    # 5. verify: undo prep on q2 — should collapse to |0>
    c.rz(2, -phi).ry(2, -theta)
    c.probability(target=[2])
    return c


if __name__ == "__main__":
    device = LocalSimulator()
    for theta, phi in [(0, 0), (np.pi/3, np.pi/4), (np.pi/2, np.pi/7), (np.pi, 0)]:
        probs = device.run(teleport_circuit(theta, phi), shots=2000).result().values[0]
        print(f"θ={theta:.2f}  φ={phi:.2f}   P(|0>)={probs[0]:.3f}  "
              f"(should be ≈ 1.000)")
