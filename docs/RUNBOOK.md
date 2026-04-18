# Runbook — reproduce the experiments

If you want to redo these runs (or pick up where we left off), here's the exact sequence.

## Prerequisites

1. AWS account with Amazon Braket access
2. Braket notebook running with the `conda_braket` kernel
3. `matplotlib` and `scikit-learn` (both pre-installed on conda_braket)

## Step-by-step

### Step 1 — Smoke test (30 sec)

Paste contents of `notebooks/01_teleport_basic.py` into a cell. Run.
Expect: four lines, each showing `P(|0>) ≈ 1.000`. Confirms teleportation works.

### Step 2 — Channel duality sweep (~3 min)

Paste contents of `notebooks/02_channel_duality.py` into a cell. Run.
Expect: 22 rows of numbers (depolarizing × 11 + dephasing × 11) and a
two-panel plot saved as `channel_duality.png`.

### Step 3 — Variational encoding discovery (~3-5 min)

Paste contents of `notebooks/03_variational_encoding.py` into a cell. Run.
Expect: a table of optimal encodings per noise type, an MLP extrapolation
test, and a scatter plot saved as `encoding_discovery.png`.

## Turning it off

**Don't forget to Stop the Braket notebook when done.** Otherwise you keep
paying ~$0.05/hr for the idle instance.

1. AWS Console → Amazon Braket → Notebooks
2. Select your notebook → Actions → Stop

Files are preserved; restart anytime.

## Cost summary

- Full run of all three scripts on LocalSimulator: **$0 compute**
- Notebook uptime during run: ~15 min × $0.05/hr = **~$0.013**
- Plus EBS storage (~$0.50/month while the notebook exists, even stopped)

## Running on real QPU (not done; noted for future)

Replace `LocalSimulator()` with:

```python
from braket.aws import AwsDevice
device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1")
```

Cost at ~$10-30 per noise point per protocol. For Step 2 (33 points) this
becomes $300-1000 — do NOT just rerun the sweep on hardware. Pick 2-3 noise
points and one protocol, or use a managed simulator (SV1 @ $0.075/min).
