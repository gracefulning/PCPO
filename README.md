# <Project Name>

This repository contains the implementation of **PCPO()**, a barrier-based safe RL algorithm that enforces safety *before* violations happen. PCPO adds a boundary-aware barrier penalty to the objective and introduces a constraint-aware intrinsic reward that is triggered only when the agent approaches the constraint boundary.


The implementation is built upon a **pruned and adapted subset of [OmniSafe](https://github.com/PKU-Alignment/omnisafe)** (Apache License 2.0). Only the core components required to run our algorithm have been retained.
---
## ⚙️ Installation
PCPO uses MuJoCo 2.1.0 (mujoco210) for continuous-control environments.
Refer to the official installation instructions here:  
➡️ https://mujoco.org/

```bash
git clone https://github.com/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

```
## Run traning:
```
export PYTHONPATH=$PWD:$PYTHONPATH

python examples/train_policy.py --algo PCPO --env-id SafetyWalker2dVelocity-v1 --parallel 1 --total-steps 10000000 --device cuda:0 --vector-env-nums 10 --torch-threads 10
```
## License

This project is released under the Apache License 2.0.

This repository contains and modifies code derived from OmniSafe (Apache-2.0).
See LICENSE and NOTICE for details.

No personal information is included in this repository.
All modifications are documented inside modified files.