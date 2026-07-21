# ONLINE SysID
A batch-wise learning approach and a recursive parameter update scheme for online learning of neural state-space (ANN-SS) models with an encoder network for initial state estimation. Contains the code implementation and example scripts for the paper titled *Online learning of neural state-space models*. The paper is available on [arXiv](https://arxiv.org/abs/2607.17614).

## Installation
**1. Clone repository**
```bash
git clone https://github.com/AIMotionLab-SZTAKI/online-sysid.git
cd online-sysid
```

**2. Create a virtual environment (recommended)**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install package and dependencies**

Install only the package and minimal required dependencies:
```bash
pip install -e .
```

Or, install dependencies also for the example scripts:
```bash
pip install -e ".[examples]"
```

## Example usage
The following script contains the basics of the toolbox. For more advanced examples about the batch-wise learning method and the recursive scheme, see the [examples](examples/) folder.

```python
from online_sysid import ANN_SS_Model
import jax
import numpy as np
import jax.numpy as jnp
import time


# create model
ny = 1
nu = 1
enc_lag = 5
model = ANN_SS_Model(nx=2, nu=nu, ny=ny, encoder_lag=enc_lag)

# initialize model parameters
key = jax.random.PRNGKey(0)  
theta = model.init_params(f_hidden_layers=2, f_nodes_per_layer=8, f_activation="tanh", encoder_hidden_layers=2,
                          encoder_nodes_per_layer=8, encoder_activation="tanh", use_h=False, key=key)

# initialize recursive update (offline JIT compilation)
R0 = 1.e4 * jnp.eye(theta.shape[0])
batch_length = 10

# see docstrings for further options
model.initialize_recursive_update_scheme(theta, R0, batch_length, time_invariant=True)

# initialize variables for the recursive estimation
theta_prev = theta.copy()
R_prev = R0.copy()
batch_idx = 0
mu_prev = 1.  # previous step-size -- only valid if step_schedule is not "harmonic"
lambda_prev = 0.  # previous forgetting factor -- only valid if step_schedule is "recursive" and time_invariant is True

# use dummy batch data
y_batch = np.zeros((batch_length, ny))
u_batch = np.zeros((batch_length, nu))

# also use dummy data for the encoder network (state initialization)
y_hist = np.zeros((enc_lag, ny))
u_hist = np.zeros((enc_lag, nu))

# evaluate the recursive step
t_start = time.time()
theta_new, R_new, mu_new, lambda_new = model.recursive_update_step(theta_prev, R_prev, batch_idx, mu_prev, lambda_prev,
                                                                   y_batch, u_batch, y_hist, u_hist)
t_end = time.time()

print(f"Update step compited in {1000*(t_end - t_start)} ms.")  # Update step completed in 1.8012523651123047 ms.
```

## Citation
If you use this repository in your research, please cite:
```bibtex
@article{gyorok_online_2026,
      title={Data-driven augmentation of first-principles models under constraint-free well-posedness and stability guarantees}, 
      author={Bendegúz Györök and Roel Drenth and Chris Verhoek and Tamás Péni and Maarten Schoukens and Roland Tóth},
      year={2026},
      journal={arXiv preprint: arXiv:2607.17614},
}
```

## License
This project is licensed under the BSD 3-Clause. See the [LICENSE](/LICENSE) file for details.

## Funding
This work was funded by the Air Force Office of Scientific Research under award number FA8655-23-1-7061 and by the European Union (ERC, COMPLETE, 101075836). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## Contact
For questions or collaboration, contact the corresponding author: [gyorokbende@sztaki.hu](mailto:gyorokbende@sztaki.hu)
