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

# evaluate with dummy data for the example

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

print(f"Update step compited in {1000*(t_end - t_start)} ms.")  # Update step compited in 1.8012523651123047 ms.
