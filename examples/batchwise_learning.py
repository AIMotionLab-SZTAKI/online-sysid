import jax
from jax import numpy as jnp
from online_sysid import ANN_SS_Model, OnlineTrainer, TrainerState, init_online_buffer, insert_segment
from matplotlib import pyplot as plt
import numpy as np
import time
import optax
import os


jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

# load data
cwd = os.path.dirname(os.path.abspath(__file__))
train_data = np.load(os.path.join(cwd, "data", "bicycle_data_train_ti.npz"))
u_train = train_data["u"].reshape(-1, 1)
y_train = train_data["y"]

test_data = np.load(os.path.join(cwd, "data", "bicycle_data_test_ti.npz"))
u_test = test_data["u"].reshape(-1, 1)
y_test = test_data["y"]

# HYPERPARAMETERS
nx = 2  # model order (state dimension)
nu = 1  # output dim.
ny = 1  # input dim.
f_hl = 2  # hidden layers in f net (state transition)
f_nodes = 4  # nodes per layer in f net
f_act = "tanh"  # activation function in f net
enc_hl = 1  # hidden layers in encoder net
enc_nodes = 16  # nodes per layer in encoder net
enc_act = "tanh"  # activation function in encoder net
encoder_lag = 5  # encoder lag (state initialization window)
batch_length = 25  # batch length (N)
buffer_size = 20  # memory of the replay buffer (m)
adam_steps = 20  # ADAM steps per training phase
lbfgs_steps = 2  # L-BFGS steps per training phase
seed = 0  # random seed for initialization

key = jax.random.PRNGKey(seed)  

# initialize model
model = ANN_SS_Model(nx=nx, nu=nu, ny=ny, encoder_lag=encoder_lag)
params = model.init_params(f_hidden_layers=f_hl, f_nodes_per_layer=f_nodes, f_activation=f_act,
                           encoder_hidden_layers=enc_hl, encoder_nodes_per_layer=enc_nodes, encoder_activation=enc_act,
                           use_h=False, key=key)

# initialize data buffer
buffer = init_online_buffer(m=buffer_size, N=batch_length, nu=nu, nx=nx, ny=ny, n=encoder_lag)

# initialize trainers
trainer_adam = OnlineTrainer(model)
opt = optax.lbfgs()
opt._name = "lbfgs"
trainer_lbfgs = OnlineTrainer(model, opt)
state_adam = trainer_adam.init_optimizer(params)
state_lbfgs = trainer_lbfgs.init_optimizer(params)

# take dummy training steps to force offline JIT compilation (no re-compile is needed online)
_, dummy_loss = trainer_adam.update(state_adam, buffer)  # dummy training step for initial JIT compile
dummy_loss.block_until_ready()
dummy_state = state_lbfgs
for _ in range(lbfgs_steps):
    dummy_state, dummy_loss = trainer_lbfgs.update(dummy_state, buffer)  # dummy training step for initial JIT compile
dummy_loss.block_until_ready()

Params = [params]
Times = []
batch_nums = int((u_train.shape[0] - encoder_lag) / batch_length)
for batch_idx in range(batch_nums):

    # new batch arrives; first batch also carries the bootstrap context window
    start = encoder_lag + batch_idx * batch_length
    if batch_idx == 0:
        u = u_train[:encoder_lag + batch_length]
        y = y_train[:encoder_lag + batch_length]
    else:
        u = u_train[start:start + batch_length]
        y = y_train[start:start + batch_length]
    buffer = insert_segment(buffer, y, u)

    # train on replay buffer
    t_start = time.time()
    for _ in range(adam_steps):
        state_adam, loss = trainer_adam.update(state_adam, buffer)
    state_lbfgs = TrainerState(state_adam.params, state_lbfgs.opt_state)
    for _ in range(lbfgs_steps):
        state_lbfgs, loss = trainer_lbfgs.update(state_lbfgs, buffer)
    state_adam = TrainerState(state_lbfgs.params, state_adam.opt_state)
    t_end = time.time()
    Times.append(t_end - t_start)
    Params.append(state_lbfgs.params)
    print(f"Batch {batch_idx:02d} | Loss: {loss:.4e}")

# JIT-compile the evaluation function
@jax.jit
def eval_param(param):
    x0 = model.encoder_fcn(u_test[:encoder_lag, :], y_test[:encoder_lag, :], param)
    yhat, _ = model.simulate(param, u_test[encoder_lag:, :], x0)
    return yhat

# Vectorize over all parameters (parallel evaluation)
yhat_all = jax.vmap(eval_param)(jnp.stack(Params))
yhat = yhat_all[-1, :, 0]  # sim. results with last params

# Vectorized loss computation
errors = y_test[encoder_lag:, :] - yhat_all
test_losses = np.sqrt(np.mean(errors**2, axis=(1, 2)))

print(f"Avg. wall-time (CPU time): {1000 * np.mean(np.array(Times))} [ms]")

plt.figure()
plt.title("Wall-time (CPU time)")
plt.plot(1000 * np.array(Times))
plt.xlabel('Batch number')
plt.ylabel('Training time [ms]')
plt.grid()
plt.show(block=False)

dt = 0.01
t_batch = ((np.arange(test_losses.shape[0]-1) + 1) * batch_length + encoder_lag) * dt
t_batch = np.hstack((np.zeros(1), t_batch))

auc = np.trapezoid(test_losses, t_batch)
print(f"AUC (t_batch vs test_losses): {auc:.6f} rad·s")

plt.figure()
plt.title("Convergence curve")
plt.plot(t_batch, test_losses, '-')
plt.axhline(6e-3, ls='--', c="r")
plt.xlabel('Sim. time [s]')
plt.ylabel('Test RMSE')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

plt.figure(layout="tight")
plt.title("Simulation with final model")
test_idx = np.arange(u_test.shape[0])
plt.plot(test_idx, y_test[:, 0], label="True data")
plt.plot(test_idx[encoder_lag:], y_test[encoder_lag:, 0] - yhat, label="Sim. error")
plt.legend()
plt.grid()
plt.xlabel("Sim. index")
plt.ylabel("y")
plt.show()
