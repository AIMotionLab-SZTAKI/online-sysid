import jax
import optax
from flax import struct


@struct.dataclass
class TrainerState:
    params: list
    opt_state: optax.OptState


class OptaxWrapper:
    """
    Wraps the L-BFGS and Adam solvers with one common class to enable the use of both optimizers via the same synthax.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        if getattr(optimizer, "_name", None) == "lbfgs":
            self._needs_value = True
            self._needs_grad = True
            self._needs_value_fn = True
        else:
            self._needs_value = False
            self._needs_grad = False
            self._needs_value_fn = False

    def init(self, params):
        return self.optimizer.init(params)

    def update(self, grads, state, params, value=None, value_fn=None):
        kwargs = {}

        if self._needs_value:
            if value is None:
                raise ValueError("Optimizer requires `value` (e.g. L-BFGS).")
            kwargs["value"] = value

        if self._needs_grad:
            kwargs["grad"] = grads

        if self._needs_value_fn:
            if value_fn is None:
                raise ValueError("Optimizer requires `value_fn` (e.g. L-BFGS line search).")
            kwargs["value_fn"] = value_fn

        return self.optimizer.update(grads, state, params, **kwargs)


class OnlineTrainer:
    """
    Online trainer object with a specific optimizer.
    """
    def __init__(self, model, optimizer=None):
        self.model = model
        if optimizer is None:
            self.optimizer = OptaxWrapper(optax.adam(learning_rate=1e-4, b1=0.9, b2=0.999))
        else:
            self.optimizer = OptaxWrapper(optimizer)
        self.train_step = self.make_train_step()

    def init_optimizer(self, init_params):
        opt_state = self.optimizer.init(init_params)
        state = TrainerState(init_params, opt_state)
        return state

    def update(self, state: TrainerState, dataset):
        return self.train_step(state, dataset)

    def make_train_step(self):
        @jax.jit
        def train_step(state: TrainerState, data):

            loss, grads = jax.value_and_grad(self.model.loss)(state.params, data)
            updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params, value=loss,
                                                       value_fn=lambda p: self.model.loss(p, data))
            params = optax.apply_updates(state.params, updates)

            return TrainerState(params, opt_state), loss

        return train_step
