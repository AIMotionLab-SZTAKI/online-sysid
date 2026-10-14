from jax import numpy as jnp
import jax
import numpy as np
from functools import partial
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import cho_factor, cho_solve
from jaxlib.xla_extension import ArrayImpl
import pickle
from online_sysid.networks import (initialize_network, generate_simple_res_net, initialize_augmentation_network,
                                   generate_feedforward_net)

from typing import Optional


@jax.jit
def xsat(x: jnp.ndarray, sat: float):
    """
    Apply saturation to the state value. (see jax-sysid)

    Args:
        x (ndarray): The state.
        sat (float): The saturation limit.

    Returns:
        ndarray: The saturated value of x.
    """
    return jnp.minimum(jnp.maximum(x, -sat), sat)  # hard saturation


# MODEL WRAPPER FOR ONLINE SYSID
class SysIDModel:
    """
    SYSID mdoel for online learning. Model wrapper for subsequent model classes.
    """
    def __init__(self, ny: int, nu: int, nx: Optional[int]=0):
        """
        Initializes the model class.

        Args:
            ny (int): Output dimension.
            nu (int): Input dimension.
            nx (int, optional): State dimension (for models that apply states). Defaults to 0.
        """
        self.ny = ny
        self.nu = nu
        self.nx = nx
    
    def init_params(**args):
        """
        Initializes the parameters of the model.
        """
        raise NotImplementedError("Should be implemented in child!")

    def simulate(self, params, u, x0):
        """
        Forward simulates the model along an input trajectory, from a given initial state.
        """
        raise NotImplementedError("Should be implemented in child!")
    
    def _generate_SS_forward_function(self):
        raise NotImplementedError("Should be implemented in child!")

    def loss(self, params: ArrayImpl, data: dict):
        """
        Implements a loss function for models built on encoder-based ANN-SS models.
        """
        raise NotImplementedError("Should be implemented in child!")
    
    def initialize_recursive_update_scheme(self, theta_0, R0, batch_length, **args):
        raise NotImplementedError("Should be implemented in child!")


class ANN_SS_Model_w_state_measurements(SysIDModel):
    """
    ANN-SS model with full state measurements, i.e., with only the state transition function being parametrized:
        x_{k+1} = f(x_k, u_k)
        y_k = x_k
    """
    def __init__(self, nx: int, nu: int, xsat: float=1000.):
        """
        Initializes the class.
        Args:
            nx (int): State dimension (due to full-state emasurements, this equals the output dimensions as well.)
            nu (int): input dimensions.
            xsat (float, optional): State saturation applied during model training to avoid numerical stability loss.
                Defaults to 1000.
        """
        super().__init__(ny=nx, nu=nu, nx=nx)
        self.xsat = xsat

    def init_params(self, hidden_layers: int, nodes_per_layer: int, activation: str, key: jax.random.PRNGKey):
        """
        Initializes the model parameters.

        Args:
            hidden_layers (int): Number of hidden layers in the state transition ANN.
            nodes_per_layer (int): Number if nodes per hidden layer in the state transition ANN.
            activation (str): Type of activation function for the ANN.
            key (PRNGKey): Random key for initialization.
        
        Returns:
            param_vec (ArrayImpl): Parameters of the model flattened into a 1D array.
        """

        params_pytree = initialize_network(input_features=self.nx+self.nu, output_features=self.nx, hidden_layers=hidden_layers,
                                           nodes_per_layer=nodes_per_layer, key=key)

        params_vec, unravel_fn = ravel_pytree(params_pytree)
        self.unravel_fn = unravel_fn

        self.f_theta = self._generate_f_function(hidden_layers, activation)

        return params_vec
    
    def simulate(self, params: ArrayImpl, u: np.ndarray, x0: np.ndarray):
        """
        Forward simulates the model along an input trajectory, from a given initial state.

        Args:
            params (ArrayImpl): Parameters of the model.
            u (ndarray): Test input sequence shaped (Nt, nu).
            x0 (ndarray): Initial state corresponding to the test data, shaped (nx,).
        
        Returns:
            x_traj (ndarray): Simulated state trajectory, shaped (Nt, nx).
        """
        def step(x, u_k):
            x_next = self.f_theta(x, u_k, params)
            return x_next, x

        _, x_traj = jax.lax.scan(step, x0, u)
        return x_traj  # y = x

    def _generate_f_function(self, hidden_layers: int, act_fun: str):
        """
        Generates the ANN function for the state transition.
        """
        net = generate_simple_res_net(0, hidden_layers, act_fun)
        @jax.jit
        def state_fcn(x, u, params):
            xu = jnp.concatenate([x, u])
            return net(xu, self.unravel_fn(params))
        return state_fcn

    def _generate_SS_forward_function(self):
        """
        Generates the SS_forward function that performs a step of the model.
        """
        @jax.jit
        def SS_forward(x, u, th, sat):
            """
            Perform a forward pass of the nonlinear model. States are saturated to avoid possible explosion of state
            values in case the system is unstable.
            """
            x_next = self.f_theta(x, u, th).reshape(-1)

            # saturate states to avoid numerical issues due to instability
            x_next = xsat(x_next, sat)
            return x_next, x
        return SS_forward

    def loss(self, params: ArrayImpl, data: dict):
        """
        Implements a loss function for ANN-SS models with available full-state measurements.

        Args:
            params (ArrayImpl): Parameters of the model.
            data (dict): Replay buffer.
        
        Returns:
            mse (float): Simulation MSE (root mean-squared error).
        """
        def loss_one_segment(params, Y_seg, U_seg, x0_seg):
            # Simulate forward
            SS_forward = self._generate_SS_forward_function()
            f = partial(SS_forward, th=params, sat=self.xsat)
            _, Y_pred = jax.lax.scan(f, x0_seg.reshape(-1), U_seg)
            return jnp.mean((Y_pred - Y_seg) ** 2)
        per_segment = jax.vmap(lambda y, u, x0: loss_one_segment(params, y, u, x0))(data["Y"], data["U"], data["x0"])

        # Mask out unfilled slots -- all ops are static-shape, no dynamic slicing
        m = data["Y"].shape[0]
        mask = jnp.arange(m) < data["filled"]  # (m,) boolean, JIT-friendly
        mse = jnp.sum(jnp.where(mask, per_segment, 0.0)) / jnp.maximum(data["filled"], 1)
        return mse


class ANN_SS_Model(SysIDModel):
    def __init__(self, nu, ny, nx, encoder_lag):
        super().__init__(nu=nu, ny=ny, nx=nx)
        self.n = encoder_lag
        self.xsat = 1000.

    def init_params(self, f_hidden_layers: int, f_nodes_per_layer: int, f_activation: str,
                    encoder_hidden_layers: int, encoder_nodes_per_layer: int, encoder_activation: str,
                    key: jax.random.PRNGKey,
                    h_hidden_layers: int=1, h_nodes_per_layer: int=8, h_activation: str="tanh", use_h: bool=True,
                    h_feedthrough: bool=False):
        """
        Initializes the parameters of the f, h, and encoder networks.

        Args:
            f_hidden_layers (int): Hidden layers of the f network (state transition).
            f_nodes_per_layer (int): Nodes/neurons per hidden layer for the f network (state transition).
            f_activation (str): Activation function of the f network (state transition).
            encoder_hidden_layers (int): Hidden layers of the encoder network (state estimation).
            encoder_nodes_per_layer (int): Nodes/neurons per hidden layer for the encoder network (state estimation).
            encoder_activation (str): Activation function of the encoder network (state estimation).
            key (PRNGKey): Random key for initialization.
            h_hidden_layers (int, optional): Hidden layers of the h network (output map). Defaults to 1.
            h_nodes_per_layer (int, optional): Nodes/neurons per hidden layer for the h network (output map). Defaults
                to 8.
            h_activation (str, optional): Activation function of the h network (output map). Defaults to 'tanh'.
            use_h (bool, optional): Whether to parametrize h or not. If True, h is parametrized as an ANN with the
                specified hyperparameters. If False, An output read-out is applied with y = [I 0] x, i.e., the first ny
                state dimensions match the output. Defaults to True.
            h_feedthrough (bool, optional): If True, the output map h depends both on the state x_k and input u_k.
                Otherwise only y_k = h(x_k) is applied. Only applicabe if `use_h = True`. Defaults to False.
        
        Returns:
            params_vec (ArrayImpl): Parameters of the model flattened into a 1D array.
        """

        keys = jax.random.split(key, 3)

        params = initialize_network(input_features=self.nx+self.nu, output_features=self.nx, hidden_layers=f_hidden_layers,
                                    nodes_per_layer=f_nodes_per_layer, key=keys[0])

        encoder_params_start_idx =len(params)
        params_enc = initialize_network(input_features=self.n * (self.ny + self.nu), output_features=self.nx,
                                        hidden_layers=encoder_hidden_layers, nodes_per_layer=encoder_nodes_per_layer,
                                        key=keys[1])
        params.extend(params_enc)

        h_params_start_idx = len(params)
        if use_h:
            if h_feedthrough:
                n_in = self.nx + self.nu
            else:
                n_in = self.nx
            params_h = initialize_network(input_features=n_in, output_features=self.ny, hidden_layers=h_hidden_layers,
                                          nodes_per_layer=h_nodes_per_layer, key=keys[2])
            params.extend(params_h)

        params_vec, unravel_fn = ravel_pytree(params)
        self.unravel_fn = unravel_fn

        self.state_fcn = self._generate_f_function(f_hidden_layers, f_activation)
        self.encoder_fcn = self._generate_encoder_function(encoder_params_start_idx, encoder_hidden_layers, encoder_activation)
        self.output_fcn = self._generate_h_function(h_params_start_idx, h_hidden_layers, h_activation, use_h, h_feedthrough)

        # Precompute jacobian functions ONCE
        self.jac_enc_theta = jax.jacrev(self.encoder_fcn, argnums=2)
        self.jac_out_x = jax.jacrev(self.output_fcn, argnums=0)
        self.jac_out_theta = jax.jacrev(self.output_fcn, argnums=2)
        self.jac_state_x = jax.jacrev(self.state_fcn, argnums=0)
        self.jac_state_theta = jax.jacrev(self.state_fcn, argnums=2)

        return params_vec

    def _generate_f_function(self, hidden_layers, act_fun):
        """
        Generates the state transition ANN.
        """
        net = generate_simple_res_net(0, hidden_layers, act_fun)
        @jax.jit
        def state_fcn(x, u, params):
            xu = jnp.concatenate([x, u])
            return net(xu, self.unravel_fn(params))
        return state_fcn

    def _generate_encoder_function(self, start_idx, hidden_layers, act_fun):
        """
        Generates the encoder function.
        """
        net = generate_simple_res_net(start_idx, hidden_layers, act_fun)
        @jax.jit
        def encoder_fcn(uhist, yhist, params):
            """Encoder function for state estimation"""
            # uhist (n, nu)
            # yhist (n, ny)
            hist = jnp.concatenate([uhist, yhist], axis=1).reshape(-1)  # (n, nu+ny) -> (n*(nu+ny),)
            return net(hist, self.unravel_fn(params))
        return encoder_fcn

    def _generate_h_function(self, start_idx, hidden_layers, act_fun, use_h, feedthrough):
        """
        Generates the output map.
        """
        if not use_h:
            @jax.jit
            def output_fcn(x, u, params):
                return x[:self.ny]  # y_in_x (similar to jax_sysid implementation)
        else:
            net = generate_simple_res_net(start_idx, hidden_layers, act_fun)
            if not feedthrough:
                # output does not depend on input
                @jax.jit
                def output_fcn(x, u, params):
                    return net(x, params)
            else:
                # feedthrough
                @jax.jit
                def output_fcn(x, u, params):
                    xu = jnp.concatenate([x, u])
                    return net(xu, self.unravel_fn(params))
        return output_fcn

    def simulate(self, params, u, x0):
        """
        Forward simulates the model along an input trajectory, from a given initial state (which needs to be
        pre-computed with the encoder network separately).

        Args:
            params (ArrayImpl): Parameters of the model.
            u (ndarray): Test input sequence shaped (Nt, nu).
            x0 (ndarray): Initial state corresponding to the test data, shaped (nx,).
        
        Returns:
            tuple: containing
            - ndarray: output trajectory shaped (Nt, ny)
            - ndarray: state trajectory shaped (Nt, nx)
        """
        def step(x, u_k):
            y = self.output_fcn(x, u_k, params)
            x_next = self.state_fcn(x, u_k, params).reshape(-1)
            yx = jnp.hstack((y, x))
            return x_next, yx

        _, yx_traj = jax.lax.scan(step, x0, u)
        return yx_traj[:, :self.ny], yx_traj[:, self.ny:]

    def _generate_SS_forward_function(self):
        """
        Generates the SS_forward function that performs a step of the model.
        """

        @jax.jit
        def SS_forward(x, u, th, sat):
            """
            Perform a forward pass of the nonlinear model. States are saturated to avoid possible explosion of state
            values in case the system is unstable.
            """
            y = self.output_fcn(x, u, th)
            x_next = self.state_fcn(x, u, th).reshape(-1)

            # saturate states to avoid numerical issues due to instability
            x_next = xsat(x_next, sat)
            return x_next, y

        return SS_forward

    def loss(self, params: ArrayImpl, data: dict):
        """
        Computes the training loss absed on the replay buffer.

        Args:
            params (ArrayImpl): Current parameters of the model.
            data (dict): Replay buffer containing the data batches.
        
        Returns:
            mse (float): Mean-squared error on the training data.
        """
        def loss_one_segment(params, Y_seg, U_seg, ctx_Y, ctx_U):
            SS_forward = self._generate_SS_forward_function()
            f = partial(SS_forward, th=params, sat=self.xsat)
            x0 = self.encoder_fcn(ctx_U, ctx_Y, params)
            _, Y_pred = jax.lax.scan(f, x0.reshape(-1), U_seg)
            error = 0.5 * jnp.mean((Y_pred - Y_seg) ** 2)
            return error
        
        per_segment = jax.vmap(lambda y, u, cy, cu: loss_one_segment(params, y, u, cy, cu))(data["Y"], data["U"],
                                                                                            data["ctx_Y"], data["ctx_U"]
                                                                                            )

        # Mask out unfilled slots -- all ops are static-shape, no dynamic slicing
        m = data["Y"].shape[0]
        mask = jnp.arange(m) < data["filled"]  # (m,) boolean, JIT-friendly
        mse = jnp.sum(jnp.where(mask, per_segment, 0.0)) / jnp.maximum(data["filled"], 1)
        return mse

    def _compute_psi(self, Xhat, U, Y, ctx_Y, ctx_U, theta_prev, epsilon):
        """
        Helper function to compute the gradient vector.
        """

        N = Y.shape[0]

        # Initial sensitivity from encoder applied to context window
        s_k = self.jac_enc_theta(ctx_U, ctx_Y, theta_prev)

        def step(carry, inputs):
            s_k, R_acc, t_acc = carry
            x, u, eps = inputs

            # Output sensitivity
            dy_dtheta = (self.jac_out_x(x, u, theta_prev) @ s_k + self.jac_out_theta(x, u, theta_prev))

            # Next state sensitivity
            s_k_next = (self.jac_state_x(x, u, theta_prev) @ s_k + self.jac_state_theta(x, u, theta_prev))

            psi_row = dy_dtheta

            # Accumulate directly
            R_acc = R_acc + psi_row.T @ psi_row
            t_acc = t_acc + psi_row.T @ eps

            return (s_k_next, R_acc, t_acc), None

        psi_R_0 = jnp.zeros((theta_prev.shape[0], theta_prev.shape[0]))
        psi_theta_0 = jnp.zeros((theta_prev.shape[0],))

        (_, psi_R, psi_theta), _ = jax.lax.scan(step, (s_k, psi_R_0, psi_theta_0), (Xhat, U, epsilon.reshape(N, self.ny)))

        return psi_R, psi_theta

    def initialize_recursive_update_scheme(self, theta_0: ArrayImpl, R0: np.ndarray, batch_length: int,
                                           time_invariant: bool=False, lambda_bar: float=0.99, inverse_reg: float=1e-3,
                                           step_schedule: str="harmonic", mu_bar: float=1.0):
        """
        Initializes the recursive scheme. Pre-compiles every component such that no JIT compilation is needed during
        runtime.

        Args:
            theta_0 (ArrayImpl): Initial values of the parameter vector.
            R0 (ndarray): Initial values of the covariance matrix.
            batch_length (int): The length of the batches throughout the model learning process.
            time_invariant (bool, optional): Whether the data-generating system dynamics are time invariant or not. 
                Influences only the step size selection. If the dynamics are time-varying, an exponential forgetting
                factor is applied for the step size. Defaults to False.
            lambda_bar (float, optional): Forgetting factor for both the time-varying and time-invariant cases.
                (Note that both step-size selections can include a forgetting factor, but computed differently.)
                Defaults to 0.99.
            inverse_reg (float): Regularization coeeficient for the matrix inverse calculation that guarantees
                positive definiteness. Defaults to 1e-3.
            step_schedule (str, optional): Effective only when `time_invariant = False`. Possible choices are 
                harmonic' and 'recursive'. Defaults to 'harmonic'.
                'harmonic' -- mu_i = mu_bar / i    (for optimal asymptotic variance choose `mu_bar = 1.`)
                'recursive' -- mu_i = mu_bar * mu_{i-1} / (mu_{i-1} + lambda_i)
                                with lambda_i = lambda_bar * lambda_{i-1} + (1 - lambda_bar)
                                useful when large errors need to be handled at the beginning due to inaccurate
                                numerical differentiation.
            mu_bar (float, optional): Asymptotic step-size. i * mu_i --> mu_bar as i --> infty. Defaults to 1.
        """
        if time_invariant:
            if step_schedule == "harmonic":
                def compute_step_size_mu(i_prev, mu_prev, lam_prev):
                    mu_new = mu_bar / (i_prev + 1)
                    return mu_new, lam_prev
            elif step_schedule == "recursive":
                def compute_step_size_mu(i_prev, mu_prev, lam_prev):
                    lam_new = lambda_bar * lam_prev + (1.0 - lambda_bar)
                    mu_new = mu_bar * mu_prev / (mu_prev + lam_new)
                    return mu_new, lam_new
            else:
                raise ValueError(f"Unknown step_schedule {step_schedule!r}; choose 'harmonic' or 'recursive'.")
        else:
            def compute_step_size_mu(i_prev, mu_prev, lam_prev):
                i_new = i_prev + 1
                mu_new = (1 - lambda_bar) / (1 - lambda_bar**i_new)
                return mu_new, lam_prev

        @jax.jit
        def SS_forward(x, u, th, sat):
            """
            Perform a forward pass of the nonlinear model. States are saturated to avoid possible explosion of state values in case the system is unstable.
            """
            y = self.output_fcn(x, u, th)
            x_next = self.state_fcn(x, u, th).reshape(-1)

            # saturate states to avoid numerical issues due to instability
            x_next = xsat(x_next, sat)
            return x_next, jnp.hstack((y, x))

        def recursive_update(theta_prev, R_prev, i_prev, mu_prev, lam_prev, Y, U, ctx_Y, ctx_U):
            # step 1: forward propagate the model using previous parameters
            x0 = self.encoder_fcn(ctx_U, ctx_Y, theta_prev)
            f = partial(SS_forward, th=theta_prev, sat=self.xsat)
            _, YXhat = jax.lax.scan(f, x0.reshape(-1), U)
            Yhat = YXhat[:, :self.ny]
            Xhat = YXhat[:, self.ny:]

            # step 2: compute error
            epsilon = Y.reshape(-1) - Yhat.reshape(-1)

            # step 3-4: compute the gradient vector and its contribution towards R and theta
            psi_R, psi_theta = self._compute_psi(Xhat, U, Y, ctx_Y, ctx_U, theta_prev, epsilon)

            # step 4: advance step size and R matrix
            mu_new, lam_new = compute_step_size_mu(i_prev, mu_prev, lam_prev)
            R_new = R_prev + mu_new * (psi_R - R_prev)

            # step 5-6: invert the R matrix and advance the parameter estimate
            A = R_new + inverse_reg * jnp.eye(theta_prev.shape[0])

            # Cholesky-factorization-based inverse
            c, low = cho_factor(A)
            x = cho_solve((c, low), psi_theta)

            theta_new = theta_prev + mu_new * x.reshape(-1)

            return theta_new, R_new, mu_new, lam_new

        self.recursive_update_step = jax.jit(recursive_update)
        # Force compilation
        compiled = self.recursive_update_step.lower(
            theta_0,
            R0,
            0,
            jnp.float64(1.0),  # mu_0
            jnp.float64(0.0),  # lam_0
            np.zeros((batch_length, self.ny)),
            np.zeros((batch_length, self.nu)),
            np.zeros((self.n, self.ny)),
            np.zeros((self.n, self.nu)),
        ).compile()

        self.recursive_update_step = compiled


class LFRaugm(ANN_SS_Model):
    def __init__(self, nu, ny, nx_b, nx_a, nz_a, nw_a, baseline_model, encoder_lag, std_xb=None, std_u=None, std_y=None,
                 mu_xb=None, mu_y=None, mu_u=None):
        self.nx_b = nx_b
        self.nx_a = nx_a
        super().__init__(nu=nu, ny=ny, nx=nx_b+nx_a, encoder_lag=encoder_lag)
        self.n = encoder_lag
        self.nz_a = nz_a
        self.nw_a = nw_a
        self.xsat = 1000.
        self.baseline_model = baseline_model

        self.std_xb = jnp.array(std_xb, dtype=jnp.float64) if std_xb is not None else jnp.ones(shape=(nx_b,), dtype=jnp.float64)
        self.std_y = jnp.array(std_y, dtype=jnp.float64) if std_y is not None else jnp.ones(shape=(ny,), dtype=jnp.float64)
        self.std_u = jnp.array(std_u, dtype=jnp.float64) if std_u is not None else jnp.ones(shape=(nu,), dtype=jnp.float64)
        self.mu_xb = jnp.array(mu_xb, dtype=jnp.float64) if mu_xb is not None else jnp.zeros(shape=(nx_b,), dtype=jnp.float64)
        self.mu_y = jnp.array(mu_y, dtype=jnp.float64) if mu_y is not None else jnp.zeros(shape=(ny,), dtype=jnp.float64)
        self.mu_u = jnp.array(mu_u, dtype=jnp.float64) if mu_u is not None else jnp.zeros(shape=(nu,), dtype=jnp.float64)

        # baseline parameter regularization used by `loss` (batchwise learning), see `set_base_regularization`
        self.nominal_base_params = None
        self.base_reg_coeff = 0.

    def set_base_regularization(self, nominal_base_params: jnp.ndarray, base_reg_coeff: float=1e-3):
        """
        Sets the baseline (physical) parameter regularization for the batchwise loss. The penalty is scaled such that
        `base_reg_coeff` has the same meaning as in `initialize_recursive_update_scheme`.

        Args:
            nominal_base_params (ndarray): Nominal baseline parameters.
            base_reg_coeff (float, optional): Trade-off coefficient for baseline parameter regularization. Defaults
                to 1e-3.
        """
        self.nominal_base_params = jnp.array(nominal_base_params, dtype=jnp.float64)
        self.base_reg_coeff = base_reg_coeff

    def loss(self, params: ArrayImpl, data: dict):
        """
        Computes the training loss based on the replay buffer. The initial baseline state of each segment is the
        measured state stored in data["x0"] (normalized), while the initial augmented state is estimated by the encoder
        from the context window (data["ctx_Y"], data["ctx_U"]).

        Args:
            params (ArrayImpl): Current parameters of the model.
            data (dict): Replay buffer containing the data batches.

        Returns:
            loss (float): Mean-squared simulation error on the training data (+ baseline parameter regularization).
        """
        def loss_one_segment(Y_seg, U_seg, ctx_Y, ctx_U, xb0_seg):
            def step(x, u):
                x_next, y = self.model_step(x, u, params)
                return xsat(x_next.reshape(-1), self.xsat), y
            xa0 = self.encoder_fcn(ctx_U, ctx_Y, params)
            x0 = jnp.hstack((xb0_seg.reshape(-1), xa0.reshape(-1)))
            _, Y_pred = jax.lax.scan(step, x0, U_seg)
            return jnp.mean((Y_pred - Y_seg) ** 2)

        per_segment = jax.vmap(loss_one_segment)(data["Y"], data["U"], data["ctx_Y"], data["ctx_U"], data["x0"])

        # Mask out unfilled slots -- all ops are static-shape, no dynamic slicing
        m = data["Y"].shape[0]
        mask = jnp.arange(m) < data["filled"]  # (m,) boolean, JIT-friendly
        mse = jnp.sum(jnp.where(mask, per_segment, 0.0)) / jnp.maximum(data["filled"], 1)

        if self.nominal_base_params is not None:
            # same objective as the recursive scheme (0.5 * ||eps||^2 + 0.25 * N * rho * ||rel. dev.||^2), divided by
            # 0.5 * N * ny to match the mean-squared error scaling
            rel_dev = (self.unravel_fn(params)[-1] - self.nominal_base_params) / self.nominal_base_params
            mse = mse + 0.5 * self.base_reg_coeff / self.ny * jnp.sum(rel_dev ** 2)
        return mse

    def init_params(self, phi_hidden_layers, phi_nodes_per_layer, phi_activation, encoder_hidden_layers,
                    encoder_nodes_per_layer, encoder_activation, key):
        
        # save hyperparameters
        self.phi_hidden_layers = phi_hidden_layers
        self.phi_act_fun = phi_activation
        self.enc_hidden_layers = encoder_hidden_layers
        self.enc_act_fun = encoder_activation

        nx_a = self.nx_a
        nx_b = self.nx_b
        keys = jax.random.split(key, 9)

        # init. network parameters
        params = initialize_augmentation_network(input_features=self.nz_a, output_features=self.nw_a,
                                                 hidden_layers=phi_hidden_layers, nodes_per_layer=phi_nodes_per_layer,
                                                 key=keys[0])
        
        # init. encoder parameters
        encoder_params_start_idx = len(params)
        self.encoder_params_start_idx = encoder_params_start_idx
        params_enc = initialize_network(input_features=self.n * (self.ny + self.nu), output_features=nx_a,
                                        hidden_layers=encoder_hidden_layers, nodes_per_layer=encoder_nodes_per_layer,
                                        key=keys[1])
        params.extend(params_enc)

        # generate matrix structure for initialization
        A_bb = jnp.zeros((nx_b, nx_b), dtype=jnp.float64)
        A_ba = jnp.zeros((nx_b, nx_a), dtype=jnp.float64)
        A_ab = jax.random.uniform(key=keys[2], shape=(nx_a, nx_b), minval=-1, maxval=1, dtype=jnp.float64)
        A_aa = jax.random.uniform(key=keys[3], shape=(nx_a, nx_a), minval=-1, maxval=1, dtype=jnp.float64)

        Bu_b = jnp.zeros((nx_b, self.nu), dtype=jnp.float64)
        Bu_a = jax.random.uniform(key=keys[4], shape=(nx_a, self.nu), minval=-1, maxval=1, dtype=jnp.float64)

        Bw_bb = jnp.hstack((jnp.eye(nx_b, dtype=jnp.float64), jnp.zeros((nx_b, self.ny), dtype=jnp.float64)))
        Bw_ba = jnp.zeros((nx_b, self.nw_a), dtype=jnp.float64)
        Bw_ab = jnp.zeros((nx_a, nx_b+self.ny), dtype=jnp.float64)
        Bw_aa = jax.random.uniform(key=keys[5], shape=(nx_a, self.nw_a), minval=-1, maxval=1, dtype=jnp.float64)

        Cy_b = jnp.zeros((self.ny, nx_b), dtype=jnp.float64)
        Cy_a = jnp.zeros((self.ny, nx_a), dtype=jnp.float64)
        Dyu = jnp.zeros((self.ny, self.nu), dtype=jnp.float64)

        Dyw_b = jnp.hstack((jnp.zeros((self.ny, nx_b), dtype=jnp.float64), jnp.eye(self.ny, dtype=jnp.float64)))
        Dyw_a = jnp.zeros((self.ny, self.nw_a), dtype=jnp.float64)

        Cz_bb = jnp.vstack((jnp.eye(nx_b, dtype=jnp.float64), jnp.zeros(shape=(self.nu, nx_b), dtype=jnp.float64)))
        Cz_ba = jnp.zeros((nx_b+self.nu,nx_a), dtype=jnp.float64)
        Cz_ab = jax.random.uniform(key=keys[6], shape=(self.nz_a, nx_b), minval=-1, maxval=1,dtype=jnp.float64)
        Cz_aa = jax.random.uniform(key=keys[7], shape=(self.nz_a, nx_a), minval=-1, maxval=1, dtype=jnp.float64)

        Dzu_b = jnp.vstack((jnp.zeros(shape=(nx_b, self.nu), dtype=jnp.float64), jnp.eye(self.nu, dtype=jnp.float64)))
        Dzu_a = jax.random.uniform(key=keys[8], shape=(self.nz_a, self.nu), minval=-1, maxval=1, dtype=jnp.float64)

        Dzw_ab = jnp.zeros((self.nz_a, nx_b+self.ny), dtype=jnp.float64)

        # add interconnection matrix to optimized variables
        params.append(Dzw_ab)  # Dzw_ab = params[-23]
        params.append(A_bb)  # A_bb = params[-22]
        params.append(A_ba)  # A_ba = params[-21]
        params.append(A_ab)  # A_ab = params[-20]
        params.append(A_aa)  # A_aa = params[-19]
        params.append(Bu_b)  # Bu_b = params[-18]
        params.append(Bu_a)  # Bu_a = params[-17]
        params.append(Bw_bb)  # Bw_bb = params[-16]
        params.append(Bw_ba)  # Bw_ba = params[-15]
        params.append(Bw_ab)  # Bw_ab = params[-14]
        params.append(Bw_aa)  # Bw_aa = params[-13]
        params.append(Cy_b)  # Cy_b = params[-12]
        params.append(Cy_a)  # Cy_a = params[-11]
        params.append(Dyu)  # Dyu = params[-10]
        params.append(Dyw_b)  # Dyw_b = params[-9]
        params.append(Dyw_a)  # Dyw_a = params[-8]
        params.append(Cz_bb)  # Cz_bb = params[-7]
        params.append(Cz_ba)  # Cz_ba = params[-6]
        params.append(Cz_ab)  # Cz_ab = params[-5]
        params.append(Cz_aa)  # Cz_aa = params[-4]
        params.append(Dzu_b)  # Dzu_b = params[-3]
        params.append(Dzu_a)  # Dzu_a = params[-2]
        params.append(self.baseline_model.init_params)  # theta = params[-1]
        
        params_vec, unravel_fn = ravel_pytree(params)
        self.unravel_fn = unravel_fn

        self.augm_fcn = self._generate_augm_function(phi_hidden_layers, phi_activation)
        self.encoder_fcn = self._generate_encoder_function(encoder_params_start_idx, encoder_hidden_layers,
                                                           encoder_activation)
        self.model_step = self._generate_model_step()

        @jax.jit
        def state_fcn(x, u, th):
            return self.model_step(x, u, th)[0]

        @jax.jit
        def output_fcn(x, u, th):
            return self.model_step(x, u, th)[1]

        # Precompute jacobian functions ONCE
        self.jac_enc_theta = jax.jacrev(self.encoder_fcn, argnums=2)
        self.jac_out_x = jax.jacrev(output_fcn, argnums=0)
        self.jac_out_theta = jax.jacrev(output_fcn, argnums=2)
        self.jac_state_x = jax.jacrev(state_fcn, argnums=0)
        self.jac_state_theta = jax.jacrev(state_fcn, argnums=2)

        return params_vec
    
    def _generate_augm_function(self, hidden_layers, act_fun):
        """
        Generates the output map.
        """
        net = generate_feedforward_net(0, hidden_layers, act_fun)

        @jax.jit
        def augm_fcn(z, params):
            return net(z, self.unravel_fn(params))
        return augm_fcn
    
    def _generate_model_step(self):
        @jax.jit
        def model_step(x, u, params):
            # f : (nx+nu) --> (nx)
            # h : (nx+nu) --> (ny)

            th = self.unravel_fn(params)

            xb = x[:self.nx_b]
            xa = x[self.nx_b:]

            # zb = Cz_bb @ xb + Cz_ba @ xa + Dzu_b @ u
            zb = th[-7] @ xb + th[-6] @ xa + th[-3] @ u

            zb_x = zb[:self.nx_b]
            zb_u = zb[self.nx_b:]
            x_plus_fp = (self.baseline_model.f(zb_x*self.std_xb+self.mu_xb, zb_u*self.std_u+self.mu_u, th[-1]) - self.mu_xb) / self.std_xb
            y_fp = (self.baseline_model.h(zb_x*self.std_xb+self.mu_xb, zb_u*self.std_u+self.mu_u, th[-1]) - self.mu_y) / self.std_y
            wb = jnp.concatenate((x_plus_fp, y_fp))

            # za = Cz_ab @ xb + Cz_aa @ xa + Dzu_a @ u + Dzw_ab @ w_b
            za = th[-5] @ xb + th[-4] @ xa + th[-2] @ u + th[-23] @ wb

            # learning component evaluation
            wa = self.augm_fcn(za, params)

            # xb+ = A_bb @ xb + A_ba @ xa + Bu_b @ u + Bw_bb @ wb + Bw_ba @ wa
            xb_next = th[-22] @ xb + th[-21] @ xa + th[-18] @ u + th[-16] @ wb + th[-15] @ wa

            # xa+ = A_ab @ xb + A_aa @ xa + Bu_a @ u + Bw_ab @ wb + Bw_aa @ wa
            xa_next = th[-20] @ xb + th[-19] @ xa + th[-17] @ u + th[-14] @ wb + th[-13] @ wa
            x_next = jnp.hstack((xb_next, xa_next))

            # y = Cy_b @ xb + Cy_a @ xa + Dyu @ u + Dyw_b @ wb + Dyw_a @ wa
            y = th[-12] @ xb + th[-11] @ xa + th[-10] @ u + th[-9] @ wb + th[-8] @ wa

            return x_next, y
        return model_step
    
    def simulate(self, params, u, x0):
        """
        Forward simulates the model along an input trajectory, from a given initial state (which needs to be
        pre-computed with the encoder network separately).

        Args:
            params (ArrayImpl): Parameters of the model.
            u (ndarray): Test input sequence shaped (Nt, nu).
            x0 (ndarray): Initial state corresponding to the test data, shaped (nx,).
        
        Returns:
            tuple: containing
            - ndarray: output trajectory shaped (Nt, ny)
            - ndarray: state trajectory shaped (Nt, nx)
        """
        def step(x, u_k):
            x_next, y = self.model_step(x, u_k, params)
            yx = jnp.hstack((y, x))
            return x_next.reshape(-1), yx

        _, yx_traj = jax.lax.scan(step, x0, u)
        return yx_traj[:, :self.ny], yx_traj[:, self.ny:]
    
    def initialize_recursive_update_scheme(self, theta_0: ArrayImpl, R0: np.ndarray, batch_length: int,
                                            time_invariant: bool=False, lambda_bar: float=0.99, inverse_reg: float=1e-3,
                                            step_schedule: str="harmonic", mu_bar: float=1.0,
                                            nominal_base_params: Optional[jnp.ndarray]=None, base_reg_coeff: float=1e-3,
                                            param_scale: Optional[jnp.ndarray]=None):
            """
            Initializes the recursive scheme. Pre-compiles every component such that no JIT compilation is needed during
            runtime.

            Args:
                theta_0 (ArrayImpl): Initial values of the parameter vector.
                R0 (ndarray): Initial values of the covariance matrix.
                batch_length (int): The length of the batches throughout the model learning process.
                time_invariant (bool, optional): Whether the data-generating system dynamics are time invariant or not. 
                    Influences only the step size selection. If the dynamics are time-varying, an exponential forgetting
                    factor is applied for the step size. Defaults to False.
                lambda_bar (float, optional): Forgetting factor for both the time-varying and time-invariant cases.
                    (Note that both step-size selections can include a forgetting factor, but computed differently.)
                    Defaults to 0.99.
                inverse_reg (float): Regularization coeeficient for the matrix inverse calculation that guarantees
                    positive definiteness. Defaults to 1e-3.
                step_schedule (str, optional): Effective only when `time_invariant = False`. Possible choices are 
                    harmonic' and 'recursive'. Defaults to 'harmonic'.
                    'harmonic' -- mu_i = mu_bar / i    (for optimal asymptotic variance choose `mu_bar = 1.`)
                    'recursive' -- mu_i = mu_bar * mu_{i-1} / (mu_{i-1} + lambda_i)
                                    with lambda_i = lambda_bar * lambda_{i-1} + (1 - lambda_bar)
                                    useful when large errors need to be handled at the beginning due to inaccurate
                                    numerical differentiation.
                mu_bar (float, optional): Asymptotic step-size. i * mu_i --> mu_bar as i --> infty. Defaults to 1.
                nominal_base_params (ndarray, optional): Nominal baseline parameters. If not None, the deviation of
                    the physical (baseline) parameters compared to this nominal value are penalized. If None, no added
                    regularization is applied. Defaults to None.
                base_reg_coeff (float, optional): Trade-off coefficient for baseline parameter regularization. Defaults
                    to 1e-3.
                param_scale (ndarray, optional): Positive scaling vector s (same shape as theta). If given, the
                    Gauss-Newton update is computed in the scaled coordinates theta / s (theta itself stays unscaled),
                    which keeps R well conditioned when the parameters have very different magnitudes (e.g., physical
                    parameters). R (incl. R0) is then expressed in the scaled coordinates. If None, no scaling is
                    applied. Defaults to None.
            """
            S = jnp.ones(theta_0.shape[0]) if param_scale is None else jnp.asarray(param_scale, dtype=jnp.float64)

            if time_invariant:
                if step_schedule == "harmonic":
                    def compute_step_size_mu(i_prev, mu_prev, lam_prev):
                        mu_new = mu_bar / (i_prev + 1)
                        return mu_new, lam_prev
                elif step_schedule == "recursive":
                    def compute_step_size_mu(i_prev, mu_prev, lam_prev):
                        lam_new = lambda_bar * lam_prev + (1.0 - lambda_bar)
                        mu_new = mu_bar * mu_prev / (mu_prev + lam_new)
                        return mu_new, lam_new
                else:
                    raise ValueError(f"Unknown step_schedule {step_schedule!r}; choose 'harmonic' or 'recursive'.")
            else:
                def compute_step_size_mu(i_prev, mu_prev, lam_prev):
                    i_new = i_prev + 1
                    mu_new = (1 - lambda_bar) / (1 - lambda_bar**i_new)
                    return mu_new, lam_prev

            @jax.jit
            def SS_forward(x, u, th, sat):
                """
                Perform a forward pass of the nonlinear model. States are saturated to avoid possible explosion of state values in case the system is unstable.
                """
                x_next, y = self.model_step(x, u, th)

                # saturate states to avoid numerical issues due to instability
                x_next = xsat(x_next.reshape(-1), sat)
                return x_next, jnp.hstack((y, x))

            def recursive_update(theta_prev, R_prev, i_prev, mu_prev, lam_prev, Y, U, ctx_Y, ctx_U, xb0):
                # step 1: forward propagate the model using previous parameters
                xa0 = self.encoder_fcn(ctx_U, ctx_Y, theta_prev)  # encoder now only estimates initial augmented states
                x0 = jnp.hstack((xb0, xa0))
                f = partial(SS_forward, th=theta_prev, sat=self.xsat)
                _, YXhat = jax.lax.scan(f, x0.reshape(-1), U)
                Yhat = YXhat[:, :self.ny]
                Xhat = YXhat[:, self.ny:]

                # step 2: compute error
                epsilon = Y.reshape(-1) - Yhat.reshape(-1)

                # step 3-4: compute the gradient vector and its contribution towards R and theta (scaled coordinates)
                psi_R, psi_theta = self._compute_psi(Xhat, U, Y, ctx_Y, ctx_U, theta_prev, epsilon)
                psi_R = S[:, None] * psi_R * S[None, :]
                psi_theta = S * psi_theta

                # step 4: advance step size and R matrix
                mu_new, lam_new = compute_step_size_mu(i_prev, mu_prev, lam_prev)
                R_new = R_prev + mu_new * (psi_R - R_prev)

                # step 5-6: invert the R matrix and advance the parameter estimate
                A = R_new + inverse_reg * jnp.eye(theta_prev.shape[0])

                # Cholesky-factorization-based inverse
                c, low = cho_factor(A)
                x = cho_solve((c, low), psi_theta)

                theta_new = theta_prev + mu_new * S * x.reshape(-1)

                return theta_new, R_new, mu_new, lam_new
            
            # set up baseline regularization terms
            if nominal_base_params is not None:
                n_theta_all = theta_0.shape[0]
                n_theta_base = nominal_base_params.shape[0]
                # curvature of r_b: rho * Lambda_b^2, scaled by N (psi_R is a batch sum) and 1/2 (GN convention)
                reg_diag = 0.5 * float(batch_length) * base_reg_coeff * jnp.hstack((jnp.zeros(n_theta_all - n_theta_base),
                                                                                    1.0 / nominal_base_params**2)).astype(jnp.float64)
                theta_nom = jnp.hstack((jnp.zeros(n_theta_all - n_theta_base), nominal_base_params)).astype(jnp.float64)

            def recursive_update_with_baseline_reg(theta_prev, R_prev, i_prev, mu_prev, lam_prev, Y, U, ctx_Y, ctx_U,
                                                   xb0):
                # step 1: forward propagate the model using previous parameters
                xa0 = self.encoder_fcn(ctx_U, ctx_Y, theta_prev)  # encoder now only estimates initial augmented states
                x0 = jnp.hstack((xb0, xa0))
                f = partial(SS_forward, th=theta_prev, sat=self.xsat)
                _, YXhat = jax.lax.scan(f, x0.reshape(-1), U)
                Yhat = YXhat[:, :self.ny]
                Xhat = YXhat[:, self.ny:]

                # step 2: compute error
                epsilon = Y.reshape(-1) - Yhat.reshape(-1)

                # step 3-4: compute the gradient vector and its contribution towards R and theta (scaled coordinates)
                psi_R, psi_theta = self._compute_psi(Xhat, U, Y, ctx_Y, ctx_U, theta_prev, epsilon)
                psi_R = S[:, None] * psi_R * S[None, :]
                psi_theta = S * psi_theta

                # step 4: advance step size and R matrix
                mu_new, lam_new = compute_step_size_mu(i_prev, mu_prev, lam_prev)
                R_new = (1. - mu_new) * R_prev + mu_new * (psi_R + jnp.diag(reg_diag * S**2))

                # step 5-6: invert the R matrix and advance the parameter estimate
                A = R_new + inverse_reg * jnp.eye(theta_prev.shape[0])

                # Cholesky-factorization-based inverse
                c, low = cho_factor(A)
                x = cho_solve((c, low), psi_theta - S * reg_diag * (theta_prev - theta_nom))

                theta_new = theta_prev + mu_new * S * x.reshape(-1)

                return theta_new, R_new, mu_new, lam_new

            if nominal_base_params is None:
                self.recursive_update_step = jax.jit(recursive_update)
            else:
                self.recursive_update_step = jax.jit(recursive_update_with_baseline_reg)
            # Force compilation
            compiled = self.recursive_update_step.lower(
                theta_0,
                R0,
                0,
                jnp.float64(1.0),  # mu_0
                jnp.float64(0.0),  # lam_0
                np.zeros((batch_length, self.ny)),
                np.zeros((batch_length, self.nu)),
                np.zeros((self.n, self.ny)),
                np.zeros((self.n, self.nu)),
                np.zeros(self.nx_b)
            ).compile()

            self.recursive_update_step = compiled

    def _compute_psi(self, Xhat, U, Y, ctx_Y, ctx_U, theta_prev, epsilon):
        """
        Helper function to compute the gradient vector.
        """

        N = Y.shape[0]

        # Initial sensitivity from encoder applied to context window (only for the augmented states)
        s_k_a = self.jac_enc_theta(ctx_U, ctx_Y, theta_prev)
        n_theta = s_k_a.shape[1]
        s_k_b = jnp.zeros((self.nx_b, n_theta))  # initial baseline state sensitivity is zero (exactly known)
        
        s_k = jnp.vstack((s_k_b, s_k_a))

        def step(carry, inputs):
            s_k, R_acc, t_acc = carry
            x, u, eps = inputs

            # Output sensitivity
            dy_dtheta = (self.jac_out_x(x, u, theta_prev) @ s_k + self.jac_out_theta(x, u, theta_prev))

            # Next state sensitivity
            s_k_next = (self.jac_state_x(x, u, theta_prev) @ s_k + self.jac_state_theta(x, u, theta_prev))

            psi_row = dy_dtheta

            # Accumulate directly
            R_acc = R_acc + psi_row.T @ psi_row
            t_acc = t_acc + psi_row.T @ eps

            return (s_k_next, R_acc, t_acc), None

        psi_R_0 = jnp.zeros((theta_prev.shape[0], theta_prev.shape[0]))
        psi_theta_0 = jnp.zeros((theta_prev.shape[0],))

        (_, psi_R, psi_theta), _ = jax.lax.scan(step, (s_k, psi_R_0, psi_theta_0), (Xhat, U, epsilon.reshape(N, self.ny)))

        return psi_R, psi_theta
    
    def save_params(self, fname, params):
        th = self.unravel_fn(params)
        
        save_dict = dict()

        # LFR matrix parameter
        save_dict["Dzw_ab"] = th[-23]
        save_dict["A_bb"] = th[-22]
        save_dict["A_ba"] = th[-21]
        save_dict["A_ab"] = th[-20]
        save_dict["A_aa"] = th[-19]
        save_dict["Bu_b"] = th[-18]
        save_dict["Bu_a"] = th[-17]
        save_dict["Bw_bb"] = th[-16]
        save_dict["Bw_ba"] = th[-15]
        save_dict["Bw_ab"] = th[-14]
        save_dict["Bw_aa"] = th[-13]
        save_dict["Cy_b"] = th[-12]
        save_dict["Cy_a"] = th[-11]
        save_dict["Dyu"] = th[-10]
        save_dict["Dyw_b"] = th[-9]
        save_dict["Dyw_a"] = th[-8]
        save_dict["Cz_bb"] = th[-7]
        save_dict["Cz_ba"] = th[-6]
        save_dict["Cz_ab"] = th[-5]
        save_dict["Cz_aa"] = th[-4]
        save_dict["Dzu_b"] = th[-3]
        save_dict["Dzu_a"] = th[-2]

        # baseline parameters
        save_dict["phys_params"] = th[-1]

        # ANN parameters
        save_dict["phi_params"] = th[:self.encoder_params_start_idx]
        save_dict["enc_params"] = th[self.encoder_params_start_idx:-23]

        # normalization constants
        save_dict["std_xb"] = self.std_xb
        save_dict["std_u"] = self.std_u
        save_dict["std_y"] = self.std_y

        # other hyperparameters
        save_dict["phi_hidden_layers"] = self.phi_hidden_layers
        save_dict["phi_act_fun"] = self.phi_act_fun
        save_dict["enc_hidden_layers"] = self.enc_hidden_layers
        save_dict["enc_act_fun"] = self.enc_act_fun
        save_dict["n"] = self.n

        with open(fname, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Parameters saved successfully!")


class LFRaugm_static(ANN_SS_Model):
    def __init__(self, nu, ny, nx, nz_a, nw_a, baseline_model, std_x=None, std_u=None, std_y=None, mu_x=None, mu_u=None,
                 mu_y=None):
        super().__init__(nu=nu, ny=ny, nx=nx, encoder_lag=0)
        self.nz_a = nz_a
        self.nw_a = nw_a
        self.xsat = 1000.
        self.baseline_model = baseline_model

        self.std_x = jnp.array(std_x, dtype=jnp.float64) if std_x is not None else jnp.ones(shape=(nx,), dtype=jnp.float64)
        self.std_y = jnp.array(std_y, dtype=jnp.float64) if std_y is not None else jnp.ones(shape=(ny,), dtype=jnp.float64)
        self.std_u = jnp.array(std_u, dtype=jnp.float64) if std_u is not None else jnp.ones(shape=(nu,), dtype=jnp.float64)
        self.mu_x = jnp.array(mu_x, dtype=jnp.float64) if mu_x is not None else jnp.zeros(shape=(nx,), dtype=jnp.float64)
        self.mu_y = jnp.array(mu_y, dtype=jnp.float64) if mu_y is not None else jnp.zeros(shape=(ny,), dtype=jnp.float64)
        self.mu_u = jnp.array(mu_u, dtype=jnp.float64) if mu_u is not None else jnp.zeros(shape=(nu,), dtype=jnp.float64)

        # baseline parameter regularization used by `loss` (batchwise learning), see `set_base_regularization`
        self.nominal_base_params = None
        self.base_reg_coeff = 0.

    def set_base_regularization(self, nominal_base_params: jnp.ndarray, base_reg_coeff: float=1e-3):
        """
        Sets the baseline (physical) parameter regularization for the batchwise loss. The penalty is scaled such that
        `base_reg_coeff` has the same meaning as in `initialize_recursive_update_scheme`.

        Args:
            nominal_base_params (ndarray): Nominal baseline parameters.
            base_reg_coeff (float, optional): Trade-off coefficient for baseline parameter regularization. Defaults
                to 1e-3.
        """
        self.nominal_base_params = jnp.array(nominal_base_params, dtype=jnp.float64)
        self.base_reg_coeff = base_reg_coeff

    def loss(self, params: ArrayImpl, data: dict):
        """
        Computes the training loss based on the replay buffer. The initial state of each segment is the measured state
        stored in data["x0"] (normalized).

        Args:
            params (ArrayImpl): Current parameters of the model.
            data (dict): Replay buffer containing the data batches.

        Returns:
            loss (float): Mean-squared simulation error on the training data (+ baseline parameter regularization).
        """
        def loss_one_segment(Y_seg, U_seg, x0_seg):
            def step(x, u):
                x_next, y = self.model_step(x, u, params)
                return xsat(x_next.reshape(-1), self.xsat), y
            _, Y_pred = jax.lax.scan(step, x0_seg.reshape(-1), U_seg)
            return jnp.mean((Y_pred - Y_seg) ** 2)

        per_segment = jax.vmap(loss_one_segment)(data["Y"], data["U"], data["x0"])

        # Mask out unfilled slots -- all ops are static-shape, no dynamic slicing
        m = data["Y"].shape[0]
        mask = jnp.arange(m) < data["filled"]  # (m,) boolean, JIT-friendly
        mse = jnp.sum(jnp.where(mask, per_segment, 0.0)) / jnp.maximum(data["filled"], 1)

        if self.nominal_base_params is not None:
            # same objective as the recursive scheme (0.5 * ||eps||^2 + 0.25 * N * rho * ||rel. dev.||^2), divided by
            # 0.5 * N * ny to match the mean-squared error scaling
            rel_dev = (self.unravel_fn(params)[-1] - self.nominal_base_params) / self.nominal_base_params
            mse = mse + 0.5 * self.base_reg_coeff / self.ny * jnp.sum(rel_dev ** 2)
        return mse

    def simulate(self, params, u, x0):
            """
            Forward simulates the model along an input trajectory, from a given initial state (which needs to be
            pre-computed with the encoder network separately).
    
            Args:
                params (ArrayImpl): Parameters of the model.
                u (ndarray): Test input sequence shaped (Nt, nu).
                x0 (ndarray): Initial state corresponding to the test data, shaped (nx,).
            
            Returns:
                tuple: containing
                - ndarray: output trajectory shaped (Nt, ny)
                - ndarray: state trajectory shaped (Nt, nx)
            """
            def step(x, u_k):
                x_next, y = self.model_step(x, u_k, params)
                yx = jnp.hstack((y, x))
                return x_next.reshape(-1), yx
    
            _, yx_traj = jax.lax.scan(step, x0, u)
            return yx_traj[:, :self.ny], yx_traj[:, self.ny:]

    def init_params(self, phi_hidden_layers, phi_nodes_per_layer, phi_activation, key):
        
        # save hyperparameters
        self.phi_hidden_layers = phi_hidden_layers
        self.phi_act_fun = phi_activation

        nx = self.nx
        keys = jax.random.split(key, 3)

        # init. network parameters
        params = initialize_augmentation_network(input_features=self.nz_a, output_features=self.nw_a,
                                                 hidden_layers=phi_hidden_layers, nodes_per_layer=phi_nodes_per_layer,
                                                 key=keys[0])

        # generate matrix structure for initialization
        A_bb = jnp.zeros((nx, nx), dtype=jnp.float64)
        Bu_b = jnp.zeros((nx, self.nu), dtype=jnp.float64)

        Bw_bb = jnp.hstack((jnp.eye(nx, dtype=jnp.float64), jnp.zeros((nx, self.ny), dtype=jnp.float64)))
        Bw_ba = jnp.zeros((nx, self.nw_a), dtype=jnp.float64)

        Cy_b = jnp.zeros((self.ny, nx), dtype=jnp.float64)
        Dyu = jnp.zeros((self.ny, self.nu), dtype=jnp.float64)

        Dyw_b = jnp.hstack((jnp.zeros((self.ny, nx), dtype=jnp.float64), jnp.eye(self.ny, dtype=jnp.float64)))
        Dyw_a = jnp.zeros((self.ny, self.nw_a), dtype=jnp.float64)

        Cz_bb = jnp.vstack((jnp.eye(nx, dtype=jnp.float64), jnp.zeros(shape=(self.nu, nx), dtype=jnp.float64)))
        Cz_ab = jax.random.uniform(key=keys[1], shape=(self.nz_a, nx), minval=-1, maxval=1,dtype=jnp.float64)

        Dzu_b = jnp.vstack((jnp.zeros(shape=(nx, self.nu), dtype=jnp.float64), jnp.eye(self.nu, dtype=jnp.float64)))
        Dzu_a = jax.random.uniform(key=keys[2], shape=(self.nz_a, self.nu), minval=-1, maxval=1, dtype=jnp.float64)

        Dzw_ab = jnp.zeros((self.nz_a, nx+self.ny), dtype=jnp.float64)

        # add interconnection matrix to optimized variables
        params.append(Dzw_ab)  # Dzw_ab = params[-14]
        params.append(A_bb)  # A_bb = params[-13]
        params.append(Bu_b)  # Bu_b = params[-12]
        params.append(Bw_bb)  # Bw_bb = params[-11]
        params.append(Bw_ba)  # Bw_ba = params[-10]
        params.append(Cy_b)  # Cy_b = params[-9]
        params.append(Dyu)  # Dyu = params[-8]
        params.append(Dyw_b)  # Dyw_b = params[-7]
        params.append(Dyw_a)  # Dyw_a = params[-6]
        params.append(Cz_bb)  # Cz_bb = params[-5]
        params.append(Cz_ab)  # Cz_ab = params[-4]
        params.append(Dzu_b)  # Dzu_b = params[-3]
        params.append(Dzu_a)  # Dzu_a = params[-2]
        params.append(self.baseline_model.init_params)  # theta = params[-1]
        
        params_vec, unravel_fn = ravel_pytree(params)
        self.unravel_fn = unravel_fn

        self.augm_fcn = self._generate_augm_function(phi_hidden_layers, phi_activation)
        self.model_step = self._generate_model_step()

        @jax.jit
        def state_fcn(x, u, th):
            return self.model_step(x, u, th)[0]

        @jax.jit
        def output_fcn(x, u, th):
            return self.model_step(x, u, th)[1]

        # Precompute jacobian functions ONCE
        self.jac_out_x = jax.jacrev(output_fcn, argnums=0)
        self.jac_out_theta = jax.jacrev(output_fcn, argnums=2)
        self.jac_state_x = jax.jacrev(state_fcn, argnums=0)
        self.jac_state_theta = jax.jacrev(state_fcn, argnums=2)

        return params_vec
    
    def _generate_augm_function(self, hidden_layers, act_fun):
        """
        Generates the output map.
        """
        net = generate_feedforward_net(0, hidden_layers, act_fun)

        @jax.jit
        def augm_fcn(z, params):
            return net(z, self.unravel_fn(params))
        return augm_fcn
    
    def _generate_model_step(self):
        @jax.jit
        def model_step(x, u, params):
            # f : (nx+nu) --> (nx)
            # h : (nx+nu) --> (ny)

            th = self.unravel_fn(params)

            # zb = Cz_bb @ x + Dzu_b @ u
            zb = th[-5] @ x + th[-3] @ u

            zb_x = zb[:self.nx]
            zb_u = zb[self.nx:]
            x_plus_fp = (self.baseline_model.f(zb_x*self.std_x+self.mu_x, zb_u*self.std_u+self.mu_u, th[-1]) - self.mu_x) / self.std_x
            y_fp = (self.baseline_model.h(zb_x*self.std_x+self.mu_x, zb_u*self.std_u+self.mu_u, th[-1]) - self.mu_y) / self.std_y
            wb = jnp.concatenate((x_plus_fp, y_fp))

            # za = Cz_ab @ x + Dzu_a @ u + Dzw_ab @ w_b
            za = th[-4] @ x + th[-2] @ u + th[-14] @ wb

            # learning component evaluation
            wa = self.augm_fcn(za, params)

            # x+ = A_bb @ x + Bu_b @ u + Bw_bb @ wb + Bw_ba @ wa
            x_next = th[-13] @ x + th[-12] @ u + th[-11] @ wb + th[-10] @ wa

            # y = Cy_b @ x + Dyu @ u + Dyw_b @ wb + Dyw_a @ wa
            y = th[-9] @ x + th[-8] @ u + th[-7] @ wb + th[-6] @ wa

            return x_next, y
        return model_step
    
    def initialize_recursive_update_scheme(self, theta_0: ArrayImpl, R0: np.ndarray, batch_length: int,
                                            time_invariant: bool=False, lambda_bar: float=0.99, inverse_reg: float=1e-3,
                                            step_schedule: str="harmonic", mu_bar: float=1.0,
                                            nominal_base_params: Optional[jnp.ndarray]=None, base_reg_coeff: float=1e-3,
                                            param_scale: Optional[jnp.ndarray]=None):
            """
            Initializes the recursive scheme. Pre-compiles every component such that no JIT compilation is needed during
            runtime.

            Args:
                theta_0 (ArrayImpl): Initial values of the parameter vector.
                R0 (ndarray): Initial values of the covariance matrix.
                batch_length (int): The length of the batches throughout the model learning process.
                time_invariant (bool, optional): Whether the data-generating system dynamics are time invariant or not. 
                    Influences only the step size selection. If the dynamics are time-varying, an exponential forgetting
                    factor is applied for the step size. Defaults to False.
                lambda_bar (float, optional): Forgetting factor for both the time-varying and time-invariant cases.
                    (Note that both step-size selections can include a forgetting factor, but computed differently.)
                    Defaults to 0.99.
                inverse_reg (float): Regularization coeeficient for the matrix inverse calculation that guarantees
                    positive definiteness. Defaults to 1e-3.
                step_schedule (str, optional): Effective only when `time_invariant = False`. Possible choices are 
                    harmonic' and 'recursive'. Defaults to 'harmonic'.
                    'harmonic' -- mu_i = mu_bar / i    (for optimal asymptotic variance choose `mu_bar = 1.`)
                    'recursive' -- mu_i = mu_bar * mu_{i-1} / (mu_{i-1} + lambda_i)
                                    with lambda_i = lambda_bar * lambda_{i-1} + (1 - lambda_bar)
                                    useful when large errors need to be handled at the beginning due to inaccurate
                                    numerical differentiation.
                mu_bar (float, optional): Asymptotic step-size. i * mu_i --> mu_bar as i --> infty. Defaults to 1.
                nominal_base_params (ndarray, optional): Nominal baseline parameters. If not None, the deviation of
                    the physical (baseline) parameters compared to this nominal value are penalized. If None, no added
                    regularization is applied. Defaults to None.
                base_reg_coeff (float, optional): Trade-off coefficient for baseline parameter regularization. Defaults
                    to 1e-3.
                param_scale (ndarray, optional): Positive scaling vector s (same shape as theta). If given, the
                    Gauss-Newton update is computed in the scaled coordinates theta / s (theta itself stays unscaled),
                    which keeps R well conditioned when the parameters have very different magnitudes (e.g., physical
                    parameters). R (incl. R0) is then expressed in the scaled coordinates. If None, no scaling is
                    applied. Defaults to None.
            """
            S = jnp.ones(theta_0.shape[0]) if param_scale is None else jnp.asarray(param_scale, dtype=jnp.float64)

            if time_invariant:
                if step_schedule == "harmonic":
                    def compute_step_size_mu(i_prev, mu_prev, lam_prev):
                        mu_new = mu_bar / (i_prev + 1)
                        return mu_new, lam_prev
                elif step_schedule == "recursive":
                    def compute_step_size_mu(i_prev, mu_prev, lam_prev):
                        lam_new = lambda_bar * lam_prev + (1.0 - lambda_bar)
                        mu_new = mu_bar * mu_prev / (mu_prev + lam_new)
                        return mu_new, lam_new
                else:
                    raise ValueError(f"Unknown step_schedule {step_schedule!r}; choose 'harmonic' or 'recursive'.")
            else:
                def compute_step_size_mu(i_prev, mu_prev, lam_prev):
                    i_new = i_prev + 1
                    mu_new = (1 - lambda_bar) / (1 - lambda_bar**i_new)
                    return mu_new, lam_prev

            @jax.jit
            def SS_forward(x, u, th, sat):
                """
                Perform a forward pass of the nonlinear model. States are saturated to avoid possible explosion of state values in case the system is unstable.
                """
                x_next, y = self.model_step(x, u, th)

                # saturate states to avoid numerical issues due to instability
                x_next = xsat(x_next.reshape(-1), sat)
                return x_next, jnp.hstack((y, x))

            def recursive_update(theta_prev, R_prev, i_prev, mu_prev, lam_prev, Y, U, x0):
                # step 1: forward propagate the model using previous parameters
                f = partial(SS_forward, th=theta_prev, sat=self.xsat)
                _, YXhat = jax.lax.scan(f, x0.reshape(-1), U)
                Yhat = YXhat[:, :self.ny]
                Xhat = YXhat[:, self.ny:]

                # step 2: compute error
                epsilon = Y.reshape(-1) - Yhat.reshape(-1)

                # step 3-4: compute the gradient vector and its contribution towards R and theta (scaled coordinates)
                psi_R, psi_theta = self._compute_psi(Xhat, U, Y, theta_prev, epsilon)
                psi_R = S[:, None] * psi_R * S[None, :]
                psi_theta = S * psi_theta

                # step 4: advance step size and R matrix
                mu_new, lam_new = compute_step_size_mu(i_prev, mu_prev, lam_prev)
                R_new = R_prev + mu_new * (psi_R - R_prev)

                # step 5-6: invert the R matrix and advance the parameter estimate
                A = R_new + inverse_reg * jnp.eye(theta_prev.shape[0])

                # Cholesky-factorization-based inverse
                c, low = cho_factor(A)
                x = cho_solve((c, low), psi_theta)

                theta_new = theta_prev + mu_new * S * x.reshape(-1)

                return theta_new, R_new, mu_new, lam_new
            
            # set up baseline regularization terms
            if nominal_base_params is not None:
                n_theta_all = theta_0.shape[0]
                n_theta_base = nominal_base_params.shape[0]
                # curvature of r_b: rho * Lambda_b^2, scaled by N (psi_R is a batch sum) and 1/2 (GN convention)
                reg_diag = 0.5 * float(batch_length) * base_reg_coeff * jnp.hstack((jnp.zeros(n_theta_all - n_theta_base),
                                                                                    1.0 / nominal_base_params**2)).astype(jnp.float64)
                theta_nom = jnp.hstack((jnp.zeros(n_theta_all - n_theta_base), nominal_base_params)).astype(jnp.float64)

            
            def recursive_update_with_baseline_reg(theta_prev, R_prev, i_prev, mu_prev, lam_prev, Y, U, x0):
                # step 1: forward propagate the model using previous parameters
                f = partial(SS_forward, th=theta_prev, sat=self.xsat)
                _, YXhat = jax.lax.scan(f, x0.reshape(-1), U)
                Yhat = YXhat[:, :self.ny]
                Xhat = YXhat[:, self.ny:]

                # step 2: compute error
                epsilon = Y.reshape(-1) - Yhat.reshape(-1)

                # step 3-4: compute the gradient vector and its contribution towards R and theta (scaled coordinates)
                psi_R, psi_theta = self._compute_psi(Xhat, U, Y, theta_prev, epsilon)
                psi_R = S[:, None] * psi_R * S[None, :]
                psi_theta = S * psi_theta

                # step 4: advance step size and R matrix
                mu_new, lam_new = compute_step_size_mu(i_prev, mu_prev, lam_prev)
                R_new = (1. - mu_new) * R_prev + mu_new * (psi_R + jnp.diag(reg_diag * S**2))

                # step 5-6: invert the R matrix and advance the parameter estimate
                A = R_new + inverse_reg * jnp.eye(theta_prev.shape[0])

                # Cholesky-factorization-based inverse
                c, low = cho_factor(A)
                x = cho_solve((c, low), psi_theta - S * reg_diag * (theta_prev - theta_nom))

                theta_new = theta_prev + mu_new * S * x.reshape(-1)

                return theta_new, R_new, mu_new, lam_new

            if nominal_base_params is None:
                self.recursive_update_step = jax.jit(recursive_update)
            else:
                self.recursive_update_step = jax.jit(recursive_update_with_baseline_reg)
            # Force compilation
            compiled = self.recursive_update_step.lower(
                theta_0,
                R0,
                0,
                jnp.float64(1.0),  # mu_0
                jnp.float64(0.0),  # lam_0
                np.zeros((batch_length, self.ny)),
                np.zeros((batch_length, self.nu)),
                np.zeros(self.nx)
            ).compile()

            self.recursive_update_step = compiled

    def _compute_psi(self, Xhat, U, Y, theta_prev, epsilon):
        """
        Helper function to compute the gradient vector.
        """

        N = Y.shape[0]
        n_theta = theta_prev.shape[0]        
        s_k = jnp.zeros((self.nx, n_theta))  # initial states are exactly known

        def step(carry, inputs):
            s_k, R_acc, t_acc = carry
            x, u, eps = inputs

            # Output sensitivity
            dy_dtheta = (self.jac_out_x(x, u, theta_prev) @ s_k + self.jac_out_theta(x, u, theta_prev))

            # Next state sensitivity
            s_k_next = (self.jac_state_x(x, u, theta_prev) @ s_k + self.jac_state_theta(x, u, theta_prev))

            psi_row = dy_dtheta

            # Accumulate directly
            R_acc = R_acc + psi_row.T @ psi_row
            t_acc = t_acc + psi_row.T @ eps

            return (s_k_next, R_acc, t_acc), None

        psi_R_0 = jnp.zeros((theta_prev.shape[0], theta_prev.shape[0]))
        psi_theta_0 = jnp.zeros((theta_prev.shape[0],))

        (_, psi_R, psi_theta), _ = jax.lax.scan(step, (s_k, psi_R_0, psi_theta_0), (Xhat, U, epsilon.reshape(N, self.ny)))

        return psi_R, psi_theta
    
    def save_params(self, fname, params):
        th = self.unravel_fn(params)
        
        save_dict = dict()

        # LFR matrix parameter
        save_dict["Dzw_ab"] = th[-14]
        save_dict["A_bb"] = th[-13]
        save_dict["Bu_b"] = th[-12]
        save_dict["Bw_bb"] = th[-11]
        save_dict["Bw_ba"] = th[-10]
        save_dict["Cy_b"] = th[-9]
        save_dict["Dyu"] = th[-8]
        save_dict["Dyw_b"] = th[-7]
        save_dict["Dyw_a"] = th[-6]
        save_dict["Cz_bb"] = th[-5]
        save_dict["Cz_ab"] = th[-4]
        save_dict["Dzu_b"] = th[-3]
        save_dict["Dzu_a"] = th[-2]

        # baseline parameters
        save_dict["phys_params"] = th[-1]

        # ANN parameters
        save_dict["phi_params"] = th[:-14]

        # normalization constants
        save_dict["std_x"] = self.std_x
        save_dict["std_u"] = self.std_u
        save_dict["std_y"] = self.std_y

        # other hyperparameters
        save_dict["phi_hidden_layers"] = self.phi_hidden_layers
        save_dict["phi_act_fun"] = self.phi_act_fun

        with open(fname, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Parameters saved successfully!")
