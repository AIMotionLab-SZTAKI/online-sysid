from jax import numpy as jnp
import jax
import numpy as np
from functools import partial
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import cho_factor, cho_solve
from jaxlib.xla_extension import ArrayImpl
from online_sysid.networks import initialize_network, generate_simple_res_net

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
    def __init__(self, nu, ny, nx, encoder_lag, forgetting_factor_lambda=1.):
        super().__init__(nu=nu, ny=ny, nx=nx)
        self.n = encoder_lag
        self.xsat = 1000.
        self.forgetting_factor_lambda = forgetting_factor_lambda

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
