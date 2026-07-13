import jax
from jax import numpy as jnp
import numpy as np
import flax.linen as nn

from typing import Optional


def initialize_network(input_features: int, output_features: int, hidden_layers: int, nodes_per_layer: int,
                       key: jax.random.PRNGKey):
    """
    Initializes the parameters of an ANN (feedforward network with a residual component) using the Xavier method.

    Args:
        input_features (int): Input dimension of the ANN.
        output_features (int): Output dimensions of the ANN.
        hidden_layers (int): Number of hidden layers in the ANN.
        nodes_per_layer (int): Number of neurons/nodes for each hidden layer.
        key (PRNGKey): Random key for initialization.
    
    Returns:
        parameters (list[ndarray]): Initialized parameters for the ANN.
    """
    if not jax.config.jax_enable_x64:
        # Enable 64-bit computations
        jax.config.update("jax_enable_x64", True)

    # list network architecture
    net_units = [nodes_per_layer]
    for i in range(hidden_layers-1):
        net_units.append(nodes_per_layer)
    net_units.append(output_features)

    parameters = []
    key_carry, key_w, key_b = jax.random.split(key, 3)

    for i, units in enumerate(net_units):
        if i == 0:
            # first layer weight has dim (num_units, input shape)
            init_bnd = np.sqrt(1 / input_features)
            w = jax.random.uniform(key=key_w, shape=(units, input_features), minval=-init_bnd, maxval=init_bnd, dtype=jnp.float64)
            b = jax.random.uniform(key=key_b, minval=-init_bnd, maxval=init_bnd, shape=(units,), dtype=jnp.float64)
        else:
            # if not first layer
            key_carry, key_w, key_b = jax.random.split(key_carry, 3)
            init_bnd = np.sqrt(1 / net_units[i-1])
            w = jax.random.uniform(key=key_w, shape=(units, net_units[i-1]), minval=-init_bnd, maxval=init_bnd, dtype=jnp.float64)
            b = jax.random.uniform(key=key_b, minval=-init_bnd, maxval=init_bnd, shape=(units,), dtype=jnp.float64)
        # append weights
        parameters.append(w)
        parameters.append(b)
    # add residual component: weight has (output dim, input_dim) shape and no bias is applied
    init_bnd = np.sqrt(1 / input_features)
    w = jax.random.uniform(key=key_carry, shape=(net_units[-1], input_features), minval=-init_bnd, maxval=init_bnd, dtype=jnp.float64)
    parameters.append(w)
    return parameters


def activation(x: float, activation: Optional[str]=None):
    """
    Element-wise activation function for an ANN.

    Args:
        x (float): input of the activation function.
        activation (str, optional): Type of the activation function. Possible entries are 'relu' for rectified linear, 
            'tanh' for hyperbolic tangent, 'sigmoid' for sigmoid/logistic, and 'swish' for the sigmoid linear unit
            (silu) function. If None, a linear activation is used. Defaults to None.
        
    Returns:
        y (float): Output of the activation function.
    """
    if activation is None:
        y =  x
    elif activation == 'relu':
        y = nn.relu(x)
    elif activation == 'tanh':
        y = jnp.tanh(x)
    elif activation == 'sigmoid':
        y = nn.sigmoid(x)
    elif activation == 'swish':
        y = nn.swish(x)
    else:
        raise NotImplementedError('Further activation functions should be implemented by user!')
    return y


def generate_simple_res_net(idx_start: int, hidden_layers: int, act_fun: str):
    """
    Returns a function corresponding to a feedforward ANN with a linear bypass (residual layer).

    Args:
        idx_start (int): Start index of the parameter vector associated with the current ANN.
        hidden_layers (int): Numebr of hidden layers.
        act_fun (str): Activation function.

    Returns:
        net (callable): ANN function.
    """
    def net(net_in, params):
        W = params[idx_start]
        b = params[idx_start+1]
        y_next = activation(net_in @ W.T + b, act_fun)
        for i in range(hidden_layers - 1):
            W = params[idx_start + 2*i + 2]
            b = params[idx_start + 2*i + 3]
            y_next = activation(y_next @ W.T + b, act_fun)
        W = params[idx_start + 2*hidden_layers]
        b = params[idx_start + 2*hidden_layers + 1]
        W_res = params[idx_start + 2*hidden_layers + 2]
        # output with linear activation and residual component
        y_out = y_next @ W.T + b + net_in @ W_res.T
        return y_out
    return net
