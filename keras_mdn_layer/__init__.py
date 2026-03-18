"""
A Mixture Density Layer for Keras
cpmpercussion: Charles Martin (University of Oslo) 2018
https://github.com/cpmpercussion/keras-mdn-layer

Hat tip to [Omimo's Keras MDN layer](https://github.com/omimo/Keras-MDN)
for a starting point for this code.

Provided under MIT License
"""

import keras
from keras import layers, ops
import numpy as np
import math


def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent
    NaN in loss."""
    return ops.elu(x) + 1 + keras.backend.epsilon()


class MDN(layers.Layer):
    """A Mixture Density Network Layer for Keras.
    This layer has a few tricks to avoid NaNs in the loss function when training:
        - Activation for variances is ELU + 1 + 1e-8 (to avoid very small values)
        - Mixture weights (pi) are trained in as logits, not in the softmax space.

    A loss function needs to be constructed with the same output dimension and number of mixtures.
    A sampling function is also provided to sample from distribution parametrised by the MDN outputs.
    """

    def __init__(self, output_dimension, num_mixtures, **kwargs):
        super(MDN, self).__init__(**kwargs)
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.mdn_mus = layers.Dense(self.num_mix * self.output_dim, name='mdn_mus')
        self.mdn_sigmas = layers.Dense(self.num_mix * self.output_dim, activation=elu_plus_one_plus_epsilon, name='mdn_sigmas')
        self.mdn_pi = layers.Dense(self.num_mix, name='mdn_pi')

    def build(self, input_shape):
        self.mdn_mus.build(input_shape)
        self.mdn_sigmas.build(input_shape)
        self.mdn_pi.build(input_shape)
        super(MDN, self).build(input_shape)

    def call(self, x, mask=None):
        return ops.concatenate([self.mdn_mus(x),
                                self.mdn_sigmas(x),
                                self.mdn_pi(x)],
                               axis=-1)

    def compute_output_shape(self, input_shape):
        """Returns output shape, showing the number of mixture parameters."""
        return (input_shape[0], (2 * self.output_dim * self.num_mix) + self.num_mix)

    def get_config(self):
        config = {
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix
        }
        base_config = super(MDN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _split_mdn_params(y_pred, output_dim, num_mixes):
    """Split MDN output tensor into mu, sigma, and pi components.
    Works with keras ops for use inside loss/metric functions."""
    total = 2 * num_mixes * output_dim + num_mixes
    y_pred = ops.reshape(y_pred, (-1, total))
    mu = y_pred[:, :num_mixes * output_dim]
    sigma = y_pred[:, num_mixes * output_dim:2 * num_mixes * output_dim]
    pi_logits = y_pred[:, -num_mixes:]
    return mu, sigma, pi_logits


def _mdn_log_prob(y_true, mu, sigma, pi_logits, output_dim, num_mixes):
    """Compute log probability of y_true under the mixture model.

    Uses the log-sum-exp trick for numerical stability:
        log p(y) = logsumexp_k( log_pi_k + sum_d(-0.5*log(2*pi) - log(sigma) - 0.5*((y-mu)/sigma)^2) )
    """
    # Reshape y_true: (batch, output_dim)
    y_true = ops.reshape(y_true, (-1, output_dim))

    # Reshape mu and sigma to (batch, num_mixes, output_dim)
    mu = ops.reshape(mu, (-1, num_mixes, output_dim))
    sigma = ops.reshape(sigma, (-1, num_mixes, output_dim))

    # Expand y_true to (batch, 1, output_dim) for broadcasting
    y_true = ops.expand_dims(y_true, axis=1)

    # Log probability of each component (diagonal covariance normal)
    # log N(y | mu, sigma^2) = -0.5*D*log(2*pi) - sum(log(sigma)) - 0.5*sum(((y-mu)/sigma)^2)
    log_component = (
        -0.5 * output_dim * math.log(2.0 * math.pi)
        - ops.sum(ops.log(sigma), axis=-1)
        - 0.5 * ops.sum(ops.square((y_true - mu) / sigma), axis=-1)
    )  # shape: (batch, num_mixes)

    # Log mixture weights via log_softmax (numerically stable)
    log_pi = ops.log_softmax(pi_logits, axis=-1)  # shape: (batch, num_mixes)

    # Log probability of mixture: logsumexp over components
    log_prob = ops.logsumexp(log_pi + log_component, axis=-1)  # shape: (batch,)
    return log_prob


def get_mixture_loss_func(output_dim, num_mixes):
    """Construct a loss function for the MDN layer parametrised by number of mixtures."""
    def mdn_loss_func(y_true, y_pred):
        mu, sigma, pi_logits = _split_mdn_params(y_pred, output_dim, num_mixes)
        log_prob = _mdn_log_prob(y_true, mu, sigma, pi_logits, output_dim, num_mixes)
        return ops.mean(-log_prob)

    return mdn_loss_func


def get_mixture_sampling_fun(output_dim, num_mixes):
    """Construct a sampling operation for the MDN layer parametrised
    by mixtures and output dimension. This can be used in a Keras model to
    generate samples directly."""

    def sampling_func(y_pred):
        mu, sigma, pi_logits = _split_mdn_params(y_pred, output_dim, num_mixes)
        batch_size = ops.shape(mu)[0]

        # Sample mixture component indices from categorical distribution
        # keras.random.categorical expects (batch, num_categories) logits
        # and returns (batch, num_samples) indices
        component_indices = keras.random.categorical(pi_logits, 1)  # (batch, 1)
        component_indices = ops.cast(ops.squeeze(component_indices, axis=-1), "int32")  # (batch,)

        # Reshape mu and sigma to (batch, num_mixes, output_dim)
        mu = ops.reshape(mu, (-1, num_mixes, output_dim))
        sigma = ops.reshape(sigma, (-1, num_mixes, output_dim))

        # Gather the selected component's mu and sigma using ops.take_along_axis
        # Expand indices to (batch, 1, output_dim) for gathering
        idx = ops.reshape(component_indices, (-1, 1, 1))
        idx = ops.broadcast_to(idx, (batch_size, 1, output_dim))
        selected_mu = ops.squeeze(ops.take_along_axis(mu, idx, axis=1), axis=1)  # (batch, output_dim)
        selected_sigma = ops.squeeze(ops.take_along_axis(sigma, idx, axis=1), axis=1)  # (batch, output_dim)

        # Sample from the selected normal: mu + sigma * N(0, 1)
        noise = keras.random.normal(ops.shape(selected_mu))
        return selected_mu + selected_sigma * noise

    return sampling_func


def get_mixture_mse_accuracy(output_dim, num_mixes):
    """Construct an MSE accuracy function for the MDN layer
    that takes one sample and compares to the true value."""
    sampling_func = get_mixture_sampling_fun(output_dim, num_mixes)

    def mse_func(y_true, y_pred):
        y_true = ops.reshape(y_true, (-1, output_dim))
        samp = sampling_func(y_pred)
        return ops.mean(ops.square(samp - y_true), axis=-1)

    return mse_func


def split_mixture_params(params, output_dim, num_mixes):
    """Splits up an array of mixture parameters into mus, sigmas, and pis
    depending on the number of mixtures and output dimension.

    Arguments:
    params -- the parameters of the mixture model
    output_dim -- the dimension of the normal models in the mixture model
    num_mixes -- the number of mixtures represented
    """
    assert len(params) == num_mixes + (output_dim * 2 * num_mixes), "The size of params needs to match the mixture configuration"
    mus = params[:num_mixes * output_dim]
    sigs = params[num_mixes * output_dim:2 * num_mixes * output_dim]
    pi_logits = params[-num_mixes:]
    return mus, sigs, pi_logits


def softmax(w, t=1.0):
    """Softmax function for a list or numpy array of logits. Also adjusts temperature.

    Arguments:
    w -- a list or numpy array of logits

    Keyword arguments:
    t -- the temperature for to adjust the distribution (default 1.0)
    """
    e = np.array(w) / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist


def sample_from_categorical(dist):
    """Samples from a categorical model PDF.

    Arguments:
    dist -- the parameters of the categorical model

    Returns:
    One sample from the categorical model, or -1 if sampling fails.
    """
    r = np.random.rand(1)  # uniform random number in [0,1]
    accumulate = 0
    for i in range(0, dist.size):
        accumulate += dist[i]
        if accumulate >= r:
            return i
    return -1


def sample_from_output(params, output_dim, num_mixes, temp=1.0, sigma_temp=1.0):
    """Sample from an MDN output with temperature adjustment.
    This calculation is done outside of the Keras model using
    Numpy.

    Arguments:
    params -- the parameters of the mixture model
    output_dim -- the dimension of the normal models in the mixture model
    num_mixes -- the number of mixtures represented

    Keyword arguments:
    temp -- the temperature for sampling between mixture components (default 1.0)
    sigma_temp -- the temperature for sampling from the normal distribution (default 1.0)

    Returns:
    One sample from the the mixture model, that is a numpy array of length output_dim
    """
    assert len(params) == num_mixes + (output_dim * 2 * num_mixes), "The size of params needs to match the mixture configuration"
    mus, sigs, pi_logits = split_mixture_params(params, output_dim, num_mixes)
    pis = softmax(pi_logits, t=temp)
    m = sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m * output_dim:(m + 1) * output_dim]
    sig_vector = sigs[m * output_dim:(m + 1) * output_dim]
    scale_matrix = np.identity(output_dim) * sig_vector  # scale matrix from diag
    cov_matrix = np.matmul(scale_matrix, scale_matrix.T)  # cov is scale squared.
    cov_matrix = cov_matrix * sigma_temp  # adjust for sigma temperature
    sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
    return sample[0]
