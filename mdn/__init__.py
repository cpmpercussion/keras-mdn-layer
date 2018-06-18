import keras
from keras import backend as K
from keras.layers import Dense, Input, merge
from keras.engine.topology import Layer
import numpy as np
from tensorflow.contrib.distributions import Categorical, Mixture, MultivariateNormalDiag
import tensorflow as tf

def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent NaN in loss."""
    return (K.elu(x) + 1 + 1e-8)
    

class MDN(Layer):
    """A Mixture Density Network Layer for Keras.
    This layer has a few tricks to avoid NaNs in the loss function when training:
        - Activation for variances is ELU + 1 + 1e-8 (to avoid very small values)
        - Mixture weights (pi) are trained in as logits, not in the softmax space.
        
    A loss function needs to be constructed with the same output dimension and number of mixtures.
    A sampling function is also provided to sample from distribution parametrised by the MDN outputs.
    """
    
    def __init__(self, output_dim, num_mix, **kwargs):
        self.output_dim = output_dim
        self.num_mix = num_mix
        with tf.name_scope('MDN'):
            self.mdn_mus     = Dense(self.num_mix * self.output_dim, name='mdn_mus') # mix*output vals, no activation
            self.mdn_sigmas  = Dense(self.num_mix * self.output_dim, activation=elu_plus_one_plus_epsilon, name='mdn_sigmas') # mix*output vals exp activation
            self.mdn_pi      = Dense(self.num_mix, name='mdn_pi') # mix vals, logits
        super(MDN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mdn_mus.build(input_shape)
        self.mdn_sigmas.build(input_shape)
        self.mdn_pi.build(input_shape)
        self.trainable_weights = self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights
        self.non_trainable_weights = self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights
        super(MDN, self).build(input_shape)
        
    def call(self, x, mask=None):
        with tf.name_scope('MDN'):
            mdn_out = keras.layers.concatenate([self.mdn_mus(x), 
                                                self.mdn_sigmas(x), 
                                                self.mdn_pi(x)], 
                                               name='mdn_outputs')
        return mdn_out
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



def get_mixture_loss_func(output_dim, num_mixes):
    """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
    
    # Construct a loss function with the right number of mixtures and outputs
    def loss_func(y_true, y_pred):
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim, 
                                                                         num_mixes * output_dim, 
                                                                         num_mixes], 
                                             axis=1, name='mdn_coef_split')
        cat = Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
            in zip(mus, sigs)]
        mixture = Mixture(cat=cat, components=coll)
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss
    
    # Actually return the loss_func
    with tf.name_scope('MDN'):
        return loss_func
    
def get_mixture_sampling_fun(output_dim, num_mixes):
    """Construct a sampling function for the MDN layer parametrised by mixtures and output dimension."""
        
    # Construct a loss function with the right number of mixtures and outputs
    def sampling_func(y_pred):
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim, 
                                                                         num_mixes * output_dim, 
                                                                         num_mixes], 
                                             axis=1, name='mdn_coef_split')
        cat = Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
            in zip(mus, sigs)]
        mixture = Mixture(cat=cat, components=coll)
        samp = mixture.sample()
        # Todo: temperature adjustment for sampling function.
        return samp
    
    # Actually return the loss_func
    with tf.name_scope('MDNLayer'):
        return sampling_func
    
def get_mixture_mse_accuracy(output_dim, num_mixes):
    """Construct an MSE accuracy function for the MDN layer 
    that takes one sample and compares to the true value."""
    
    # Construct a loss function with the right number of mixtures and outputs
    def mse_func(y_true, y_pred):
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim, 
                                                                         num_mixes * output_dim, 
                                                                         num_mixes], 
                                             axis=1, name='mdn_coef_split')
        cat = Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
            in zip(mus, sigs)]
        mixture = Mixture(cat=cat, components=coll)
        samp = mixture.sample()
        mse = tf.reduce_mean(tf.square(samp - y_true), axis=-1)
        # Todo: temperature adjustment for sampling functon.
        return mse
    
    # Actually return the loss_func
    with tf.name_scope('MDNLayer'):
        return mse_func


# # Sample from Categorical distribution
# def sample_mixture(mus, sigs, pis):
#     """Sample from a mixture of 1D normal distributions parameterised by mus, sigma, and pi."""
#     m = np.random.choice(range(len(pis)), p=pis)
#     return(np.random.normal(mus[m],sigs[m],1))

def split_mixture_params(params, mixtures, dim):
    """Splits up an array of mixture parameters into mus, sigmas, and pis 
    depending on the number of mixtures and output dimension."""
    mus = params[:mixtures*dim]
    sigs = params[mixtures*dim:2*mixtures*dim]
    pis = params[2*mixtures*dim:]
    return mus,sigs,pis

def adjust_temp(pi_pdf, temp):
    """ Adjusts temperature of a PDF describing a categorical model """
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf

def get_pi_idx(x, pdf, temp=1.0, greedy=False):
    """Samples from a categorical model PDF, optionally greedily."""
    if greedy:
        return np.argmax(pdf)
    pdf = adjust_temp(np.copy(pdf), temp)
    accumulate = 0
    for i in range(0, pdf.size):
        accumulate += pdf[i]
        if accumulate >= x:
            return i
    tf.logging.info('Error sampling mixture model.')
    return -1

def sample_from_categorical(dist, temp):
    """Sample from a categorical model with temperature adjustment."""
    r = np.random.rand(1)
    return get_pi_idx(r,dist,temp)

def softmax(w, t=1.0):
    """Softmax function for a list or numpy array of logits."""
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

# def sample_from_output_1D(params, mixtures, dim, temp=1.0):
#     """Sample from a 1D MDN output with temperature adjustment."""
#     mus = params[:mixtures*dim]
#     sigs = params[mixtures*dim:2*mixtures*dim]
#     pis = params[-mixtures:]
#     m = sample_from_categorical(pis, temp=temp)
#     return(np.random.normal(mus[m],sigs[m],1))

def sample_from_output(params, mixtures, dim, temp=1.0):
    """Sample from an MDN output with temperature adjustment."""
    mus = params[:mixtures*dim]
    sigs = params[mixtures*dim:2*mixtures*dim]
    pis = softmax(params[-mixtures:], t=temp)
    m = sample_from_categorical(pis, temp=temp)
    mus_vector = mus[m*dim:(m+1)*dim]
    sig_vector = sigs[m*dim:(m+1)*dim]
    cov_matrix = np.identity(dim) * sig_vector
    sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
    return sample