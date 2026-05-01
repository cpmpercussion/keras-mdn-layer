# Keras Mixture Density Network Layer

[![Coverage Status](https://coveralls.io/repos/github/cpmpercussion/keras-mdn-layer/badge.svg?branch=master)](https://coveralls.io/github/cpmpercussion/keras-mdn-layer?branch=master)
[![Build and test keras-mdn-layer](https://github.com/cpmpercussion/keras-mdn-layer/actions/workflows/python-app.yml/badge.svg)](https://github.com/cpmpercussion/keras-mdn-layer/actions/workflows/python-app.yml)
![MIT License](https://img.shields.io/github/license/cpmpercussion/keras-mdn-layer.svg?style=flat)
[![DOI](https://zenodo.org/badge/137585470.svg)](https://zenodo.org/badge/latestdoi/137585470)
[![PyPI version](https://badge.fury.io/py/keras-mdn-layer.svg)](https://badge.fury.io/py/keras-mdn-layer)

A mixture density network (MDN) Layer for Keras using pure Keras 3 operations. This makes it a bit more simple to experiment with neural networks that predict multiple real-valued variables that can take on multiple equally likely values.

This layer can help build MDN-RNNs similar to those used in [RoboJam](https://github.com/cpmpercussion/robojam), [Sketch-RNN](https://experiments.withgoogle.com/sketch-rnn-demo), [handwriting generation](https://distill.pub/2016/handwriting/), and maybe even [world models](https://worldmodels.github.io). You can do a lot of cool stuff with MDNs!

One benefit of this implementation is that you can predict any number of real-values. The loss function implements the negative log-likelihood of a mixture of multivariate normal distributions with a diagonal covariance matrix, derived directly from the standard multivariate Gaussian (Bishop PRML Eq. 2.43). In previous work, the loss function has often been specified by hand which is fine for 1D or 2D prediction, but becomes a bit more annoying after that.

Two important functions are provided for training and prediction:

- `get_mixture_loss_func(output_dim, num_mixtures)`: This function generates a loss function with the correct output dimensions and number of mixtures.
- `sample_from_output(params, output_dim, num_mixtures, temp=1.0, sigma_temp=1.0)`: This function samples from the mixture distribution output by the model.

## Installation

This project requires Python 3.11+ and Keras 3 with a backend installed (TensorFlow, JAX, or PyTorch). You can install this package from [PyPI](https://pypi.org/project/keras-mdn-layer/) via `pip` like so:

    python3 -m pip install keras-mdn-layer

You will also need a Keras 3 backend. For example:

    python3 -m pip install tensorflow   # or jax/jaxlib, or torch

And finally, import the module in Python: `import keras_mdn_layer as mdn`

Alternatively, you can clone or download this repository and then install via `poetry install`.

## Tested Configurations

This library is tested on CI across the following matrix:

| Python  | Platforms                      |
|---------|--------------------------------|
| 3.11    | Ubuntu, macOS, Windows         |
| 3.12    | Ubuntu, macOS, Windows         |
| 3.13    | Ubuntu, macOS, Windows         |

CI uses TensorFlow as the Keras backend. Other Keras 3 backends (JAX, PyTorch) should also work but are not regularly tested.

## Testing

The test suite (54 tests) covers:

- **Layer mechanics**: construction, serialization (`get_config`), output shapes, weight counts, model save/load
- **Mathematical correctness**: analytical NLL verification against single and multi-component Gaussians (Bishop PRML Eq. 2.43), softmax identities, batch loss averaging
- **Sampling**: statistical tests for mean/std recovery, bimodal mixture separation, shape and finiteness checks
- **Training convergence**: loss reduction, parameter recovery for known distributions (N(0,1), N(5, 0.25))
- **Numerical stability**: extreme sigma values, extreme logits, large distances from mean
- **End-to-end pipelines**: predict/split/sample roundtrip, LSTM-MDN (MDRNN) models in both training and inference configurations, TimeDistributed MDN

## Build

This project builds using `poetry`. To build a wheel use `poetry build`.

## Examples

Some examples are provided in the notebooks directory.

To run these using `poetry`, run `poetry install` and then open jupyter `poetry run jupyter lab`.

There's scripts for fitting multivalued functions, a standard MDN toy problem:

<img src="https://preview.ibb.co/mZzkpd/Keras_MDN_Demo.jpg" alt="Keras MDN Demo" border="0">

There's also a script for generating fake kanji characters:

<img src="https://i.ibb.co/yFvtgkL/kanji-mdn-examples.png" alt="kanji test 1" border="0" width="600"/>

And finally, for learning how to generate musical touch-screen performances with a temporal component:

<img src="https://i.ibb.co/WpzSCV8/robojam-examples.png" alt="Robojam Model Examples" border="0">

## How to use

The MDN layer should be the last in your network and you should use `get_mixture_loss_func` to generate a loss function. Here's an example of a simple network with one Dense layer followed by the MDN.

    import keras
    import keras_mdn_layer as mdn

    N_HIDDEN = 15  # number of hidden units in the Dense layer
    N_MIXES = 10  # number of mixture components
    OUTPUT_DIMS = 2  # number of real-values predicted by each mixture component

    model = keras.Sequential()
    model.add(keras.layers.Dense(N_HIDDEN, batch_input_shape=(None, 1), activation='relu'))
    model.add(mdn.MDN(OUTPUT_DIMS, N_MIXES))
    model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMS,N_MIXES), optimizer=keras.optimizers.Adam())
    model.summary()

Fit as normal:

    history = model.fit(x=x_train, y=y_train)

The predictions from the network are parameters of the mixture models, so you have to apply the `sample_from_output` function to generate samples.

    y_test = model.predict(x_test)
    y_samples = np.apply_along_axis(mdn.sample_from_output, 1, y_test, OUTPUT_DIMS, N_MIXES, temp=1.0)

See the notebooks directory for examples in jupyter notebooks!

### Load/Save Model

Saving models is straight forward:

    model.save('test_save.keras')

But loading requires `custom_objects` to be filled with the MDN layer, and a loss function with the appropriate parameters:

    m_2 = keras.models.load_model('test_save.keras', custom_objects={'MDN': mdn.MDN, 'mdn_loss_func': mdn.get_mixture_loss_func(1, N_MIXES)})

## TFLite / LiteRT Export

MDRNN models built with this layer can be exported to TFLite format for fast on-device inference using [LiteRT](https://ai.google.dev/edge/litert) (formerly TensorFlow Lite). An example script is provided in `examples/tflite_mdrnn.py`.

Because LSTMs require static shapes for TFLite conversion, wrap the model call in a `tf.function` with a fixed input signature before converting:

    import tensorflow as tf

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, seq_length, dimension], dtype=tf.float32)
    ])
    def serve(inputs):
        return model(inputs)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serve.get_concrete_function()]
    )
    tflite_model = converter.convert()

Then load and run inference with LiteRT (`pip install ai-edge-litert`):

    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

In a benchmark of a 2-layer LSTM MDRNN (dim=8, 64 hidden units, 5 mixtures, seq_length=30), LiteRT inference was **~46x faster** than `model.predict()` and **~180x faster** than `model()` direct call:

| Method          | Mean latency |
|-----------------|-------------|
| LiteRT          | 0.4 ms      |
| Keras `predict` | 19 ms       |
| Keras `__call__` | 75 ms      |

The TFLite output matches Keras output to within ~1e-6, so there is no accuracy loss. The full example with training, conversion, inference, and benchmarking is in [`examples/tflite_mdrnn.py`](examples/tflite_mdrnn.py).

## Acknowledgements

- Hat tip to [Omimo's Keras MDN layer](https://github.com/omimo/Keras-MDN) for a starting point for this code.
- Super hat tip to [hardmaru's MDN explanation, projects, and good ideas for sampling functions](http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/) etc.
- Many good ideas from [Axel Brando's Master's Thesis](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation)
- Mixture Density Networks in Edward [tutorial](http://edwardlib.org/tutorials/mixture-density-network).

## References

1. Christopher M. Bishop. 1994. Mixture Density Networks. [Technical Report NCRG/94/004](http://publications.aston.ac.uk/373/). Neural Computing Research Group, Aston University. http://publications.aston.ac.uk/373/
2. Axel Brando. 2017. Mixture Density Networks (MDN) for distribution and uncertainty estimation. Master's thesis. Universitat Politecnica de Catalunya.
3. A. Graves. 2013. Generating Sequences With Recurrent Neural Networks. ArXiv e-prints (Aug. 2013). https://arxiv.org/abs/1308.0850
4. David Ha and Douglas Eck. 2017. A Neural Representation of Sketch Drawings. ArXiv e-prints (April 2017). https://arxiv.org/abs/1704.03477
5. Charles P. Martin and Jim Torresen. 2018. RoboJam: A Musical Mixture Density Network for Collaborative Touchscreen Interaction. In Evolutionary and Biologically Inspired Music, Sound, Art and Design: EvoMUSART '18, A. Liapis et al. (Ed.). Lecture Notes in Computer Science, Vol. 10783. Springer International Publishing. DOI:[10.1007/9778-3-319-77583-8_11](http://dx.doi.org/10.1007/9778-3-319-77583-8_11)
