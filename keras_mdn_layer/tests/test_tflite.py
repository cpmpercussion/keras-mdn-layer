"""Tests for TFLite conversion and LiteRT inference of MDRNN models."""

import keras
import keras_mdn_layer as mdn
import numpy as np
import pytest


def build_mdrnn_model(
    dimension: int,
    n_hidden_units: int,
    n_mixtures: int,
    n_layers: int,
    inference: bool,
    seq_length: int = 30,
):
    """Builds a Keras MDRNN model for training or inference."""
    if inference:
        state_input_output = True
        sequence_length = 1
        time_dist = False
    else:
        state_input_output = False
        sequence_length = seq_length
        time_dist = True

    data_input = keras.layers.Input(
        shape=(sequence_length, dimension), name="inputs"
    )
    lstm_in = data_input
    state_inputs = []
    state_outputs = []

    for layer_i in range(n_layers):
        return_sequences = True
        if (layer_i == n_layers - 1) and not time_dist:
            return_sequences = False
        state_input = None
        if state_input_output:
            state_h_input = keras.layers.Input(
                shape=(n_hidden_units,), name=f"state_h_{layer_i}"
            )
            state_c_input = keras.layers.Input(
                shape=(n_hidden_units,), name=f"state_c_{layer_i}"
            )
            state_input = [state_h_input, state_c_input]
            state_inputs += state_input
        lstm_out, state_h_output, state_c_output = keras.layers.LSTM(
            n_hidden_units,
            name=f"lstm_{layer_i}",
            return_sequences=return_sequences,
            return_state=True,
        )(lstm_in, initial_state=state_input)
        lstm_in = lstm_out
        state_outputs += [state_h_output, state_c_output]

    mdn_layer = mdn.MDN(dimension, n_mixtures, name="mdn_outputs")
    if time_dist:
        mdn_layer = keras.layers.TimeDistributed(mdn_layer, name="td_mdn")
    mdn_out = mdn_layer(lstm_out)

    if inference:
        inputs = [data_input] + state_inputs
        outputs = [mdn_out] + state_outputs
    else:
        inputs = data_input
        outputs = mdn_out

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    if not inference:
        loss_func = mdn.get_mixture_loss_func(dimension, n_mixtures)
        optimizer = keras.optimizers.Adam()
        model.compile(loss=loss_func, optimizer=optimizer)

    return model


def generate_sequential_data(n_samples, seq_length, dimension):
    """Generate simple synthetic sequential data."""
    X = np.zeros((n_samples, seq_length, dimension))
    y = np.zeros((n_samples, seq_length, dimension))
    for i in range(n_samples):
        phase = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 2 * np.pi, seq_length + 1)
        for d in range(dimension):
            signal = np.sin(t + phase + d * 0.5) + np.random.normal(
                0, 0.1, seq_length + 1
            )
            X[i, :, d] = signal[:-1]
            y[i, :, d] = signal[1:]
    return X.astype(np.float32), y.astype(np.float32)


@pytest.fixture
def trained_mdrnn():
    """Build and train a small MDRNN model for testing."""
    dimension = 2
    n_hidden_units = 16
    n_mixtures = 3
    n_layers = 1
    seq_length = 8

    model = build_mdrnn_model(
        dimension, n_hidden_units, n_mixtures, n_layers=n_layers,
        inference=False, seq_length=seq_length,
    )
    X, y = generate_sequential_data(50, seq_length, dimension)
    model.fit(X, y, epochs=2, batch_size=16, verbose=0)
    return model, dimension, n_mixtures, seq_length


@pytest.fixture
def tflite_model(trained_mdrnn):
    """Convert trained model to TFLite format using a concrete function."""
    import tensorflow as tf

    model, dimension, _, seq_length = trained_mdrnn

    # Trace with a fixed input shape (batch_size=1) for TFLite compatibility
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, seq_length, dimension], dtype=tf.float32)
    ])
    def serve(inputs):
        return model(inputs)

    concrete_func = serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    return converter.convert()


def test_tflite_conversion(tflite_model):
    """Test that an MDRNN model can be converted to TFLite format."""
    assert tflite_model is not None
    assert len(tflite_model) > 0


def test_litert_inference(trained_mdrnn, tflite_model):
    """Test that LiteRT can load a TFLite MDRNN model and run inference."""
    from ai_edge_litert.interpreter import Interpreter

    model, dimension, n_mixtures, seq_length = trained_mdrnn

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check input/output shapes
    assert input_details[0]["shape"].tolist() == [1, seq_length, dimension]
    n_mdn_params = 2 * dimension * n_mixtures + n_mixtures
    assert output_details[0]["shape"].tolist() == [1, seq_length, n_mdn_params]

    # Run inference
    test_input = np.random.randn(1, seq_length, dimension).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    assert output.shape == (1, seq_length, n_mdn_params)
    assert np.all(np.isfinite(output))


def test_litert_matches_keras(trained_mdrnn, tflite_model):
    """Test that LiteRT output matches Keras model output."""
    from ai_edge_litert.interpreter import Interpreter

    model, dimension, n_mixtures, seq_length = trained_mdrnn

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_input = np.random.randn(1, seq_length, dimension).astype(np.float32)

    # Keras output
    keras_output = model.predict(test_input, verbose=0)

    # LiteRT output
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    litert_output = interpreter.get_tensor(output_details[0]["index"])

    np.testing.assert_allclose(keras_output, litert_output, atol=1e-4)


def test_litert_output_is_samplable(trained_mdrnn, tflite_model):
    """Test that we can sample from the LiteRT MDN output using numpy utilities."""
    from ai_edge_litert.interpreter import Interpreter

    model, dimension, n_mixtures, seq_length = trained_mdrnn

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_input = np.random.randn(1, seq_length, dimension).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    # Sample from each timestep
    for t in range(seq_length):
        params = output[0, t, :]
        sample = mdn.sample_from_output(params, dimension, n_mixtures, temp=1.0)
        assert sample.shape == (dimension,)
        assert np.all(np.isfinite(sample))


def test_tflite_save_and_load(trained_mdrnn, tflite_model, tmp_path):
    """Test saving TFLite model to disk and loading it back."""
    from ai_edge_litert.interpreter import Interpreter

    model, dimension, n_mixtures, seq_length = trained_mdrnn

    # Save to file
    tflite_path = tmp_path / "mdrnn.tflite"
    tflite_path.write_bytes(tflite_model)
    assert tflite_path.exists()
    assert tflite_path.stat().st_size > 0

    # Load from file
    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_input = np.random.randn(1, seq_length, dimension).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    assert output.shape == (1, seq_length, 2 * dimension * n_mixtures + n_mixtures)
    assert np.all(np.isfinite(output))
