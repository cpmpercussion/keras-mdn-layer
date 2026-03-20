"""
Example: Build an MDRNN model, train it, convert to TFLite, and run inference with LiteRT.

This demonstrates the full workflow:
1. Build a training MDRNN model using build_mdrnn_model
2. Train on synthetic sequential data
3. Convert to TFLite format using tf.lite.TFLiteConverter
4. Load the TFLite model with LiteRT (ai-edge-litert)
5. Run inference and compare outputs
"""

import time
import numpy as np
import keras
import keras_mdn_layer as mdn


# -- Model builder (same as in tests/test_model_builder_def.py) --

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


# -- Synthetic data generation --

def generate_sequential_data(n_samples, seq_length, dimension):
    """Generate simple synthetic sequential data (sine waves with noise)."""
    X = np.zeros((n_samples, seq_length, dimension))
    y = np.zeros((n_samples, seq_length, dimension))
    for i in range(n_samples):
        phase = np.random.uniform(0, 2 * np.pi)
        freq = np.random.uniform(0.5, 2.0)
        t = np.linspace(0, 2 * np.pi, seq_length + 1)
        for d in range(dimension):
            signal = np.sin(freq * t + phase + d * 0.5) + np.random.normal(
                0, 0.1, seq_length + 1
            )
            X[i, :, d] = signal[:-1]
            y[i, :, d] = signal[1:]
    return X.astype(np.float32), y.astype(np.float32)


# -- TFLite conversion and LiteRT inference --

def convert_to_tflite(model, seq_length, dimension):
    """Convert a Keras model to TFLite format using a concrete function.

    LSTMs require fixed input shapes for TFLite conversion, so we trace
    the model with a concrete input signature (batch_size=1).
    """
    import tensorflow as tf

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, seq_length, dimension], dtype=tf.float32)
    ])
    def serve(inputs):
        return model(inputs)

    concrete_func = serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
    return tflite_model


def run_litert_inference(tflite_model, input_data):
    """Run inference using LiteRT (ai-edge-litert) interpreter."""
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")

    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Get output tensor
    output = interpreter.get_tensor(output_details[0]["index"])
    return output


def main():
    # Model parameters
    dimension = 8
    n_hidden_units = 64
    n_mixtures = 5
    n_layers = 2
    seq_length = 30

    print("=== Building MDRNN training model ===")
    train_model = build_mdrnn_model(
        dimension, n_hidden_units, n_mixtures, n_layers=n_layers,
        inference=False, seq_length=seq_length,
    )
    train_model.summary()

    print("\n=== Generating synthetic data ===")
    X_train, y_train = generate_sequential_data(200, seq_length, dimension)
    print(f"X shape: {X_train.shape}, y shape: {y_train.shape}")

    print("\n=== Training model ===")
    train_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

    print("\n=== Converting to TFLite ===")
    tflite_model = convert_to_tflite(train_model, seq_length, dimension)
    print(f"TFLite model size: {len(tflite_model)} bytes")

    # Save to file
    tflite_path = "mdrnn_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved to {tflite_path}")

    print("\n=== Running inference with LiteRT ===")
    test_input = X_train[:1]  # single sample
    litert_output = run_litert_inference(tflite_model, test_input)
    print(f"LiteRT output shape: {litert_output.shape}")

    # Compare with Keras model output
    keras_output = train_model.predict(test_input)
    print(f"Keras output shape: {keras_output.shape}")

    max_diff = np.max(np.abs(keras_output - litert_output))
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Outputs match (within tolerance): {max_diff < 1e-4}")

    # Demonstrate sampling from the LiteRT output
    print("\n=== Sampling from LiteRT output ===")
    n_mdn_params = 2 * dimension * n_mixtures + n_mixtures
    for t in range(min(3, seq_length)):
        params = litert_output[0, t, :]
        assert len(params) == n_mdn_params
        sample = mdn.sample_from_output(params, dimension, n_mixtures, temp=0.5)
        print(f"  Step {t}: sampled {sample}")

    # -- Benchmarking --
    n_runs = 100
    print(f"\n=== Benchmarking ({n_runs} runs) ===")

    # Benchmark LiteRT inference
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Warmup
    for _ in range(5):
        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()

    litert_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()
        interpreter.get_tensor(output_details[0]["index"])
        litert_times.append(time.perf_counter() - t0)

    litert_times = np.array(litert_times) * 1000  # ms
    print(f"LiteRT:  mean={litert_times.mean():.3f}ms  "
          f"std={litert_times.std():.3f}ms  "
          f"min={litert_times.min():.3f}ms  "
          f"max={litert_times.max():.3f}ms")

    # Benchmark Keras model.predict
    # Warmup
    for _ in range(5):
        train_model.predict(test_input, verbose=0)

    keras_predict_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        train_model.predict(test_input, verbose=0)
        keras_predict_times.append(time.perf_counter() - t0)

    keras_predict_times = np.array(keras_predict_times) * 1000
    print(f"Keras predict:  mean={keras_predict_times.mean():.3f}ms  "
          f"std={keras_predict_times.std():.3f}ms  "
          f"min={keras_predict_times.min():.3f}ms  "
          f"max={keras_predict_times.max():.3f}ms")

    # Benchmark Keras model.__call__ (no overhead of predict())
    # Warmup
    for _ in range(5):
        train_model(test_input)

    keras_call_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        train_model(test_input)
        keras_call_times.append(time.perf_counter() - t0)

    keras_call_times = np.array(keras_call_times) * 1000
    print(f"Keras __call__: mean={keras_call_times.mean():.3f}ms  "
          f"std={keras_call_times.std():.3f}ms  "
          f"min={keras_call_times.min():.3f}ms  "
          f"max={keras_call_times.max():.3f}ms")

    print(f"\nSpeedup vs predict: {keras_predict_times.mean() / litert_times.mean():.1f}x")
    print(f"Speedup vs __call__: {keras_call_times.mean() / litert_times.mean():.1f}x")


if __name__ == "__main__":
    main()
