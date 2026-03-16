from tensorflow import keras
import tensorflow as tf
import keras_mdn_layer as mdn
import numpy as np
import click


def test_elu_plus_one_plus_epsilon():
    """Test that the custom activation is always positive."""
    x = tf.constant([-10.0, -1.0, 0.0, 1.0, 10.0])
    result = mdn.elu_plus_one_plus_epsilon(x).numpy()
    assert np.all(result > 0), "Activation should always be positive"
    # ELU(0) + 1 + eps should be ~1.0
    assert np.isclose(result[2], 1.0, atol=1e-6)


def test_mdn_get_config():
    """Test that MDN get_config returns correct parameters for serialization."""
    layer = mdn.MDN(3, 5)
    config = layer.get_config()
    assert config["output_dimension"] == 3
    assert config["num_mixtures"] == 5
    assert "name" in config


def test_mdn_compute_output_shape():
    """Test compute_output_shape returns the right dimensions."""
    layer = mdn.MDN(3, 5)
    shape = layer.compute_output_shape((None, 10))
    # expected: (None, 2 * 3 * 5 + 5) = (None, 35)
    assert shape == (None, 35)

    layer2 = mdn.MDN(1, 10)
    shape2 = layer2.compute_output_shape((None, 8))
    # expected: (None, 2 * 1 * 10 + 10) = (None, 30)
    assert shape2 == (None, 30)


def test_split_mixture_params():
    """Test splitting concatenated MDN parameters."""
    output_dim = 2
    num_mixes = 3
    # total params: num_mixes * output_dim (mus) + num_mixes * output_dim (sigmas) + num_mixes (pis)
    # = 6 + 6 + 3 = 15
    params = np.arange(15, dtype=np.float64)
    mus, sigs, pi_logits = mdn.split_mixture_params(params, output_dim, num_mixes)
    assert len(mus) == num_mixes * output_dim
    assert len(sigs) == num_mixes * output_dim
    assert len(pi_logits) == num_mixes
    np.testing.assert_array_equal(mus, np.arange(6))
    np.testing.assert_array_equal(sigs, np.arange(6, 12))
    np.testing.assert_array_equal(pi_logits, np.arange(12, 15))


def test_split_mixture_params_invalid():
    """Test that split_mixture_params raises on wrong-sized input."""
    try:
        mdn.split_mixture_params(np.zeros(10), output_dim=2, num_mixes=3)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_softmax():
    """Test softmax produces valid probability distribution."""
    logits = [1.0, 2.0, 3.0]
    result = mdn.softmax(logits)
    assert np.isclose(result.sum(), 1.0)
    assert np.all(result > 0)
    # higher logit should have higher probability
    assert result[2] > result[1] > result[0]


def test_softmax_temperature():
    """Test that temperature affects softmax sharpness."""
    logits = [1.0, 2.0, 3.0]
    # Low temperature should make distribution sharper
    cold = mdn.softmax(logits, t=0.1)
    hot = mdn.softmax(logits, t=10.0)
    assert cold[2] > hot[2], "Low temp should concentrate on max"
    # Very low temperature should be near one-hot
    assert cold[2] > 0.99


def test_softmax_uniform():
    """Test softmax with equal logits gives uniform distribution."""
    logits = [5.0, 5.0, 5.0, 5.0]
    result = mdn.softmax(logits)
    np.testing.assert_allclose(result, 0.25, atol=1e-7)


def test_sample_from_categorical():
    """Test that sample_from_categorical returns valid indices."""
    np.random.seed(42)
    dist = np.array([0.1, 0.2, 0.7])
    samples = [mdn.sample_from_categorical(dist) for _ in range(100)]
    assert all(0 <= s <= 2 for s in samples)
    # With this distribution, index 2 should be most common
    counts = [samples.count(i) for i in range(3)]
    assert counts[2] > counts[0]


def test_sample_from_output_basic():
    """Test sample_from_output with known parameters."""
    np.random.seed(42)
    output_dim = 2
    num_mixes = 3
    # Build fake params: mus, sigmas, pi_logits
    mus = np.zeros(num_mixes * output_dim)
    sigmas = np.ones(num_mixes * output_dim)
    pi_logits = np.array([0.0, 0.0, 0.0])
    params = np.concatenate([mus, sigmas, pi_logits])

    sample = mdn.sample_from_output(params, output_dim, num_mixes)
    assert sample.shape == (output_dim,)


def test_sample_from_output_temperature():
    """Test that sigma_temp affects spread around a single mixture component."""
    np.random.seed(42)
    output_dim = 1
    num_mixes = 1
    mus = np.array([0.0])
    sigmas = np.array([1.0])
    pi_logits = np.array([0.0])
    params = np.concatenate([mus, sigmas, pi_logits])

    # Single mixture: std is purely from sigma_temp
    samples_cold = [mdn.sample_from_output(params, output_dim, num_mixes, sigma_temp=0.01)[0] for _ in range(200)]
    samples_hot = [mdn.sample_from_output(params, output_dim, num_mixes, sigma_temp=100.0)[0] for _ in range(200)]
    assert np.std(samples_cold) < np.std(samples_hot)


def test_sample_from_output_invalid():
    """Test that sample_from_output rejects wrong-sized params."""
    try:
        mdn.sample_from_output(np.zeros(5), output_dim=2, num_mixes=3)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_loss_function_runs():
    """Test that the MDN loss function computes a finite scalar loss."""
    output_dim = 2
    num_mixes = 3
    loss_func = mdn.get_mixture_loss_func(output_dim, num_mixes)

    # Build fake predictions: mus + sigmas + pi_logits
    batch_size = 4
    mus = tf.zeros((batch_size, num_mixes * output_dim))
    sigmas = tf.ones((batch_size, num_mixes * output_dim))
    pi_logits = tf.zeros((batch_size, num_mixes))
    y_pred = tf.concat([mus, sigmas, pi_logits], axis=-1)
    y_true = tf.zeros((batch_size, output_dim))

    loss = loss_func(y_true, y_pred)
    assert loss.numpy().shape == ()  # scalar
    assert np.isfinite(loss.numpy())


def test_loss_function_decreases_with_closer_predictions():
    """Test that loss is lower when predictions are closer to true values."""
    output_dim = 1
    num_mixes = 1
    loss_func = mdn.get_mixture_loss_func(output_dim, num_mixes)

    y_true = tf.constant([[5.0]])
    sigmas = tf.constant([[1.0]])
    pi_logits = tf.constant([[0.0]])

    # Prediction close to true
    y_pred_close = tf.concat([tf.constant([[5.0]]), sigmas, pi_logits], axis=-1)
    # Prediction far from true
    y_pred_far = tf.concat([tf.constant([[100.0]]), sigmas, pi_logits], axis=-1)

    loss_close = loss_func(y_true, y_pred_close).numpy()
    loss_far = loss_func(y_true, y_pred_far).numpy()
    assert loss_close < loss_far


def test_mixture_sampling_fun():
    """Test that get_mixture_sampling_fun produces samples of the right shape."""
    output_dim = 2
    num_mixes = 3
    sampling_func = mdn.get_mixture_sampling_fun(output_dim, num_mixes)

    batch_size = 4
    mus = tf.zeros((batch_size, num_mixes * output_dim))
    sigmas = tf.ones((batch_size, num_mixes * output_dim))
    pi_logits = tf.zeros((batch_size, num_mixes))
    y_pred = tf.concat([mus, sigmas, pi_logits], axis=-1)

    samples = sampling_func(y_pred)
    assert samples.shape == (batch_size, output_dim)
    assert np.all(np.isfinite(samples.numpy()))


def test_mixture_mse_accuracy():
    """Test that get_mixture_mse_accuracy returns non-negative MSE values."""
    output_dim = 2
    num_mixes = 3
    mse_func = mdn.get_mixture_mse_accuracy(output_dim, num_mixes)

    batch_size = 4
    mus = tf.zeros((batch_size, num_mixes * output_dim))
    sigmas = tf.ones((batch_size, num_mixes * output_dim)) * 0.001  # very small sigma
    pi_logits = tf.zeros((batch_size, num_mixes))
    y_pred = tf.concat([mus, sigmas, pi_logits], axis=-1)
    y_true = tf.zeros((batch_size, output_dim))

    mse = mse_func(y_true, y_pred)
    assert mse.shape == (batch_size,)
    assert np.all(np.isfinite(mse.numpy()))
    assert np.all(mse.numpy() >= 0)


def test_training_reduces_loss():
    """Test that training a simple MDN model actually reduces loss."""
    np.random.seed(42)
    tf.random.set_seed(42)
    N_MIXES = 3

    model = keras.Sequential()
    model.add(keras.layers.Dense(15, batch_input_shape=(None, 1), activation='relu'))
    model.add(mdn.MDN(1, N_MIXES))
    model.compile(loss=mdn.get_mixture_loss_func(1, N_MIXES), optimizer=keras.optimizers.Adam(learning_rate=0.01))

    # Simple data: y = x
    x_train = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)
    y_train = x_train.copy()

    history = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    assert history.history['loss'][-1] < history.history['loss'][0], "Loss should decrease during training"


def test_multidimensional_output():
    """Test MDN with multiple output dimensions."""
    N_MIXES = 3
    OUTPUT_DIM = 5
    model = keras.Sequential()
    model.add(keras.layers.Dense(10, batch_input_shape=(None, 2), activation='relu'))
    model.add(mdn.MDN(OUTPUT_DIM, N_MIXES))
    model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIM, N_MIXES), optimizer='adam')

    expected_output_size = 2 * OUTPUT_DIM * N_MIXES + N_MIXES  # 35
    assert model.output_shape == (None, expected_output_size)

    # Predict and sample
    x = np.random.randn(2, 2).astype(np.float32)
    y_pred = model.predict(x, verbose=0)
    assert y_pred.shape == (2, expected_output_size)

    sample = mdn.sample_from_output(y_pred[0], OUTPUT_DIM, N_MIXES)
    assert sample.shape == (OUTPUT_DIM,)


def test_build_mdn():
    """Make sure an MDN model can be constructed"""
    N_HIDDEN = 5
    N_MIXES = 5
    model = keras.Sequential()
    model.add(keras.layers.Dense(N_HIDDEN, batch_input_shape=(None, 1), activation='relu'))
    model.add(keras.layers.Dense(N_HIDDEN, activation='relu'))
    model.add(mdn.MDN(1, N_MIXES))
    model.compile(loss=mdn.get_mixture_loss_func(1, N_MIXES), optimizer=keras.optimizers.Adam())
    assert isinstance(model, keras.Sequential)


def test_number_of_weights():
    """Make sure the number of trainable weights is set up correctly"""
    N_HIDDEN = 5
    N_MIXES = 5
    inputs = keras.layers.Input(shape=(1,))
    x = keras.layers.Dense(N_HIDDEN, activation='relu')(inputs)
    m = mdn.MDN(1, N_MIXES)
    predictions = m(x)
    model = keras.Model(inputs=inputs, outputs=predictions)
    model.compile(loss=mdn.get_mixture_loss_func(1, N_MIXES), optimizer=keras.optimizers.Adam())
    num_mdn_params = np.sum([w.get_shape().num_elements() for w in m.trainable_weights])
    assert (num_mdn_params == 90)


def test_save_mdn():
    """Make sure an MDN model can be saved and loaded"""
    N_HIDDEN = 5
    N_MIXES = 5
    model = keras.Sequential()
    model.add(keras.layers.Dense(N_HIDDEN, batch_input_shape=(None, 1), activation='relu'))
    model.add(mdn.MDN(1, N_MIXES))
    model.compile(loss=mdn.get_mixture_loss_func(1, N_MIXES), optimizer=keras.optimizers.Adam())
    model.save('test_save.keras', overwrite=True)
    m_2 = keras.models.load_model('test_save.keras', custom_objects={'MDN': mdn.MDN, 'mdn_loss_func': mdn.get_mixture_loss_func(1, N_MIXES)})
    assert isinstance(m_2, keras.Sequential)

def test_output_shapes():
    """Checks that the output shapes on an MDN model end up correct. Builds a 1-layer LSTM network to do it."""
    # parameters
    N_HIDDEN = 5
    N_MIXES = 10
    N_DIMENSION = 7

    # set up a 1-layer MDRNN 
    inputs = keras.layers.Input(shape=(1,N_DIMENSION))
    lstm_1_state_h_input = keras.layers.Input(shape=(N_HIDDEN,))
    lstm_1_state_c_input = keras.layers.Input(shape=(N_HIDDEN,))
    lstm_1_state_input = [lstm_1_state_h_input, lstm_1_state_c_input]
    lstm_1, state_h_1, state_c_1 = keras.layers.LSTM(N_HIDDEN, return_state=True)(inputs, initial_state=lstm_1_state_input)
    lstm_1_state_output = [state_h_1, state_c_1]
    mdn_out = mdn.MDN(N_DIMENSION, N_MIXES)(lstm_1)
    decoder = keras.Model(inputs=[inputs] + lstm_1_state_input, outputs=[mdn_out] + lstm_1_state_output)

    # create starting input and generate one output
    starting_input = np.zeros((1, 1, N_DIMENSION), dtype=np.float32)
    initial_state = [np.zeros((1,N_HIDDEN), dtype=np.float32), np.zeros((1,N_HIDDEN), dtype=np.float32)]
    output_list = decoder([starting_input] + initial_state) # run the network
    mdn_parameters = output_list[0][0].numpy()

    # sample from the output to test sampling functions
    generated_sample = mdn.sample_from_output(mdn_parameters, N_DIMENSION, N_MIXES)
    print("Sample shape:", generated_sample.shape)
    print("Sample:", generated_sample)
    # test that the length of the generated sample is the same as N_DIMENSION
    assert len(generated_sample) == N_DIMENSION



def test_mdrnn_training_model():
    """Builds a time-distributed MDN model for sequence-to-sequence prediction."""
    sequence_length = 50
    n_hidden_units = 256
    dimension = 3
    n_mixtures = 10

    inputs = keras.layers.Input(shape=(sequence_length,dimension), name='inputs')
    lstm1_out, state_h_1_output, state_c_1_output = keras.layers.LSTM(n_hidden_units, name='lstm1', return_sequences=True, return_state=True)(inputs)
    lstm2_out, state_h_2_output, state_c_2_output = keras.layers.LSTM(n_hidden_units, name='lstm2', return_sequences=True, return_state=True)(lstm1_out)
    mdn_layer = mdn.MDN(dimension, n_mixtures, name='mdn_outputs')
    mdn_out = keras.layers.TimeDistributed(mdn_layer, name='td_mdn')(lstm2_out)

    model = keras.models.Model(inputs=inputs, outputs=mdn_out, name="time-distributed-mdrnn-training-mode")

    model.compile(loss=mdn.get_mixture_loss_func(dimension,n_mixtures), optimizer='adam')
    model.summary()
    assert isinstance(model, keras.models.Model)


def test_mdrnn_inference_model():
    sequence_length = 1
    n_hidden_units = 256
    dimension = 3
    n_mixtures = 10

    inputs = keras.layers.Input(shape=(sequence_length,dimension), name='inputs')
    lstm1_out, state_h_1_output, state_c_1_output = keras.layers.LSTM(n_hidden_units, name='lstm1', return_sequences=True, return_state=True)(inputs)
    lstm2_out, state_h_2_output, state_c_2_output = keras.layers.LSTM(n_hidden_units, name='lstm2', return_sequences=False, return_state=True)(lstm1_out)
    mdn_layer = mdn.MDN(dimension, n_mixtures, name='mdn_outputs')
    mdn_out = mdn_layer(lstm2_out)
    model = keras.models.Model(inputs=inputs, outputs=mdn_out, name="mdrnn-inference-mode")
    model.compile(loss=mdn.get_mixture_loss_func(dimension,n_mixtures), optimizer='adam')
    model.summary()
    assert isinstance(model, keras.models.Model)


def test_mdn_builder_function():
    """Tests the MDRNN model builder function for both training and inference models."""

    SCALE_FACTOR = 10

    def mdrnn_model_name(dimension: int, n_rnn_layers: int, n_hidden_units: int, n_mixtures: int) -> str:
        """Returns the name of a model using its parameters"""
        name = f"musicMDRNN"
        name += f"-dim{dimension}"
        name += f"-layers{n_rnn_layers}"
        name += f"-units{n_hidden_units}"
        name += f"-mixtures{n_mixtures}"
        name += f"-scale{SCALE_FACTOR}"
        return name

    def build_mdrnn_model(dimension: int, n_hidden_units: int, n_mixtures: int, n_layers: int, inference: bool, seq_length = 30):
        """Builds a Keras MDRNN model with specified parameters.
        Can either be a training model or inference model which affects the configured 
        sequence length and whether a loss function is added.
        """
        # Set parameters for inference/training versions.
        if inference:
            state_input_output = True
            sequence_length = 1
            time_dist = False
        else:
            state_input_output = False
            sequence_length = seq_length
            time_dist = True
        # inputs
        data_input = keras.layers.Input(
            shape=(sequence_length, dimension), name="inputs"
        )
        lstm_in = data_input  # starter input for lstm
        state_inputs = []  # storage for LSTM state inputs
        state_outputs = []  # storage for LSTM state outputs

        # lstm layers
        for layer_i in range(n_layers):
            return_sequences = True
            if (layer_i == n_layers - 1) and not time_dist:
                # return sequences false if last layer, and not time distributed.
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
                return_state=True,  # state_input_output # better to keep these outputs and just not use.
            )(lstm_in, initial_state=state_input)
            lstm_in = lstm_out
            state_outputs += [state_h_output, state_c_output]

        # mdn layer
        mdn_layer = mdn.MDN(dimension, n_mixtures, name="mdn_outputs")
        click.secho(f"MDN layer type is: {type(mdn_layer)}", fg="blue")
        click.secho(f"Is MDN a keras.layer.Layer? {isinstance(mdn_layer, keras.layers.Layer)}", fg="blue")
        click.secho(f"MDN layer shape: {mdn_layer.compute_output_shape(lstm_out.shape)}", fg="blue")
        if time_dist:
            mdn_layer = keras.layers.TimeDistributed(mdn_layer, name="td_mdn")
        mdn_out = mdn_layer(lstm_out)  # apply mdn
        if inference:
            # for inference, need to track state of the model
            inputs = [data_input] + state_inputs
            outputs = [mdn_out] + state_outputs
        else:
            # for training we don't need to keep track of state in the model
            inputs = data_input
            outputs = mdn_out
        name = mdrnn_model_name(dimension, n_layers, n_hidden_units, n_mixtures)
        new_model = keras.models.Model(
            inputs=inputs, outputs=outputs, name=name
        )

        if not inference:
            # only need loss function and compile when training
            loss_func = mdn.get_mixture_loss_func(dimension, n_mixtures)
            optimizer = keras.optimizers.Adam()
            new_model.compile(loss=loss_func, optimizer=optimizer)
        # else:
        #     new_model.compile()  # no loss or optimizer needed for inference.
        new_model.summary()
        return new_model
    
    # run the tests.
    n_hidden_units = 32
    dimension = 8
    n_mixtures = 5
    n_layers = 2

    # build a training model
    train_model = build_mdrnn_model(dimension, n_hidden_units, n_mixtures, n_layers=n_layers, inference=False, seq_length=50)
    assert isinstance(train_model, keras.models.Model)
    # build an inference model
    inference_model = build_mdrnn_model(dimension, n_hidden_units, n_mixtures, n_layers=n_layers, inference=True, seq_length = 1)
    assert isinstance(inference_model, keras.models.Model)
