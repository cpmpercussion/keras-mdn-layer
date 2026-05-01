import keras
import keras_mdn_layer as mdn
import numpy as np
import click


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
                return_state=True,
            )(lstm_in, initial_state=state_input)
            lstm_in = lstm_out
            state_outputs += [state_h_output, state_c_output]

        # mdn layer
        mdn_layer = mdn.MDN(dimension, n_mixtures, name="mdn_outputs")
        click.secho(f"MDN layer type is: {type(mdn_layer)}", fg="blue")
        click.secho(f"Is MDN a keras.layers.Layer? {isinstance(mdn_layer, keras.layers.Layer)}", fg="blue")
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
