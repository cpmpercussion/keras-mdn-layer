from tensorflow import keras
import keras_mdn_layer as mdn
import numpy as np


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



def test_build_time_distributed_MDN():
    """Builds a time-distributed MDN model for sequence-to-sequence prediction."""
    SEQ_LEN = 50
    HIDDEN_UNITS = 256
    OUTPUT_DIMENSION = 3
    NUMBER_MIXTURES = 10

    inputs = keras.layers.Input(shape=(SEQ_LEN,OUTPUT_DIMENSION), name='inputs')
    lstm1_out = keras.layers.LSTM(HIDDEN_UNITS, name='lstm1', return_sequences=True)(inputs)
    lstm2_out = keras.layers.LSTM(HIDDEN_UNITS, name='lstm2', return_sequences=True)(lstm1_out)
    mdn_layer = mdn.MDN(OUTPUT_DIMENSION, NUMBER_MIXTURES, name='mdn_outputs')
    mdn_out = keras.layers.TimeDistributed(mdn_layer, name='td_mdn')(lstm2_out)

    model = keras.models.Model(inputs=inputs, outputs=mdn_out, name="kanji-training-td")
    model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMENSION,NUMBER_MIXTURES), optimizer='adam')
    model.summary()
    assert isinstance(model, keras.models.Model)

def test_rnn_inference_model():
    HIDDEN_UNITS = 256
    OUTPUT_DIMENSION = 3
    NUMBER_MIXTURES = 10

    decoder = keras.Sequential()
    decoder.add(keras.layers.LSTM(HIDDEN_UNITS, batch_input_shape=(1,1,OUTPUT_DIMENSION), return_sequences=True, stateful=True))
    decoder.add(keras.layers.LSTM(HIDDEN_UNITS, stateful=True))
    decoder.add(mdn.MDN(OUTPUT_DIMENSION, NUMBER_MIXTURES))
    decoder.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMENSION,NUMBER_MIXTURES), optimizer=keras.optimizers.Adam())
    decoder.summary()
    assert isinstance(decoder, keras.Sequential)
