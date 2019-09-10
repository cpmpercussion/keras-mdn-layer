import keras
import mdn
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
    assert isinstance(model, keras.engine.sequential.Sequential)


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
    model.save('test_save.h5')
    m_2 = keras.models.load_model('test_save.h5', custom_objects={'MDN': mdn.MDN, 'mdn_loss_func': mdn.get_mixture_loss_func(1, N_MIXES)})
    assert isinstance(m_2, keras.engine.sequential.Sequential)


def test_training_ann_one_epoch():
    """Trains a small feed-forward model for one epoch."""
    NSAMPLE = 100
    y_data = np.float32(np.random.uniform(-10.5, 10.5, NSAMPLE))
    r_data = np.random.normal(size=NSAMPLE)
    x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
    x_data = x_data.reshape((NSAMPLE, 1))
    N_HIDDEN = 5
    N_MIXES = 5
    model = keras.Sequential()
    model.add(keras.layers.Dense(N_HIDDEN, batch_input_shape=(None, 1), activation='relu'))
    model.add(keras.layers.Dense(N_HIDDEN, activation='relu'))
    model.add(mdn.MDN(1, N_MIXES))
    model.compile(loss=mdn.get_mixture_loss_func(1, N_MIXES), optimizer=keras.optimizers.Adam())
    history = model.fit(x=x_data, y=y_data, batch_size=128, epochs=1, validation_split=0.15)
    assert isinstance(history, keras.callbacks.History)


def test_training_rnn_one_epoch():
    # Training Hyperparameters:
    SEQ_LEN = 30
    BATCH_SIZE = 64
    HIDDEN_UNITS = 32
    EPOCHS = 1
    OUTPUT_DIMENSION = 4
    NUMBER_MIXTURES = 5
    # make neural network
    model = keras.Sequential()
    model.add(keras.layers.LSTM(HIDDEN_UNITS, batch_input_shape=(None,SEQ_LEN,OUTPUT_DIMENSION), return_sequences=True))
    model.add(keras.layers.LSTM(HIDDEN_UNITS))
    model.add(mdn.MDN(OUTPUT_DIMENSION, NUMBER_MIXTURES))
    model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMENSION,NUMBER_MIXTURES), optimizer=keras.optimizers.Adam())
    model.summary()
    # make fake data
    X = np.random.rand(BATCH_SIZE*10, SEQ_LEN, OUTPUT_DIMENSION)
    y = np.random.rand(BATCH_SIZE*10, OUTPUT_DIMENSION)
    # train for one epoch
    history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS)
    assert isinstance(history, keras.callbacks.History)


def test_training_tf_td_rnn_one_epoch():
    SEQ_LEN = 10
    BATCH_SIZE = 64
    HIDDEN_UNITS = 32
    EPOCHS = 1
    OUTPUT_DIMENSION = 3
    NUMBER_MIXTURES = 10
    # make fake data
    X = np.random.rand(BATCH_SIZE*10, SEQ_LEN, OUTPUT_DIMENSION)
    y = np.random.rand(BATCH_SIZE*10, SEQ_LEN, OUTPUT_DIMENSION)
    # setup network
    inputs = keras.layers.Input(shape=(SEQ_LEN, OUTPUT_DIMENSION), name='inputs')
    lstm1_out = keras.layers.LSTM(HIDDEN_UNITS, name='lstm1', return_sequences=True)(inputs)
    lstm2_out = keras.layers.LSTM(HIDDEN_UNITS, name='lstm2', return_sequences=True)(lstm1_out)
    mdn_out = keras.layers.TimeDistributed(mdn.MDN(OUTPUT_DIMENSION, NUMBER_MIXTURES, name='mdn_outputs'), name='td_mdn')(lstm2_out)
    model = keras.models.Model(inputs=inputs, outputs=mdn_out)
    model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMENSION, NUMBER_MIXTURES), optimizer='adam')
    model.summary()
    history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS)
    assert isinstance(history, keras.callbacks.History)
