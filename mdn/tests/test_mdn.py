from tensorflow.compat.v1 import keras
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
    model.save('test_save.h5', overwrite=True, save_format="h5")
    m_2 = keras.models.load_model('test_save.h5', custom_objects={'MDN': mdn.MDN, 'mdn_loss_func': mdn.get_mixture_loss_func(1, N_MIXES)})
    assert isinstance(m_2, keras.Sequential)
