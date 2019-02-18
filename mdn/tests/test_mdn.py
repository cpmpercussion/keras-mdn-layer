import keras
import mdn


def test_build_mdn():
    N_HIDDEN = 5
    N_MIXES = 5
    model = keras.Sequential()
    model.add(keras.layers.Dense(N_HIDDEN, batch_input_shape=(None, 1), activation='relu'))
    model.add(keras.layers.Dense(N_HIDDEN, activation='relu'))
    model.add(mdn.MDN(1, N_MIXES))
    model.compile(loss=mdn.get_mixture_loss_func(1, N_MIXES), optimizer=keras.optimizers.Adam())
    assert isinstance(model, keras.engine.sequential.Sequential)
