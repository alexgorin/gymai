import tensorflow as tf


def load_model(path: str, optimizer="Adam", loss={'output_layer': 'mse'}, **kwargs) -> tf.keras.Model:
    model = tf.keras.models.load_model(path)
    model.compile(optimizer, loss, **kwargs)
    return model
