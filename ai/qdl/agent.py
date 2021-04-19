from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from ai.common.agent import Agent


@dataclass
class QLAgent(Agent):
    model: tf.keras.Model
    observation_size: int

    def act(self, observation) -> int:
        prediction = self.model.predict(np.ndarray(shape=(1, self.observation_size), buffer=observation)).flatten()
        return np.argmax(prediction)
