import gym
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from ai.common.env_runner import EnvRunner
from ai.common.agent import ExplorerFactory, RandomExplorer
from ai.common.model import load_model
from ai.common.history import HistoryColumns, History

from ai.qdl.agent import QLAgent
from ai.qdl.training import Trainer
import datetime
from ai.qdl.training import TrainRunner


def get_model(observation_size: int, action_count: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(observation_size,)),
            tf.keras.layers.BatchNormalization(),
            layers.Dense(20, activation="relu", name="layer1"),
            layers.Dropout(0.2),
            layers.Dense(20, activation="relu", name="layer2"),
            layers.Dense(action_count, activation="linear", name="output_layer"),
        ]
    )
    model.compile(optimizer="Adam", loss={'output_layer': 'mse'})
    return model


def new_agent(env: gym.Env) -> QLAgent:
    observation_size = env.observation_space.shape[0]
    action_count = env.action_space.n
    model = get_model(observation_size, action_count)
    return QLAgent(model, observation_size)


def main():
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 10000  # Specific for CartPole-v0
    observation_size = env.observation_space.shape[0]
    action_count = env.action_space.n
    np.random.seed(0)
    model = get_model(observation_size, action_count)
    # model = load_model("../model_dumps/model7")
    agent = QLAgent(model, observation_size)
    # result = EnvRunner(env, agent, render=True).run(10, 10000)
    log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    explorer_factory = ExplorerFactory(
        RandomExplorer, env.action_space.n, init_randomization_factor=0.7, randomization_factor_decay=0)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=1000),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]
    sample_size = 10000
    trainer = Trainer(
        model,
        model_fit_args=dict(
            batch_size=sample_size, verbose=0, epochs=20000, shuffle=False, callbacks=callbacks),
        sample_size=sample_size,
    )
    file_writer = tf.summary.create_file_writer(log_dir)
    render = False
    TrainRunner(
        env, agent, render, explorer_factory, trainer,
        History(max_row_count=200000), file_writer, train_every=sample_size // 10
    ).run(5000, 2000)
    placeholder_for_breakpoint = 0


if __name__ == "__main__":
    main()
