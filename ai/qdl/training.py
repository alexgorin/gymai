from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import gym
import numpy as np
import pandas as pd
import tensorflow as tf

from ai.common.agent import ExplorerFactory
from ai.common.env_runner import EnvRunner
from ai.common.history import History, HistoryColumns
from ai.common.utils import to_array_of_lists
from ai.qdl.agent import QLAgent


def init_q_values_nearest_terminal(history_sample: pd.DataFrame, actions_count: int) -> np.ndarray:
    avg_step_distance = np.mean(history_sample[
        [HistoryColumns.OBSERVATION.name, HistoryColumns.NEXT_OBSERVATION.name]
    ].apply(
        lambda row: np.linalg.norm(row[HistoryColumns.OBSERVATION.name] - row[HistoryColumns.NEXT_OBSERVATION.name]),
        axis=1,
    ))
    avg_reward_per_step = np.mean(history_sample[[HistoryColumns.REWARD.name]])[HistoryColumns.REWARD.name]
    avg_reward_per_distance = avg_reward_per_step / avg_step_distance
    terminals = history_sample[history_sample[HistoryColumns.DONE.name] == True][
        [HistoryColumns.NEXT_OBSERVATION.name, HistoryColumns.REWARD.name]
    ]
    rewards_to_nearest_terminal = history_sample[[
        HistoryColumns.OBSERVATION.name, HistoryColumns.DONE.name, HistoryColumns.REWARD.name
    ]].apply(
        _reward_to_nearest_terminal, args=(terminals, actions_count, avg_reward_per_distance, False), axis=1,
    )
    rewards_to_nearest_terminal_next = history_sample[[
        HistoryColumns.NEXT_OBSERVATION.name, HistoryColumns.DONE.name, HistoryColumns.REWARD.name
    ]].apply(
        _reward_to_nearest_terminal, args=(terminals, actions_count, avg_reward_per_distance, True), axis=1,
    )
    return rewards_to_nearest_terminal, rewards_to_nearest_terminal_next


def _reward_to_nearest_terminal(
        row, terminals: pd.DataFrame, actions_count: int,
        avg_reward_per_distance: float, is_next_observation: bool
) -> List[float]:
    observation, done, reward = row
    if done and is_next_observation:
        return [reward] * actions_count

    distance, terminal_reward = min(
        terminals.apply(
            lambda row: (np.linalg.norm(observation - row[0]), row[1]),
            axis=1
        ),
        key=lambda row: row[0]
    )
    return [
        terminal_reward + distance * avg_reward_per_distance
    ] * actions_count


def init_q_values_path_to_terminal(history_sample: pd.DataFrame, actions_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Requires sample of consequent moves, not randomized"""
    rewards = []
    for row_index in reversed(range(len(history_sample.index))):
        observation, action, reward, next_observation, done = history_sample.iloc[row_index]
        if done or row_index == len(history_sample.index) - 1:
            rewards.append((
                [reward] * actions_count, [0] * actions_count
            ))
        else:
            next_observation_reward = rewards[-1][0][0]
            rewards.append(
                ([next_observation_reward + reward] * actions_count, [next_observation_reward] * actions_count)
            )
    return tuple(zip(*reversed(rewards)))


@dataclass
class Trainer:
    model: tf.keras.Model
    model_fit_args: Dict
    sample_size: int = 1000
    discount: float = 0.99
    target_model_update_period = 5
    target_model: tf.keras.Model = field(init=False)
    count_till_target_model_update: int = field(default=0, init=False)
    initial_epoch: int = field(default=0, init=False)

    def __post_init__(self):
        self.target_model = tf.keras.models.clone_model(self.model)

    def update_target_model(self):
        if self.count_till_target_model_update == 0:
            self.target_model.set_weights(self.model.get_weights())
            self.count_till_target_model_update = self.target_model_update_period
        else:
            self.count_till_target_model_update -= 1

    def _is_first_iteration(self) -> bool:
        return self.initial_epoch == 0

    def train_iteration(self, history: History):
        data_sample = history.data() if self._is_first_iteration() else self._sample_history(history)

        # (len(data_sample.index), observation_size)
        observations = data_sample.apply(
            lambda row: row[HistoryColumns.OBSERVATION], axis=1, result_type='expand')

        # (len(data_sample.index), observation_size)
        next_observations = data_sample.apply(
            lambda row: row[HistoryColumns.NEXT_OBSERVATION], axis=1, result_type='expand')

        if self._is_first_iteration():
            actions_count = self.model.output_shape[1]
            data_sample['predicted_q_values'], data_sample['predicted_q_values_next'] = init_q_values_path_to_terminal(
                data_sample, actions_count)
        else:
            data_sample['predicted_q_values'] = to_array_of_lists(self.target_model.predict(observations))
            data_sample['predicted_q_values_next'] = to_array_of_lists(self.target_model.predict(next_observations))

        # (len(data_sample.index), action_count)
        adjusted_q_values = data_sample.apply(self._adjusted_q_values, axis=1, result_type='expand')

        training_history = self.model.fit(observations, adjusted_q_values, **self._set_epoch_values())
        self.initial_epoch += len(training_history.epoch)
        self.update_target_model()

    def _set_epoch_values(self) -> Dict:
        """ Adjusting the epoch indices for tensorboard """
        epochs = self.model_fit_args.get('epochs', 1)
        fit_args = {
            **self.model_fit_args,
            **{
                'initial_epoch': self.initial_epoch,
                'epochs': epochs + self.initial_epoch,
            }
        }
        return fit_args

    def _adjusted_q_values(self, row) -> np.ndarray:
        observation, action, reward, next_observation, done, predicted_q_value, predicted_q_value_next = row
        adjusted_q_value = list(predicted_q_value)
        if done:
            adjusted_q_value[action] = reward
        else:
            adjusted_q_value[action] = reward + self.discount * max(predicted_q_value_next)
        return adjusted_q_value

    def _sample_history(self, history: History) -> pd.DataFrame:
        df = history.data()
        terminal_state_observations = df.loc[df[HistoryColumns.DONE.name]][HistoryColumns.OBSERVATION.name]
        terminal_state_proximity_weights = df.apply(
            lambda row: self._terminal_state_proximity_weight(min((
                self._distance(row[HistoryColumns.OBSERVATION], terminal_state_observation)
                for terminal_state_observation in terminal_state_observations)
            )),
            axis=1
        )
        return history.data().sample(self.sample_size, weights=terminal_state_proximity_weights).reset_index(drop=True)
        # return history.data().sample(self.sample_size).reset_index(drop=True)

    @staticmethod
    def _terminal_state_proximity_weight(distance_to_terminal_state: float) -> float:
        return 1 / (1 + distance_to_terminal_state)

    @staticmethod
    def _distance(values1: List, values2: List):
        return np.sqrt(sum((val1 - val2)**2 for val1, val2 in zip(values1, values2)))


class TrainRunner(EnvRunner):
    def __init__(
            self, env: gym.Env, agent: QLAgent, render: bool, explorer_factory: ExplorerFactory, trainer: Trainer,
            history: History, file_writer, train_every: int = 1000,
    ):
        super().__init__(env, explorer_factory.explorer(agent), render)
        self.history = history
        self.trainer = trainer
        self.file_writer = file_writer
        self.train_every = train_every
        self.steps_till_training = self.train_every
        self._agent = agent

    def on_step(self, episode_index, step_index, observation):
        action, next_observation, reward, done, info = super().on_step(episode_index, step_index, observation)
        # reward = reward if not done else -1000  # Specific to CartPole-v0 world
        self.history.update(observation, action, reward, next_observation, done)
        self.steps_till_training -= 1
        return action, next_observation, reward, done, info

    def on_done(self, episode_index, step_index, observation, action, next_observation, reward, done, info):
        if self.steps_till_training <= 0:
            if self.history.size() >= self.trainer.sample_size:
                self._log_to_tensorboard()
                print("Training")
                self.trainer.train_iteration(self.history)
                self.steps_till_training = self.train_every

    def _log_to_tensorboard(self):
        avg_reward, avg_step_count = self._performance()
        step = self.trainer.initial_epoch
        with self.file_writer.as_default():
            tf.summary.scalar("Avg reward", avg_reward, step=step)
            tf.summary.scalar("Avg step count", avg_step_count, step=step)
        self.file_writer.flush()
        print(f"Avg steps count: {avg_step_count}, avg reward: {avg_reward}")

    def _prev_iteration_performance(self) -> Tuple[float, float]:
        episode_count, overall_steps_count, overall_reward = 0, 0, 0
        for steps_count, reward in reversed(self.episode_reward_history):
            episode_count += 1
            overall_steps_count += steps_count
            overall_reward += reward
            if overall_steps_count >= self.train_every:
                return overall_reward / episode_count, overall_steps_count / episode_count

    def _performance(self) -> Tuple[float, float]:
        test_env_runner = EnvRunner(self.env, self._agent, True)
        test_results = test_env_runner.run(10, 10000)
        episode_step_counts, episode_rewards = tuple(zip(*test_results))
        return np.mean(episode_rewards), np.mean(episode_step_counts)
