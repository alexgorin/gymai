from dataclasses import dataclass, field
from typing import List, Tuple

import gym

from ai.common.agent import Agent


@dataclass
class EnvRunner:
    env: gym.Env
    agent: Agent
    render: bool = False
    episode_reward_history: List[Tuple[int, float]] = field(default_factory=list, init=False)

    def run(self, episode_count: int, steps_count: int) -> List[float]:
        self.on_start()
        done, step_index = False, 0
        self.episode_reward_history = []
        for episode_index in range(episode_count):
            episode_reward = 0
            observation = self.env.reset()
            self.on_episode_start(episode_index, observation)
            for step_index in range(steps_count):
                action, next_observation, reward, done, info = self.on_step(episode_index, step_index, observation)
                episode_reward += reward
                if done:
                    self.on_done(episode_index, step_index, observation, action, next_observation, reward, done, info)
                    break
                else:
                    observation = next_observation
            self.episode_reward_history.append((step_index + 1, episode_reward))
            self.on_episode_finish(episode_index, done, steps_count=step_index)

        self.on_finish()
        return self.episode_reward_history

    def on_start(self):
        pass

    def on_finish(self):
        self.env.close()

    def on_episode_start(self, episode_index, observation):
        pass

    def on_episode_finish(self, episode_index, done, steps_count):
        if done:
            print(f"Episode {episode_index} finished after {steps_count} steps")
        else:
            print(f"Episode {episode_index} lasted longer than {steps_count + 1} steps")

    def on_step(self, episode_index, step_index, observation):
        if self.render:
            self.env.render()
        action = self.agent.act(observation)
        next_observation, reward, done, info = self.env.step(action)
        return action, next_observation, reward, done, info

    def on_done(self, episode_index, step_index, observation, action, next_observation, reward, done, info):
        pass
