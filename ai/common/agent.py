from abc import ABC
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf


class Agent:
    def act(self, observation) -> int:
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()


@dataclass
class Explorer(Agent, ABC):
    agent: Agent
    action_count: int


class ExplorerFactory:
    def __init__(self, explorer_class: type, action_count: int, **kwargs):
        self.explorer_class = explorer_class
        self.action_count = action_count
        self.explorer_params = kwargs

    def explorer(self, agent) -> Explorer:
        return self.explorer_class(agent, self.action_count, **self.explorer_params)


@dataclass
class RandomExplorer(Explorer):
    init_randomization_factor: float
    randomization_factor_decay: float = 0
    _call_count: int = field(default=0, init=False)

    def act(self, observation):
        rand_factor = self.init_randomization_factor * (1 - self.randomization_factor_decay) ** self._call_count
        self._call_count += 1
        if np.random.random() < rand_factor:
            random_action = np.random.randint(self.action_count)
            return random_action
        else:
            return self.agent.act(observation)
