from enum import IntEnum

import pandas as pd


class HistoryColumns(IntEnum):
    OBSERVATION = 0
    ACTION = 1
    REWARD = 2
    NEXT_OBSERVATION = 3
    DONE = 4


class History:
    def __init__(self, max_row_count: int = 100000):
        self._columns = [c.name for c in HistoryColumns]
        self._data = pd.DataFrame(columns=self._columns)
        self._max_row_count = max_row_count

    def update(self, observation, action, reward, next_observation, done):
        self._data = self._data.append(
            dict(zip(self._columns, [observation, action, reward, next_observation, done])),
            ignore_index=True
        )[-self._max_row_count:]

    def data(self) -> pd.DataFrame:
        return self._data

    def size(self) -> int:
        return len(self.data().index)