import numpy as np

from .golem_env import *


class ForexEnv(GolemTradingEnv):

    def __init__(self, df, window_size, frame_bound, unit_side='left'):
        assert len(frame_bound) == 2
        assert unit_side.lower() in ['left', 'right']

        self.frame_bound = frame_bound
        self.unit_side = unit_side.lower()
        super().__init__(df, window_size)
 
    def _process_data(self):
        signal = self.df.loc[:, 'Signal'].to_numpy()

        signal[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        signal = signal[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        #diff = np.insert(np.diff(prices), 0, 0)
        #signal_features = np.column_stack((self.df[['Open','High','Low','Close']]))
        signal_features = self.df[['Open','High','Low','Close']]

        return signal, signal_features


    def _calculate_reward(self, action):
        step_reward = 0  # pip

        if (action == 1 and self.signal[self._current_tick] == 1) or (action == 2 and self.signal[self._current_tick] == -1):
            step_reward += 100
        elif (action == 0 and self.signal[self._current_tick] == 0):
            step_reward += 10
        elif (action == 0 and self.signal[self._current_tick] == 1) or (action == 0 and self.signal[self._current_tick] == -1):
            step_reward -= 100
        elif (action == 1 and self.signal[self._current_tick] == -1) or (action == 1 and self.signal[self._current_tick] == 0) or (action == 2 and self.signal[self._current_tick] == 1)\
             or (action == 2 and self.signal[self._current_tick] == 0):
            step_reward -= 10

        return step_reward


    def _update_profit(self, step_reward):
        self._total_profit += step_reward


    def max_possible_profit(self):
        current_tick = self._start_tick
        profit = 0

        while current_tick <= self._end_tick:

            if (self.signal[current_tick] == 1) or (self.signal[current_tick] == -1):
                profit += 100
            elif (self.signal[current_tick] == 0):
                profit += 10

            current_tick = current_tick - 1

        return profit
