import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class Actions(Enum):
    Hold = 0
    Buy = 1
    Sell = 2


class Positions(Enum):
    Flat = 0
    Long = 1
    Short = 2


class GolemTradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size):
        assert df.ndim == 2

        # Fix the random seed state
        self.seed()
        # Let's call it df, won't we?
        self.df = df
        # Do the same with window_size
        self.window_size = window_size
        # Get Signal & Features from df
        self.signal, self.signal_features = self._process_data()
        # Make the shape
        self.shape = (window_size, self.signal_features.shape[1])

        # How lany Action do we have
        self.action_space = spaces.Discrete(len(Actions))
        # Dimension of our space, and caracteristics
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # For one episode, we begin at window_size
        self._start_tick = self.window_size
        # And we finish at the end of price (processed df)
        self._end_tick = len(self.signal) - 1

        # Initialisation of variables
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None


    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 40
        self._position = Positions.Flat
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()


    def step(self, action):
        
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True
        else:
            self._done = False

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(step_reward)

        if action == 0:
            self._position = Positions.Flat
        elif action == 1:
            self._position = Positions.Long
        elif action == 2:
            self._position = Positions.Short
        
        '''if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True
        else:
            trade = False

        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick'''

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info


    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size):self._current_tick]


    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            elif position == Positions.Flat:
                color = 'orange'
            if color:
                plt.scatter(tick, self.ssignal[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.signal)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)


    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.signal)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.signal[short_ticks], 'ro')
        plt.plot(long_ticks, self.signal[long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
        
        
    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()


    def _process_data(self):
        raise NotImplementedError


    def _calculate_reward(self, action):
        raise NotImplementedError


    def _update_profit(self, action):
        raise NotImplementedError


    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
