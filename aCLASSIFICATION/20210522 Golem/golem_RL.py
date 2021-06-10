__author__ = 'LumberJack (Jyss)'
__copyright__ = 'D.A.G. 26 - 5781'
__version__ = 'v2.0'

####################################################################
####################################################################
############################### GOLEM FX ###########################
####################################################################
####################################################################

'''
In this Version, the most significative change is the use of RL
Structure : 

Generation of home made signal in a window, where _target cross High/Low AND _sl doesn't cross Low/High
Making of the features
Scaling
RL (A2C) => action is compared to signal and rewarded according to the results obtained
Testing it in bt
'''

# Gym Stuff
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


# Stable Baselines : RL Stuff
#from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

# Proceesing librairies
import numpy as np
import pandas as pd
from enum import Enum
from matplotlib import pyplot as plt
from natsort import natsorted
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Side Stuff
import time
import joblib
import warnings

# Technical analysis
from finta import TA

warnings.filterwarnings("ignore")

TIK = ['AUD','NZD','GBP','JPY','CHF','CAD','SEK','NOK','ILS','MXN','USD','EUR']
RATE = [0.776,0.721,1.3912,1/105.91,1/0.892,1/1.2681,1/8.2884,1/8.4261,1/3.2385,1/20.1564,1,1.21]
df_ratefx = pd.DataFrame(index=TIK)
df_ratefx['rate'] = RATE

x = 'EUR/USD'
_period = 'm5'
_period2 = 'H1'
_ticker = x.replace('/','')
_start = '2010-01-01' # start the train there '2010-01-01'
_mid = '2016-06-30' # stop the train and begin the test there '2016-08-31'
_stop = '2017-12-31' # stop the test there. After that, it is kept for oos '2017-12-31'
_last = '2021-04-29' # '2020-12-31'
_nb_bougie_exit = 5555555555
_trigger_reengage = 0
_trigger_target = 1
_trigger_invers = 0
_trigger_sl = 1
_trigger_rsi = 1
_verbose = 0
_cash_ini = 200000
_target = 0.002
_sl = 0.001
_exposure = 2
_rate = df_ratefx.loc[x[4:],'rate']
_size = _cash_ini / df_ratefx.loc[x[:3],'rate']
_trigger_spread = 0.025
_no_access = 0


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
        self.action_space = spaces.MultiDiscrete(len(Actions))
        # Dimension of our space, and caracteristics
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=self.shape,dtype=np.float32)

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


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Flat
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()


    def _step(self, action):
        
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
        flat_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)
            elif self._position_history[i] == Positions.Flat:
                flat_ticks.append(tick)

        plt.plot(short_ticks, self.signal[short_ticks], 'ro')
        plt.plot(long_ticks, self.signal[long_ticks], 'go')
        plt.plot(long_ticks, self.signal[long_ticks],color='orange', marker='x')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit + 'MaxProfit : '
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


class ForexEnv(GolemTradingEnv):

    def __init__(self, df, window_size, frame_bound, unit_side='left'):
        assert len(frame_bound) == 2
        #assert unit_side.lower() in ['left', 'right']

        self.frame_bound = frame_bound
        #self.unit_side = unit_side.lower()
        super().__init__(df, window_size)
 
    def _process_data(self):
        signal = self.df.loc[:, 'Signal'].to_numpy()

        signal[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        signal = signal[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        #diff = np.insert(np.diff(prices), 0, 0)
        #signal_features = np.column_stack((self.df[['Open','High','Low','Close']]))
        signal_features = self.df

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

def split_df(df):
    """[Split the dtaframe in train/test/oos and reduce train and test to dataframe with signal!=0 only]

    Args:
        df ([pandas]): [the datframe to split already featured]
        _start ([date]): [beginning of the df]
        _mid ([date]): [stop of the train and beginning of the test]
        _stop ([date]): [stop of the test and beginning of the oos]
        _last ([date]): [end of the oos]
    """    
    df = df.dropna()
    df['Date'] = pd.to_datetime(df.index)
    df_train = df[(df.Date>=_start)&(df.Date<=_mid)]
    df_test = df[(df.Date>_mid)&(df.Date<=_stop)]
    df_oos = df[(df.Date>_stop)&(df.Date <= _last)]
    df_train = df_train.drop(['Date'],axis=1)
    df_test = df_test.drop(['Date'],axis=1)
    df_oos = df_oos.drop(['Date'],axis=1)
    return(df_train, df_test,df_oos)

def klines(df):
    _condition1 = df.Close >= df.Open
    df['Color'] = np.where(_condition1,1,-1)
    _condition2 = df.Color = 1
    df['UpperShadow'] = np.where(_condition2,(df.High-df.Close),(df.High-df.Open))
    df['LowerShadow'] = np.where(_condition2,(df.Open-df.Low),(df.Close-df.Low))
    df['Body'] = abs(df.Close-df.Open)
    return (df)

def strategy(df,_window=40):

    #df['RSI_2'] = TA.RSI(ohlc=df,int=2,str='Close')

    df['Window_High_Ask'] = df.HighAsk.iloc[::-1].rolling(_window).max().iloc[::-1] # Limite SL Short
    df['Window_High_Bid'] = df.HighBid.iloc[::-1].rolling(_window).max().iloc[::-1] # Limite Target Long
    df['Window_Low_Ask'] = df.LowAsk.iloc[::-1].rolling(_window).min().iloc[::-1] # Limite Target Short
    df['Window_Low_Bid'] = df.LowBid.iloc[::-1].rolling(_window).min().iloc[::-1] # Limite SL Long
    df['Window_sl_Short'] = df.CloseBid + (df.CloseBid * _sl) # Short pour SL
    df['Window_sl_Long'] = df.CloseAsk - (df.CloseAsk * _sl) # Long pour SL
    df['Window_tp_Short'] = df.CloseBid - (df.CloseBid * _target) # Short pour TP
    df['Window_tp_Long'] = df.CloseAsk + (df.CloseAsk * _target) # Long pour TP

    ##### CONDITIONS LONG
    _condition_1 = (df['Window_tp_Long'] <= df['Window_High_Bid']) & (df['Window_sl_Long'] <= df['Window_Low_Bid'])

    ##### CONDITIONS SHORT
    _condition_1_bar = (df['Window_tp_Short'] >= df['Window_Low_Ask']) & (df['Window_sl_Short'] >= df['Window_High_Ask'])

    ##### 1 condition
    df['Signal'] = np.where(_condition_1,1,np.where(_condition_1_bar,-1,0))
    df = df.drop(['Symbol','Date','DateIndex','Window_High_Ask','Window_High_Bid','Window_Low_Ask','Window_Low_Bid','Window_sl_Short','Window_sl_Long','Window_tp_Short','Window_tp_Long'], axis=1)
    return(df.sort_index(axis=0))

def undersample(df):
    
    print('Avant Resampling :')
    print('Classe 0',df[df.Signal==0].shape[0])
    print('Classe 1',df[df.Signal==1].shape[0])
    print('Classe -1',df[df.Signal==-1].shape[0])

    # Class count
    count_class_0, count_class_1, count_class_2 = df_train.Signal.value_counts()
    # Divide by class
    df_class_0 = df[df.Signal== 0]
    df_class_1 = df[df.Signal != 0]

    df_class_0_under = df_class_0.sample(int((count_class_1+count_class_1)/1.5))
    df = pd.concat([df_class_0_under, df_class_1], axis=0)
    df = df.sort_index()
    print('Random under-sampling:')
    print(df.Signal.value_counts())

    # Classify and report the results
    print('\nAprès resample:')
    print('Classe 0',df[df.Signal==0].shape[0])
    print('Classe 1',df[df.Signal==1].shape[0])
    print('Classe -1',df[df.Signal==-1].shape[0])
    return df

if __name__ == "__main__":
    print("\nTest du moteur de RL. Rien n'est optimisé pour la performance pour le moment. Seul le moteur est testé")
    
    print("\nChargement de la base et reverse sorting")
    df = joblib.load('BASES/EURUSD_m5')

    print("\nSplit en 3 bases, train, test et oos")

    df = klines(df)

    df = strategy(df)


    df_train, df_test, df_oos = split_df(df)

    df_oos_raw = df_oos.copy().dropna()

    df_train = pd.concat([df_train , df_test])
    df_train.sort_index(inplace=True)

    scaler = MinMaxScaler()

    for i in df.columns.unique():
        if i != 'Signal' and i != 'Color':
            df_train[i] = scaler.fit_transform(df_train[i].values.reshape(-1, 1))
            #df_test[i] = scaler.fit_transform(df_test[i].values.reshape(-1, 1))
            df_oos[i] = scaler.fit_transform(df_oos[i].values.reshape(-1, 1))

    df_train = df_train.dropna()
    df_train = undersample(df_train)
    #df_test = df_test.dropna()
    df_oos = df_oos.dropna()
    
    df_oos= df_oos.reindex(natsorted(df_oos.columns), axis=1)
    df_train = df_train.reindex(natsorted(df_train.columns), axis=1)
    df_oos_raw = df_oos_raw.reindex(natsorted(df_oos_raw.columns), axis=1)

    print("Sorting the 3 bases in Ascending=False")
    df_train.sort_index(ascending=False,inplace=True)
    df_test.sort_index(ascending=False,inplace=True)
    df_oos.sort_index(ascending=False,inplace=True)
    print('\nTail du train')
    print(df_train.tail())
    print("\nTail du test")
    print(df_test.tail())
    print("\nTail du oos")
    print(df_oos.tail())
    #print('Value max :',max_possible_profit(df_train))
    print("\nCréation de l'environnement")
    _env = ForexEnv(df_oos,frame_bound=(40,int(df_oos.shape[0])),window_size=40)
    
    print("\nTest random sans learning")
    _state = _env.reset()
    _reward_max = 0
    _reward_min = 0
    while True:
        _action = _env.action_space.sample()
        _n_state, _reward, _done, _info = _env._step(_action)
        if _reward > _reward_max:
            _reward_max = _reward
            print('\nNew reward_max :',_reward,'\n')
        if _reward < _reward_min:
            _reward_min = _reward
            print('\nNew reward_min :',_reward,'\n')
        if _done :
            print('Info :',_info)
            break
    plt.figure(figsize=(24,6))
    plt.cla()
    _env.render_all()
    plt.show()
    
    _env = ForexEnv(df_train,frame_bound=(40,int(df_train.shape[0])),window_size=40)
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=40000, verbose=1)
    eval_callback = EvalCallback(_env, callback_on_new_best=callback_on_best, verbose=1)

    print("\nMise en marche de l'agent pour l'apprentissage")
    _model = A2C('MlpPolicy',_env,verbose=1)
    _model.learn(total_timesteps=100,callback=eval_callback)

    print("\nEvaluation du système")
    _obs = _env.reset()
    while True:
        _obs = df_test
        _action, _states = _model.predict(_obs)
        _obs, _reward, _done, _info = _env.step(_action)
        if _done :
            print('Info :',_info)
            break
    plt.figure(figsize=(24,6))
    plt.cla()
    _env.render_all()
    plt.show()