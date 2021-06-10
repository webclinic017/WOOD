import numpy as np
import itertools
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pds
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import deque
import random
import numpy as np
import joblib
from tqdm import tqdm


class Golem_env(gym.Env):
  
  def __init__(self, df_train):
    
    print('\n__init__Env')
    self.df_train = df_train # Our dataframe experience
    self.nb_candle = self.df_train.shape[0] # Lenght of the experience
    self.cursor = None # In which candle aare we

    self.position = None # If it's Long, Short or Flat
    self.signal = self.df_train.Signal # Assign the Signal column where will be our target
    self.target = None # The target to compare to
    self.espace_action = spaces.Discrete(3) # How many states have we got : 0 = Hold, 1 = Buy, 2 = Sell 
    self.espace_observation = spaces.MultiDiscrete(self.df_train+self.signal)
    self.reward = None
    
    self._seed()
    self._reset()


  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _observation(self):
    obs = []
    obs.extend(self.position)
    return obs

  def _reset(self):
    self.cursor = 0
    self.position = [0]
    return self._observation()

  def _step(self, action):
    print('\n_step')
    assert self.espace_action.contains(action)
    self.cursor += 1
    self.target= self.signal[self.curseur] 

    if self.target == 1 and action == 1:
      self.reward += 1000
    elif self.target == 0 and action == 0:
      self.reward += 100
    elif self.target == -1 and action == 2:
      self.reward += 1000
    elif self.target == 1 and action == 0:
      self.reward += -2000
    elif self.target == -1 and action == 0:
      self.reward += -2000
    elif self.target == 0 and action == 1:
     self.reward += -1000
    elif self.target == -1 and action == 1:
      self.reward += -1000
    elif self.target == 0 and action == 2:
      self.reward += -1000
    elif self.target == 1 and action == 2:
      self.reward += -1000
  
    done = self.cursor == self.nb_candle - 1
    info = {'valeur_actuelle': self.reward}
    return self._observation(), self.reward, done, info





class Golem(object):

  def __init__(self, state_size, action_size):
    print('\n__init__Golem')
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=2000)
    self.gamma = 0.95  
    self.epsilon = 1.0  
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = ANN(state_size, action_size)


  def memory_dump(self, state, action, reward, next_state, done):
    self.memory.append((state, action, self.reward, next_state, done))


  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])  


  def replay(self, batch_size=32):
    minibatch = random.sample(self.memoire, batch_size)

    states = np.array([tup[0][0] for tup in minibatch])
    actions = np.array([tup[1] for tup in minibatch])
    rewards = np.array([tup[2] for tup in minibatch])
    next_states = np.array([tup[3][0] for tup in minibatch])
    done = np.array([tup[4] for tup in minibatch])
    
    #Q(e(t+1),a)
    q_prime = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)

    q_prime[done] = rewards[done]

    #Q(e,a)
    q = self.model.predict(states)
    q[range(batch_size), actions] = q_prime

    self.model.fit(states, q, epochs=1, verbose=0)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay


#### ANN qui va determiner l'action optimale à choisir
def ANN(nb_obs, nb_action, nb_couche=3, nb_neurone=55,
        activation='relu', loss='mse'):
  
  print('\ANN')
  #### Initialisation du modèle 
  model = Sequential()
  model.add(Dense(nb_neurone, input_dim=nb_obs, activation=activation))
  
  #### Boucle pour rajouter le nombre de couche cachée voulue
  for _ in tqdm(range(nb_couche)):
    model.add(Dense(nb_neurone, activation=activation))
  #### Sortie du modèle et compilation
  model.add(Dense(nb_action, activation='linear'))
  model.compile(loss=loss, optimizer=Adam())
  print(model.summary())
  return model


print('Initialisation...')
##### MISE EN PLACE DES VARIABLES DE LANCEMENT
_epoch = 15
batch_size = 32
print('\nChargement des bases')
##### CREATION DE LA DATASET EN LA TRANSPOSANT
df = joblib.load("BASES/test_EURUSD_m5") 
df.sort_index(ascending=False,inplace=True)
df_train = df.iloc[:600000,:]
##### MISE EN PLACE DU MODELE
print('\nMise en place du modèle')
env = Golem_env(df_train)
state_size = env.espace_observation.shape
action_size = env.espace_action.n
golem = Golem(state_size, action_size)


##### LANCEMENT DU MODELE
if __name__ == "__main__":

  print('\n Lancement du modèle...')

  for t in range(_epoch):
    state = env.reset()

    for time in tqdm(range(env.nb_candle)):
      action = golem.act(state)
      next_state, _reward, done, info = env._step(action)
      golem.memory(state, action, _reward, next_state, done)
      state = next_state
      if done:
        print("etape: {}/{}, etape et valeur: {}".format(t + 1, _epoch, info))
        break
      if len(golem.memory) > batch_size:
        golem.replay(batch_size)

##### TEST DU MODELE
df_test = df.iloc[600000:,:]
env = Golem_env(df_test)
episode = 55 
if __name__ == "__main__":
  for t in tqdm(range(_epoch)):
    state = env.reset()
    for time in range(env.n_jour):
      action = golem.act(state)
      next_state, _reward, done, info = env._step(action)
      state = next_state
      if done:
        print("etape: {}/{}, etape et valeur: {}".format(t + 1, _epoch, info))
        break