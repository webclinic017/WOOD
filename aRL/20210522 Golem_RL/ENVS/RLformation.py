import numpy as np
import itertools
import gym
from gym import spaces
from gym.utils import seeding



class env_financier(gym.Env):
  
  def __init__(self, data_train, invest_ini=20000):
    
    self.train_data = np.around(data_train)
    self.nb_pf, self.n_jour = self.train_data.shape

    self.invest_ini = invest_ini
    self.curseur = None
    self.position_prise = None
    self.prix_position = None
    self.budget_disp = None

    self.espace_action = spaces.Discrete(3**self.nb_pf)

    variation_max_prix = self.train_data.max(axis=1)
    variation_actif = [[0, invest_ini * 2 // h] for h in variation_max_prix]
    variation_prix = [[0, h] for h in variation_max_prix]
    variation_budget_disp = [[0, invest_ini * 2]]
    self.espace_observation = spaces.MultiDiscrete(variation_actif + variation_prix + variation_budget_disp)

    
    self._seed()
    self._reset()


  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _observation(self):
    obs = []
    obs.extend(self.position_prise)
    obs.extend(list(self.prix_position))
    obs.append(self.budget_disp)
    return obs

  def _reset(self):
    self.curseur = 0
    self.position_prise = [0] * self.nb_pf
    self.prix_position = self.train_data[:, self.curseur]
    self.budget_disp = self.invest_ini
    return self._observation()




  def _etape(self, action):
    assert self.espace_action.contains(action)
    val_precedente = np.sum(self.position_prise * self.prix_position) + self.budget_disp
    self.curseur += 1
    self.prix_position = self.train_data[:, self.curseur] 
    action_combo = list(map(list, itertools.product([0, 1, 2], repeat=self.nb_pf)))
    action_vec = action_combo[action]

    
    sell_index = []
    buy_index = []
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)

   
    if sell_index:
      for i in sell_index:
        self.budget_disp += self.prix_position[i] * self.position_prise[i]
        self.position_prise[i] = 0
    if buy_index:
      can_buy = True
      while can_buy:
        for i in buy_index:
          if self.budget_disp > self.prix_position[i]:
            self.position_prise[i] += 1 
            self.budget_disp -= self.prix_position[i]
          else:
            can_buy = False
    valeur_curseur = np.sum(self.position_prise * self.prix_position) + self.budget_disp
    recompense = 85 * (valeur_curseur - val_precedente)
    done = self.curseur == self.n_jour - 1
    info = {'valeur_actuelle': valeur_curseur}
    return self._observation(), recompense, done, info



from collections import deque
import random
import numpy as np



class Agent(object):

  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memoire = deque(maxlen=2000)
    self.gamma = 0.95  
    self.epsilon = 1.0  
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = modele(state_size, action_size)


  def souvenir(self, etat, action, recompense, prochain_etat, done):
    self.memoire.append((etat, action, recompense, prochain_etat, done))


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


import numpy as np
from sklearn.preprocessing import StandardScaler


def affiche_transformation(env):
  min_ = [0] * (env.nb_pf * 2 + 1)
  max_ = []
  max_price = env.train_data.max(axis=1)
  min_price = env.train_data.min(axis=1)
  max_cash = env.invest_ini * 3 
  max_stock_owned = max_cash // min_price
  for i in max_stock_owned:
    max_.append(i)
  for i in max_price:
    max_.append(i)
  max_.append(max_cash)
  scaler = StandardScaler()
  scaler.fit([min_, max_])
  return scaler


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#### ANN qui va determiner l'action optimale à choisir
def modele(nb_obs, nb_action, nb_couche=3, nb_neurone=55,
        activation='tanh', loss='mse'):
  #### Initialisation du modèle 
  model = Sequential()
  model.add(Dense(nb_neurone, input_dim=nb_obs, activation=activation))
  
  #### Boucle pour rajouter le nombre de couche cachée voulue
  for _ in range(nb_couche):
    model.add(Dense(nb_neurone, activation=activation))
  #### Sortie du modèle et compilation
  model.add(Dense(nb_action, activation='linear'))
  model.compile(loss=loss, optimizer=Adam())
  print(model.summary())
  return model


##### IMPORTATION DES LIBRAIRIES
import numpy as np
import pandas as pds

##### MISE EN PLACE DES VARIABLES DE LANCEMENT
episode = 15
investissement_disponible = 150000
batch_size = 32
valeur_portefeuil = []

##### CREATION DE LA DATASET EN LA TRANSPOSANT
dataset = pds.read_excel("CAC40_udemy.xlsx") 
matrice = dataset[["NQ100", "SP500", "CAC40", "DJI30", "GOLD"]].dropna().reset_index(drop=True).transpose()
data = matrice.iloc[:,0:15000].values
data = np.around(data)


##### MISE EN PLACE DU MODELE
env = env_financier(data, investissement_disponible)
taille_etat = env.espace_observation.shape
taille_action = env.espace_action.n
agent = Agent(taille_etat, taille_action)
transformation = affiche_transformation(env)


##### LANCEMENT DU MODELE
if __name__ == "__main__":
  for t in range(episode):
    etat = env.reset()
    etat = transformation.transform([etat])
    for time in range(env.n_jour):
      action = agent.act(etat)
      prochain_etat, recompense, done, info = env._etape(action)
      prochain_etat = transformation.transform([prochain_etat])
      agent.souvenir(etat, action, recompense, prochain_etat, done)
      etat = prochain_etat
      if done:
        print("etape: {}/{}, etape et valeur: {}".format(
          t + 1, episode, info['valeur_actuelle']))
        valeur_portefeuil.append(info['valeur_actuelle']) 
        break
      if len(agent.memoire) > batch_size:
        agent.replay(batch_size)

##### TEST DU MODELE
data_test = matrice.iloc[:,15000:64975].values
data_test = np.around(data_test)
env = env_financier(data_test, investissement_disponible)
episode = 55 
if __name__ == "__main__":
  for t in range(episode):
    etat = env.reset()
    etat = transformation.transform([etat])
    for time in range(env.n_jour):
      action = agent.act(etat)
      prochain_etat, recompense, done, info = env._etape(action)
      prochain_etat = transformation.transform([prochain_etat])
      etat = prochain_etat
      if done:
        print("etape: {}/{}, etape et valeur: {}".format(
          t + 1, episode, info['valeur_actuelle']))
        valeur_portefeuil.append(info['valeur_actuelle']) 
        break