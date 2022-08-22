import sys
IN_COLAB = "google.colab" in sys.modules

import gym
import numpy as np
import tensorflow as tf

from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
import tensorflow_probability as tfp

env_name = "CartPole-v0"
# set environment
env = gym.make(env_name)
env.seed(1)     # reproducible, general Policy gradient has high variance

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_size = 64
total_episodes = 1000  # Set total number of episodes to train agent on.

class Network(Model):
    def __init__(self):
        super(Network, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer_a1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer_c1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.poicy = tf.keras.layers.Dense(action_size,activation=None)
        self.value = tf.keras.layers.Dense(1)

    def call(self, state):
        hidden1 = self.hidden1(state)
        hidden2 = self.hidden2(hidden1)
        
        layer_a1 = self.layer_a1(hidden2)
        poicy = self.poicy(layer_a1)

        layer_c1 = self.layer_c1(hidden2)
        value = self.value(layer_c1)
        return value, poicy
    
class AC_agent():
    def __init__(self):
        self.gamma = 0.99
        self.model = Network()
        
        self.opt = optimizers.Adam(lr=0.001, )
    
    def get_action(self, state):
        curr_Q, prob = self.model(np.array([state]))
        prob = tf.nn.softmax(prob)
        #print(prob)
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def compute_actor_loss(self, pred, action, td):
        pred = tf.nn.softmax(pred)
        dist = tfp.distributions.Categorical(probs=pred, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*td
        return loss

    def train_step(self, state, action, reward, next_state, done):
        state = np.array([state])
        next_state = np.array([next_state])
        
        model_variable = self.model.trainable_variables
        
        with tf.GradientTape() as tape:
            curr_Q, curr_pred =  self.model(state,training=True)
            next_Q, _ = self.model(next_state, training=True)
            td = reward + self.gamma*next_Q*(1-int(done)) - curr_Q
            actor_loss = self.compute_actor_loss(curr_pred,action,td)
            critic_loss = td**2
            loss = actor_loss + critic_loss
        grads = tape.gradient(loss, model_variable)
        self.opt.apply_gradients(zip(grads, model_variable))
        return loss


if __name__ == "__main__":
    agent = AC_agent()

    episode = 0

    for episode in range(total_episodes):
        done = False
        state = env.reset()
        total_reward = 0
        all_loss = []

        while not done:
            # env.render()
            action = agent.get_action(state)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            loss = agent.train_step(state, action, reward, next_state, done)
            all_loss.append(loss)

            state = next_state
            total_reward += reward

            if done:

                #print("total step for this episord are {}".format(t))
                print("total reward after {} steps is {}".format(episode+1, total_reward))

