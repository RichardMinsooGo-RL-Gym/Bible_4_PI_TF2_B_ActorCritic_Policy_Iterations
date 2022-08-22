import sys
IN_COLAB = "google.colab" in sys.modules

import gym
import numpy as np
import tensorflow as tf

from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
import tensorflow_probability as tfp

seed = 1234
env_name = "CartPole-v0"
# set environment
env = gym.make(env_name)
env.seed(seed)     # reproducible, general Policy gradient has high variance

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_size = 64
total_episodes = 500  # Set total number of episodes to train agent on.

class Network(Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(hidden_size,activation='relu')
        self.d2 = tf.keras.layers.Dense(hidden_size,activation='relu')
        self.out = tf.keras.layers.Dense(action_size,activation='softmax')

    def call(self, state):
        x = tf.convert_to_tensor(state)
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)
        return x

class PG_agent():
    def __init__(self):
        self.gamma = 1
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model = Network()
    
    def get_action(self, state):
        prob = self.model(np.array([state]))
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def compute_actor_loss(self,prob, action, reward): 
        
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*reward
        return loss

    def train_step(self, states, rewards, actions):
        sum_reward = 0
        discnt_rewards = []
        rewards.reverse()
        for r in rewards:
            sum_reward = r + self.gamma*sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()  

        for state, reward, action in zip(states, discnt_rewards, actions):
            with tf.GradientTape() as tape:
                p = self.model(np.array([state]), training=True)
                loss = self.compute_actor_loss(p, action, reward)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))


if __name__ == "__main__":
    tf.random.set_seed(336699)
    agent = PG_agent()

    episode = 0

    for episode in range(total_episodes):
        done = False
        state = env.reset()
        total_reward = 0
        states      = []
        actions     = []
        rewards     = []
        while not done:
            # env.render()
            action = agent.get_action(state)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
            total_reward += reward

            if done:
                agent.train_step(states, rewards, actions)
                #print("total step for this episord are {}".format(t))
                print("total reward after {} steps is {}".format(episode+1, total_reward))

