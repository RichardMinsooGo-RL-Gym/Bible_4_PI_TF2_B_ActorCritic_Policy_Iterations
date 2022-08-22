import sys
IN_COLAB = "google.colab" in sys.modules
import gym
import numpy as np
import tensorflow as tf

from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
import random


class Network(Model):
    def __init__(self, action_size):
        super(Network, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer_a1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer_c1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.poicy = tf.keras.layers.Dense(action_size, activation='softmax')
        self.value = tf.keras.layers.Dense(1)

    def call(self, state):
        hidden1 = self.hidden1(state)
        hidden2 = self.hidden2(hidden1)
        
        layer_a1 = self.layer_a1(hidden2)
        poicy = self.poicy(layer_a1)

        layer_c1 = self.layer_c1(hidden2)
        value = self.value(layer_c1)

        return poicy, value

class A2CAgent:
    def __init__(self, env):
        
        self.env = env
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.gamma = 0.99
        self.model = Network(self.action_size)
        
        self.opt = optimizers.Adam(lr=0.001, )
    
    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy, _ = self.model(state)
        policy = np.array(policy)[0]
        action = np.random.choice(self.action_size, p=policy)
        return action

    def train_step(self, state, action, reward, next_state, done):

        states      = tf.convert_to_tensor(state, dtype=tf.float32)
        actions     = tf.convert_to_tensor(action, dtype=tf.int32)
        rewards     = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype=tf.float32)
        dones       = tf.convert_to_tensor(done, dtype=tf.float32)
        
        model_variable = self.model.trainable_variables
        
        with tf.GradientTape() as tape:
            tape.watch(model_variable)
            _, curr_Qs = self.model(states)
            _, next_Qs = self.model(next_states)
            curr_Qs, next_Qs = tf.squeeze(curr_Qs), tf.squeeze(next_Qs)
            
            target_values = tf.stop_gradient(self.gamma * (1-dones) * next_Qs + rewards)
            critic_loss   = tf.reduce_mean(tf.square(target_values - curr_Qs) * 0.5)

            probs, _  = self.model(states)
            entropy = tf.reduce_mean(- probs * tf.math.log(probs+1e-8)) * 0.1
            onehot_action = tf.one_hot(actions, self.action_size)
            action_policy = tf.reduce_sum(onehot_action * probs, axis=1)
            adv = tf.stop_gradient(target_values - curr_Qs)
            actor_loss = -tf.reduce_mean(tf.math.log(action_policy+1e-8) * adv) - entropy

            loss = actor_loss + critic_loss
        grads = tape.gradient(loss, model_variable)
        self.opt.apply_gradients(zip(grads, model_variable))

    def train(self, max_episodes=1000):
        episode = 0

        for episode in range(max_episodes):
            episode_reward = 0
            done = False
            state = env.reset()
            all_loss = []
            
            states      = []
            actions     = []
            rewards     = []
            next_states = []
            dones       = []

            while not done:
                # env.render()
                action = self.get_action(state)
                # print(action)
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                episode_reward += reward

                if done:
                    # state, action, reward, next_state, done
                    self.train_step(
                        state=states, action=actions,
                        reward=rewards, next_state=next_states, done=dones)
                    print("total reward after {} steps is {}".format(episode+1, episode_reward))

if __name__ == "__main__":
    seed = 1234
    env_name = "CartPole-v0"
    # set environment
    max_episodes = 500  # Set total number of episodes to train agent on.
    env = gym.make(env_name)
    env.seed(seed)     # reproducible, general Policy gradient has high variance

    hidden_size = 64
    agent = A2CAgent(env)
    agent.train(max_episodes)


