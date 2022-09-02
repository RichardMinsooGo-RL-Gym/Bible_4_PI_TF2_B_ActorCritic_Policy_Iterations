import sys
IN_COLAB = "google.colab" in sys.modules

import numpy as np
import tensorflow as tf
import gym
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate
import random
from collections import deque

tf.keras.backend.set_floatx('float64')

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.buffer)

class Actor(tf.keras.Model):
    
    def __init__(self, state_size, action_size, action_bound):
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_size,)),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='tanh'),
            Lambda(lambda x: x * self.action_bound)
        ])

    def train(self, states, q_grads):
        
        with tf.GradientTape() as tape:
            grads = tape.gradient(self.model(states), self.model.trainable_variables, -q_grads)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def predict(self, state):
        return self.model.predict(state)

class Critic(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(critic_lr)

    def create_model(self):
        state_input = Input((self.state_size,))
        s1 = Dense(32, activation='relu')(state_input)
        s2 = Dense(32, activation='relu')(s1)
        action_input = Input((self.action_size,))
        a1 = Dense(32, activation='relu')(action_input)
        c1 = concatenate([s2, a1], axis=-1)
        c2 = Dense(16, activation='relu')(c1)
        output = Dense(1, activation='linear')(c2)
        return tf.keras.Model([state_input, action_input], output)
    
    def predict(self, inputs):
        return self.model.predict(inputs)
    
    def q_grads(self, states, actions):
        actions = tf.convert_to_tensor(actions)
        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_values = self.model([states, actions])
            q_values = tf.squeeze(q_values)
        return tape.gradient(q_values, actions)

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model([states, actions], training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

class Agent:
    def __init__(self, env_name, gamma):
        
        self.env = gym.make(env_name)
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        
        self.gamma = gamma
        self.buffer = ReplayBuffer()

        self.actor = Actor(self.state_size, self.action_size, self.action_bound)
        self.critic = Critic(self.state_size, self.action_size)
        
        self.target_actor = Actor(self.state_size, self.action_size, self.action_bound)
        self.target_critic = Critic(self.state_size, self.action_size)

        actor_weights = self.actor.model.get_weights()
        critic_weights = self.critic.model.get_weights()
        self.target_actor.model.set_weights(actor_weights)
        self.target_critic.model.set_weights(critic_weights)
        
    
    def target_update(self):
        actor_weights = self.actor.model.get_weights()
        t_actor_weights = self.target_actor.model.get_weights()
        critic_weights = self.critic.model.get_weights()
        t_critic_weights = self.target_critic.model.get_weights()

        for i in range(len(actor_weights)):
            t_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * t_actor_weights[i]

        for i in range(len(critic_weights)):
            t_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * t_critic_weights[i]
        
        self.target_actor.model.set_weights(t_actor_weights)
        self.target_critic.model.set_weights(t_critic_weights)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        probs =  self.actor.model.predict(state)[0]
        return probs

    def td_target(self, rewards, q_values, dones):
        targets = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = gamma * q_values[i]
        return targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch
    
    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho * (mu-x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)
    
    def train_step(self):
        for _ in range(10):
            states, actions, rewards, next_states, dones = self.buffer.sample()
            target_q_values = self.target_critic.predict([next_states, self.target_actor.predict(next_states)])
            td_targets = self.td_target(rewards, target_q_values, dones)
            
            self.critic.train(states, actions, td_targets)
            
            s_actions = self.actor.predict(states)
            s_grads = self.critic.q_grads(states, s_actions)
            grads = np.array(s_grads).reshape((-1, self.action_size))
            self.actor.train(states, grads)
            self.target_update()

    def train(self):
        episode = 0
        while max_episodes > episode:
        # for episode in range(max_episodes):
            episode_reward = 0
            done = False
            state = self.env.reset()
            bg_noise = np.zeros(self.action_size)

            while not done:
                action = self.get_action(state)
                noise = self.ou_noise(bg_noise, dim=self.action_size)
                action = np.clip(action + noise, -self.action_bound, self.action_bound)
                
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, (reward+8)/8, next_state, done)
                bg_noise = noise
                state = next_state
                episode_reward += reward
            if self.buffer.size() >= batch_size and self.buffer.size() >= train_start:
                self.train_step()                

            print('EP{} EpisodeReward={}'.format(episode+1, episode_reward))
            episode += 1

if __name__ == "__main__":
    
    env_name = "Pendulum-v1"
    # set environment
    actor_lr = 0.0005
    critic_lr = 0.001
    gamma = 0.99
    hidden_size = 128
    batch_size = 64
    tau = 0.05
    train_start = 2000
    max_episodes = 500  # Set total number of episodes to train agent on.
    agent = Agent(env_name, gamma)
    agent.train()


