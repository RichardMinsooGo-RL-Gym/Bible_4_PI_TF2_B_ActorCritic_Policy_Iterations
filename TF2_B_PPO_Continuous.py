import sys
IN_COLAB = "google.colab" in sys.modules

import numpy as np
import tensorflow as tf
import gym
from tensorflow.keras.layers import Input, Dense, Lambda

tf.keras.backend.set_floatx('float64')

class Actor(tf.keras.Model):
    
    def __init__(self, state_size, action_size, action_bound, std_bound):
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)

    def create_model(self):
        state_input = Input((self.state_size,))
        dense_1 = Dense(32, activation='relu')(state_input)
        dense_2 = Dense(32, activation='relu')(dense_1)
        out_mu = Dense(self.action_size, activation='tanh')(dense_2)
        mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = Dense(self.action_size, activation='softplus')(dense_2)
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        mu, std = self.model.predict(state)
        action = np.random.normal(mu[0], std[0], size=self.action_size)
        action = np.clip(action, -self.action_bound, self.action_bound)
        log_policy = self.log_pdf(mu, std, action)

        return log_policy, action

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
        ratio = tf.exp(log_new_policy - tf.stop_gradient(log_old_policy))
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0-clip_ratio, 1.0+clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, log_old_policy, states, actions, gaes):
        
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            loss = self.compute_loss(
                log_old_policy, log_new_policy, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

class Critic(tf.keras.Model):

    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_size,)),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

class Agent():
    def __init__(self, env):
        
        self.env = env
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]
        
        self.actor = Actor(self.state_size, self.action_size,
                           self.action_bound, self.std_bound)
        self.critic = Critic(self.state_size)
    
    def gae_target(self, rewards, curr_Q, next_Q, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_Q
        
        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + gamma * forward_val - curr_Q[k]
            gae_cumulative = gamma * lmbda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = curr_Q[k]
            n_step_targets[k] = gae[k] + curr_Q[k]
        return gae, n_step_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch
    
    def train(self, max_episodes=1000):
        for episode in range(max_episodes):
            episode_reward = 0
            done = False
            state = self.env.reset()
            
            states  = []
            actions = []
            rewards = []
            old_policys = []
            
            while not done:
                # self.env.render()
                log_old_policy, action = self.actor.get_action(state)
                
                next_state, reward, done, _ = self.env.step(action)
                
                state  = np.reshape(state, [1, self.state_size])
                action = np.reshape(action, [1, 1])
                reward = np.reshape(reward, [1, 1])
                
                next_state = np.reshape(next_state, [1, self.state_size])
                
                log_old_policy = np.reshape(log_old_policy, [1, 1])
                
                states.append(state)
                actions.append(action)
                rewards.append((reward+8)/8)
                old_policys.append(log_old_policy)

                if len(states) >= update_interval or done:
                    states  = self.list_to_batch(states)
                    actions = self.list_to_batch(actions)
                    rewards = self.list_to_batch(rewards)
                    old_policys = self.list_to_batch(old_policys)
                    
                    curr_Q = self.critic.model.predict(states)
                    next_Q = self.critic.model.predict(next_state)
                    
                    gaes, td_targets = self.gae_target(
                        rewards, curr_Q, next_Q, done)

                    for epoch in range(epochs):
                        actor_loss = self.actor.train(
                            old_policys, states, actions, gaes)
                        critic_loss = self.critic.train(states, td_targets)

                    states     = []
                    actions    = []
                    rewards    = []
                    old_policys = []

                episode_reward += reward[0][0]
                state = next_state[0]

            print('EP{} EpisodeReward={}'.format(episode+1, episode_reward))

if __name__ == "__main__":
    
    env_name = 'Pendulum-v0'
    # set environment
    actor_lr = 0.0005
    critic_lr = 0.001
    gamma = 0.99
    update_interval = 50
    clip_ratio = 0.1
    lmbda = 0.95
    epochs = 3
    
    max_episodes = 500  # Set total number of episodes to train agent on.
    env = gym.make(env_name)
    agent = Agent(env)
    
    agent.train(max_episodes)


