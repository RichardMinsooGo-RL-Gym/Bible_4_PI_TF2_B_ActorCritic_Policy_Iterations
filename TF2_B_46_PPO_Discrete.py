import sys
IN_COLAB = "google.colab" in sys.modules

import numpy as np
import tensorflow as tf
import gym
from tensorflow.keras.layers import Input, Dense

tf.keras.backend.set_floatx('float64')

class Actor(tf.keras.Model):
    
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)
        self.entropy_beta = 0.01

    def create_model(self):
        state_input = Input((self.state_size,))
        dense_1 = Dense(hidden_size, activation='relu')(state_input)
        dense_2 = Dense(hidden_size, activation='relu')(dense_1)
        policy = Dense(self.action_size, activation='softmax')(dense_2)
        return tf.keras.models.Model(state_input, policy)
    
    def compute_loss(self, old_policy, new_policy, actions, gaes):
        gaes = tf.stop_gradient(gaes)
        old_log_p = tf.math.log(
            tf.reduce_sum(old_policy * actions))
        old_log_p = tf.stop_gradient(old_log_p)
        log_p = tf.math.log(tf.reduce_sum(
            new_policy * actions))
        ratio = tf.math.exp(log_p - old_log_p)
        clipped_ratio = tf.clip_by_value(
            ratio, 1 - clip_ratio, 1 + clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, old_policy, states, actions, gaes):
        actions = tf.one_hot(actions, self.action_size)
        actions = tf.reshape(actions, [-1, self.action_size])
        actions = tf.cast(actions, tf.float64)
        
        with tf.GradientTape() as tape:
            curr_P = self.model(states, training=True)
            loss = self.compute_loss(old_policy, curr_P, actions, gaes)
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
            Dense(hidden_size, activation='relu'),
            Dense(hidden_size, activation='relu'),
            # Dense(16, activation='relu'),
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

class PPOAgent():
    def __init__(self, env_name, gamma):
        
        self.env = gym.make(env_name)
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.gamma = gamma
        
        self.actor = Actor(self.state_size, self.action_size,
                           )
        self.critic = Critic(self.state_size)
    
    def gae_target(self, rewards, curr_Qs, next_Q, done):
        td_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        R_to_go = 0
        
        if not done:
            R_to_go = next_Q
        
        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + gamma * R_to_go - curr_Qs[k]
            gae_cumulative = gamma * lmbda * gae_cumulative + delta
            gae[k] = gae_cumulative
            R_to_go = curr_Qs[k]
            td_targets[k] = gae[k] + curr_Qs[k]
        return gae, td_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch
    
    def train(self):
        episode = 0
        while max_episodes > episode:
        # for episode in range(max_episodes):
            episode_reward = 0
            done = False
            state = self.env.reset()
            
            states      = []
            actions     = []
            rewards     = []
            old_policys = []
            
            while not done:
                probs = self.actor.model.predict(
                    np.reshape(state, [1, self.state_size]))
                action = np.random.choice(self.action_size, p=probs[0])
                
                next_state, reward, done, _ = self.env.step(action)
                
                state      = np.reshape(state, [1, self.state_size])
                action     = np.reshape(action, [1, 1])
                reward     = np.reshape(reward, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_size])
                
                states.append(state)
                actions.append(action)
                rewards.append(reward * 0.01)
                old_policys.append(probs)

                state = next_state[0]
                episode_reward += reward[0][0]
                
                if len(states) >= update_interval or done:
                    states  = self.list_to_batch(states)
                    actions = self.list_to_batch(actions)
                    rewards = self.list_to_batch(rewards)
                    old_policys = self.list_to_batch(old_policys)
                    
                    curr_Qs = self.critic.model.predict(states)
                    next_Q = self.critic.model.predict(next_state)
                    
                    gaes, td_targets = self.gae_target(
                        rewards, curr_Qs, next_Q, done)
                    
                    for epoch in range(epochs):
                        actor_loss = self.actor.train(
                            old_policys, states, actions, gaes)
                        critic_loss = self.critic.train(states, td_targets)

                    states     = []
                    actions    = []
                    rewards    = []
                    old_policys = []

            print('EP{} EpisodeReward={}'.format(episode+1, episode_reward))
            episode += 1

if __name__ == "__main__":
    
    env_name = "CartPole-v0"
    # set environment
    actor_lr = 0.0005
    critic_lr = 0.001
    gamma = 0.99
    hidden_size = 128
    update_interval = 50
    clip_ratio = 0.1
    lmbda = 0.95
    epochs = 7
    
    max_episodes = 500  # Set total number of episodes to train agent on.
    agent = PPOAgent(env_name, gamma)
    agent.train()
    # agent.save_model()

