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

    def create_model(self):
        state_input = Input((self.state_size,))
        dense_1 = Dense(hidden_size, activation='relu')(state_input)
        dense_2 = Dense(hidden_size, activation='relu')(dense_1)
        policy = Dense(self.action_size, activation='softmax')(dense_2)
        return tf.keras.models.Model(state_input, policy)
    
    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(
            actions, logits, sample_weight=tf.stop_gradient(advantages))
        return policy_loss

    def train(self, states, actions, advantages):
        
        with tf.GradientTape() as tape:
            curr_P = self.model(states, training=True)
            loss = self.compute_loss(actions, curr_P, advantages)
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

class A2CAgent():
    def __init__(self, env_name, gamma):
        
        self.env = gym.make(env_name)
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.gamma = gamma
        
        self.actor = Actor(self.state_size, self.action_size,
                           )
        self.critic = Critic(self.state_size)
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        probs = self.actor.model.predict(state)
        return np.random.choice(self.action_size, p=probs[0])
    
        

    def td_target(self, reward, next_state, done):
        if done:
            return reward
        v_value = self.critic.model.predict(
            np.reshape(next_state, [1, self.state_size]))
        return np.reshape(reward + self.gamma * v_value[0], [1, 1])

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

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
            
            states  = []
            actions = []
            td_targets = []
            advatnages = []

            while not done:
                action = self.get_action(state)
                
                next_state, reward, done, _ = self.env.step(action)
                
                state      = np.reshape(state, [1, self.state_size])
                action     = np.reshape(action, [1, 1])
                reward     = np.reshape(reward, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_size])
                
                td_target = self.td_target(reward * 0.01, next_state, done)
                advantage = self.advatnage(
                    td_target, self.critic.model.predict(state))
                
                states.append(state)
                actions.append(action)
                td_targets.append(td_target)
                advatnages.append(advantage)

                state = next_state[0]
                episode_reward += reward[0][0]
                
                if len(states) >= update_interval or done:
                    states  = self.list_to_batch(states)
                    actions = self.list_to_batch(actions)
                    td_targets = self.list_to_batch(td_targets)
                    advantages = self.list_to_batch(advatnages)
                    
                    actor_loss  = self.actor.train(states, actions, advantages)
                    critic_loss = self.critic.train(states, td_targets)

                    states     = []
                    actions    = []
                    td_targets = []
                    advatnages = []


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
    max_episodes = 500  # Set total number of episodes to train agent on.
    agent = A2CAgent(env_name, gamma)
    agent.train()
    # agent.save_model()

