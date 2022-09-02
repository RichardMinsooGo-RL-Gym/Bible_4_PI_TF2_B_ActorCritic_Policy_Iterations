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
    
    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(
            actions, logits, sample_weight=tf.stop_gradient(advantages))
        entropy = entropy_loss(logits, logits)
        return policy_loss - self.entropy_beta * entropy

    def train(self, states, actions, advantages):
        
        with tf.GradientTape() as tape:
            curr_P = self.model(states, training=True)
            loss = self.compute_loss(actions, curr_P, advantages)
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
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        probs = self.actor.model.predict(state)
        return np.random.choice(self.action_size, p=probs[0])
    
    def n_step_td_target(self, rewards, done):
        td_targets = np.zeros_like(rewards)
        R_to_go = 0
        
        for k in reversed(range(0, len(rewards))):
            R_to_go = rewards[k] + self.gamma * R_to_go 
            td_targets[k] = R_to_go
        return td_targets

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
            rewards = []

            while not done:
                action = self.get_action(state)
                
                next_state, reward, done, _ = self.env.step(action)
                
                state      = np.reshape(state, [1, self.state_size])
                action     = np.reshape(action, [1, 1])
                reward     = np.reshape(reward, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_size])
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state[0]
                episode_reward += reward[0][0]
                
                if done:
                    states  = self.list_to_batch(states)
                    actions = self.list_to_batch(actions)
                    rewards = self.list_to_batch(rewards)
                    
                    td_targets   = self.n_step_td_target(rewards, done)
                    actor_loss  = self.actor.train(states, actions, td_targets)

                    states     = []
                    actions    = []
                    rewards    = []


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

