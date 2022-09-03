import sys
IN_COLAB = "google.colab" in sys.modules

import numpy as np
import tensorflow as tf
import gym
from tensorflow.keras.layers import Input, Dense

from threading import Thread
from multiprocessing import cpu_count
tf.keras.backend.set_floatx('float64')

GLOBAL_EP = 0

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

class Worker(Thread):
    def __init__(self, id, env, gamma, global_actor, global_critic):
        Thread.__init__(self)
        self.name = "w%i" % id
        
        self.env = env
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.gamma = gamma
        self.global_actor = global_actor
        self.global_critic = global_critic
        
        self.actor = Actor(self.state_size, self.action_size,
                           )
        self.critic = Critic(self.state_size)
        
        # sync local networks with global networks
        self.sync_with_global()
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        probs = self.actor.model.predict(state)
        return np.random.choice(self.action_size, p=probs[0])
    
    def n_step_td_target(self, rewards, next_Q, done):
        td_targets = np.zeros_like(rewards)
        R_to_go = 0
        
        if not done:
            R_to_go = next_Q
        
        for k in reversed(range(0, len(rewards))):
            R_to_go = rewards[k] + self.gamma * R_to_go 
            td_targets[k] = R_to_go
        return td_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch
    
    def sync_with_global(self):
        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())
    
    def run(self):
        global GLOBAL_EP
        while max_episodes > GLOBAL_EP:
        # for episode in range(max_episodes):
            episode_reward = 0
            done = False
            state = self.env.reset()
            
            states     = []
            actions    = []
            rewards    = []
            
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
                
                if len(states) >= update_interval or done:
                    states  = self.list_to_batch(states)
                    actions = self.list_to_batch(actions)
                    rewards = self.list_to_batch(rewards)
                    
                    curr_Qs = self.critic.model.predict(states)
                    next_Q = self.critic.model.predict(next_state)
                    
                    td_targets   = self.n_step_td_target(rewards, next_Q, done)
                    # advantages   = td_targets - self.critic.model.predict(states)
                    advantages   = td_targets - curr_Qs
                    
                    actor_loss = self.global_actor.train(states, actions, advantages)
                    critic_loss = self.global_critic.train(states, td_targets)

                    self.sync_with_global()
                    states     = []
                    actions    = []
                    rewards    = []

            print(self.name + ' | EP{} EpisodeReward={}'.format(GLOBAL_EP+1, episode_reward))
            GLOBAL_EP += 1

class A3CAgent:
    
    def __init__(self, env_name, gamma):
        env = gym.make(env_name)
        self.env_name = env_name
        self.gamma = gamma
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        self.global_actor = Actor(self.state_size, self.action_size,
                                 )
        self.global_critic = Critic(self.state_size)
        
        self.num_workers = cpu_count()
        
    def train(self):
        print("Training on {} cores".format(self.num_workers))
        input("Enter to start")
        self.workers = []

        for i in range(self.num_workers):
            env = gym.make(self.env_name)
            self.workers.append(Worker(
                i, env, self.gamma, self.global_actor, self.global_critic))
        
        # [worker.start() for worker in self.workers]
        # [worker.join() for worker in self.workers]
        
        for worker in self.workers:
            worker.start()

        for worker in self.workers:
            worker.join()
    
    # def save_model(self):
    #     self.global_critic.save("a3c_value_model.h5")
    #     self.global_actor.save("a3c_policy_model.h5")

if __name__ == "__main__":
    
    env_name = "CartPole-v0"
    # set environment
    actor_lr = 0.0005
    critic_lr = 0.001
    gamma = 0.99
    hidden_size = 128
    update_interval = 50
    
    max_episodes = 500  # Set total number of episodes to train agent on.
    agent = A3CAgent(env_name, gamma)
    agent.train()
    # agent.save_model()

