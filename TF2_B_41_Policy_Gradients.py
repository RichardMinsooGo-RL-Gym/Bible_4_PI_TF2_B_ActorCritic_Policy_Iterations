import gym
import numpy as np
import tensorflow as tf

env_name = "CartPole-v1"
# set environment
env = gym.make(env_name)
env.seed(1)     # reproducible, general Policy gradient has high variance

gamma = 0.99
learning_rate = 0.01
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_size = 64
total_episodes = 500  # Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5
is_visualize = False

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class PolicyNetworks(tf.keras.Model):
    def __init__(self):
        super(PolicyNetworks, self).__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_size, activation='softmax')

    def call(self, x):
        H1_output = self.hidden_layer_1(x)
        outputs = self.output_layer(H1_output)

        return outputs

def pg_loss(outputs, actions, rewards):
    indexes = tf.range(0, tf.shape(outputs)[0]) * tf.shape(outputs)[1] + actions
    responsible_outputs = tf.gather(tf.reshape(outputs, [-1]), indexes)

    loss = -tf.reduce_mean(tf.math.log(responsible_outputs) * rewards)

    return loss

optimizer = tf.optimizers.Adam(learning_rate)

def train_step(model, states, actions, rewards):
    with tf.GradientTape() as tape:
        outputs = model(states)
        loss = pg_loss(outputs, actions, rewards)
    dqn_grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(dqn_grads, model.trainable_variables))

# Declare Policy Gradient Networks
PG_model = PolicyNetworks()

i = 0
total_reward = []
total_length = []

# train start
while i < total_episodes:
    s = env.reset()
    running_reward = 0
    ep_history = []

    for j in range(max_ep):
        if is_visualize == True:
            env.render()
        # Probabilistically pick an action given our network outputs.
        s = np.expand_dims(s, 0)
        a_dist = PG_model(s).numpy()
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)

        s1, r, d, _ = env.step(a)  # Get reward and next state
        ep_history.append([s, a, r, s1])
        s = s1
        running_reward += r

        if d == True:
            ep_history = np.array(ep_history)
            ep_history[:, 2] = discount_rewards(ep_history[:, 2])

            # Make state list to numpy array
            np_states = np.array(ep_history[0, 0])
            for idx in range(1, ep_history[:, 0].size):
                np_states = np.append(np_states, ep_history[idx, 0], axis=0)

            # Update the network parameter
            if i % update_frequency == 0 and i != 0:
                train_step(PG_model, np_states, ep_history[:, 1], ep_history[:, 2])

            total_reward.append(running_reward)
            total_length.append(j)
            break

    # Print last 100 episode's mean score
    if i % 100 == 0:
        print(np.mean(total_reward[-100:]))
    i += 1