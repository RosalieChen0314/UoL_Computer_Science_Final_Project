import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# build DQN model
def build_dqn(state_size, num_actions, learning_rate=0.001):
    model = tf.keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        # output: Q(Hold), Q(Buy), Q(Sell)
        layers.Dense(num_actions, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, num_actions, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, replay_buffer_size):
        self.state_size = state_size
        self.num_actions = num_actions
        self.gamma = gamma # discount factor
        # epsilon-greedy parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (epsilon_decay)
        # replay buffer: each state is stored as a tuple (state, action, reward, state_next, done)
        # deque: FIFO queue --> when the buffer is full, the oldest record is removed
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        # build Q-network model
        self.model = build_dqn(state_size=state_size, num_actions=num_actions, learning_rate=learning_rate)
        # target Q-network model --> compute the target Q-values for training the main model
        self.target_model = build_dqn(state_size=state_size, num_actions=num_actions, learning_rate=learning_rate)
        self.update_target_model()

    # update target model weights
    # main model is updated every step, while target model is updated less frequently to stabilize training
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # save transition to replay buffer for transition replay
    def remember(self, state, action, reward, state_next, done):
        self.replay_buffer.append((state, action, reward, state_next, done))

    # choose action based on epsilon-greedy function
    def act(self, state, training=True):
        # episilon-greedy: with probability epsilon, choose a random action, otherwise choose the best action
        if (training and np.random.random() < self.epsilon):
            return random.randrange(self.num_actions)
        # model expects input shape of (batch_size, state_size), add an extra dimension to the state
        state_tensor = np.expand_dims(state, axis=0)
        action_rewards = self.model.predict(state_tensor, verbose=0)
        # return the action with the highest Q-value
        return int(np.argmax(action_rewards[0]))

    # replay episodes
    def replay(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None
        # sample a random minibatch of transitions from the replay buffer 
        minibatch = random.sample(self.replay_buffer, batch_size)
        # extract transition components (state, action, reward, state_next, done)  
        state_sample = np.array([transition[0] for transition in minibatch], dtype=np.float32)
        action_sample = np.array([transition[1] for transition in minibatch], dtype=np.int32)
        rewards_sample = np.array([transition[2] for transition in minibatch], dtype=np.float32)
        state_next_sample = np.array([transition[3] for transition in minibatch], dtype=np.float32)
        done_sample = np.array([transition[4] for transition in minibatch], dtype=np.float32)
            
        # current Q values (action rewards) for the current states
        current_action_rewards = (self.model.predict(state_sample,verbose=0))
        # next-state Q values
        future_rewards = (self.target_model.predict(state_next_sample,verbose=0))
        # get max Q value for next state
        max_future_rewards = np.max(future_rewards, axis=1)
        # copy current values
        updated_q_values = current_action_rewards.copy()

        # Bellman equation: Q(s, a) = r + gamma * max(Q(s', a'))
        for i in range(batch_size):
            action = action_sample[i]
            if done_sample[i]:
                target_value = (rewards_sample[i])
            else:
                target_value = (rewards_sample[i] + self.gamma * max_future_rewards[i])
            updated_q_values[i, action] = target_value

        # train Q network: calculate MSE loss between predicted Q-values and updated target Q-values
        loss = self.model.train_on_batch(state_sample, updated_q_values)
        return float(loss)

    # decay epsilon after each episode to reduce exploration over time
    def decay_epsilon(self):
        if (self.epsilon > self.epsilon_min):
            self.epsilon *= (self.epsilon_decay)
            self.epsilon = max(self.epsilon,self.epsilon_min)