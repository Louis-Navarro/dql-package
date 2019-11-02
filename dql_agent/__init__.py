import os
import random
import collections

import numpy as np
import tensorflow as tf


class DeepQNetwork:
    def __init__(self, input_shape, n_actions, hyper_params):
        self.params = hyper_params
        self.n_actions = n_actions
        self.model = self.create_model(input_shape)
        self.memory = collections.deque(maxlen=2000)
        self.load()
        self.trained = 0

    def create_model(self, input_shape):
        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.Dense(1024,
                                  activation='sigmoid',
                                  input_shape=(input_shape, )))

        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))

        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(2048, activation='sigmoid'))

        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(2048, activation='softmax'))

        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(self.n_actions))

        model.compile(tf.keras.optimizers.Adam(self.params['lr']), 'mse')
        return model

    def predict(self, state):
        if random.random() <= self.params['epsilon']:
            action = random.randint(0, self.n_actions - 1)
            return action

        state = tf.convert_to_tensor(np.expand_dims(state, 0))
        softmax_scores = self.model.predict(state)
        action_index = tf.random.categorical(softmax_scores, 1).numpy()
        return action_index[0, 0]

    def batch_training(self, batch_size, epochs):
        self.print_training()

        batch = np.array(random.sample(self.memory, batch_size))

        batch_states = np.array(batch[:, 0].tolist())
        batch_actions = batch[:, 1]
        batch_rewards = batch[:, 2]
        batch_next_states = np.array(batch[:, 3].tolist())

        batch_rewards += self.params['gamma'] * np.amax(
            self.model.predict(batch_next_states), axis=1)

        targets = self.model.predict(batch_states)
        for index in range(len(targets)):
            reward = batch_rewards[index]
            action = batch_actions[index]
            targets[index, action] = reward

        self.model.fit(batch_states, targets, epochs=epochs, verbose=1)

        if not self.trained % 50:
            self.save()

        if self.params['epsilon'] > self.params['epsilon_final']:
            self.params['epsilon'] *= self.params['epsilon_decay']

    def print_training(self):
        self.trained += 1
        print(f'{(len(str(self.trained)) + 40) * "-"}')
        print(f'        TRAINING FOR THE {self.trained} TIMES        ')
        print(f'{(len(str(self.trained)) + 40) * "-"}')

    def save(self):
        self.model.save(f'save_{self.trained}.h5')
        print('Saved model')

    def load(self):
        if os.path.isfile('save.h5'):
            self.model = tf.keras.models.load_model('save.h5')
            print('Loaded last model')
