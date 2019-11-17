"""
This module allows you to use deep-q-learning easily.
"""
import os
import random
import collections

import numpy as np
import tensorflow as tf


class DeepQNetwork:
    """A object used in deep-q-learning to interact with the model.

    Parameters
    ----------
    input_shape : int
        The number of inputs for the neural network, must be an integer greater
        than 0.
    n_actions : int
        The number of actions the ai can take, e.g: go left, go right...
    hyper_params : dict
        Dictionnary containing all the hyper parameters:
            lr: float
                The learning rate used to train the model, usually 0.001.
            gamma: float
                Percentage of the reward of the next state, usually between
                0.9 and 1.
            epsilon: float
                Percentage of chance of taking a random action instead of using
                the model, usually 1 and the beginning.
            epsilon_decay: float
                Percentage of the epsilon to keep at each step, usually ~0.99.
            epsilon_final: float
                The minimum value of epsilon, if this value is reached, the
                epsilon is not going to change anymore.

    Attributes
    ----------
    params : dict
        Contains all of the value of hyper_params.
    model : Sequential
        The neural network.
    create_model : method
        Method that creates a model and returns it.
    memory : deque
        Deque that contains a list of state, action, reward, and the next
        state, which is used to train the model.
    load : method
        Method that loads the last saved model.
    trained : int
        The number of times the model was trained.
    n_actions: int
        The number of actions the ai can take.
    """

    def __init__(self, input_shape, n_actions, hyper_params):
        self.params = hyper_params
        self.n_actions = n_actions
        self.model = self.create_model(input_shape)
        self.memory = collections.deque(maxlen=2000)
        self.load()
        self.trained = 0

    def create_model(self, input_shape):
        """Method that creates a model compiled it with the Adam optimizer and
        the MSE loss function.

        Parameters
        ----------
        input_shape : int
            The number of inputs for the neural network, must be an integer
            greater than 0.

        Returns
        -------
        Sequential
            Returns the model

        """
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
        """Method that return the index of the action to take, either a random
        integer if a random number if smaller than or equal to the epsilon, or
        uses the model to predict the action.

        Parameters
        ----------
        state : array-like
            A array-like object of shape (B, C) or (C) where B is the batch
            size and C the number of inputs.

        Returns
        -------
        int
            Returns an integer greater than 0 and smaller than the number of
            actions

        """
        if random.random() <= self.params['epsilon']:
            action = random.randint(0, self.n_actions - 1)
            return action

        state = tf.convert_to_tensor(np.expand_dims(state, 0))
        softmax_scores = self.model.predict(state)
        action_index = tf.random.categorical(softmax_scores, 1).numpy()
        return action_index[0, 0]

    def batch_training(self, batch_size, epochs):
        """Method called to train the model on random samples in the memory.

        Parameters
        ----------
        batch_size : int
            The number corresponding to the amount of samples on which the
            model will be trained.
        epochs : int
            The number of epochs to train the model on the samples.

        Returns
        -------
        None
            This method doesn't return anything, it just trains the model.

        """
        self._print_training()

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

    def _print_training(self):
        """This methods print how many times it trained the model

        Returns
        -------
        None
            This method doesn't return anything, it just prints

        """
        self.trained += 1
        print(f'{(len(str(self.trained)) + 40) * "-"}')
        print(f'        TRAINING FOR THE {self.trained} TIMES        ')
        print(f'{(len(str(self.trained)) + 40) * "-"}')

    def save(self):
        """Methods that saves the current model in the save.h5 file.

        Returns
        -------
        None
            This method doesn't return anything, it just saves the model

        """
        self.model.save(f'save.h5')
        print('Saved model')

    def load(self):
        """Load the last saved model if it exists.

        Returns
        -------
        None
            This method doesn't return anything it just loads the model

        """
        if os.path.isfile('save.h5'):
            self.model = tf.keras.models.load_model('save.h5')
            print('Loaded last model')
