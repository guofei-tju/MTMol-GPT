from typing import Iterator
import numpy as np
from operator import itemgetter


class MolRLReplayBuffer:
    def __init__(self, max_buffer_size=512, shuffle=False):
        self.max_buffer_size = max_buffer_size
        self.buffer = []
        self.start_index = 0
        self.shuffle = shuffle
        self.indices = np.arange(max_buffer_size)
        if shuffle:
            np.random.shuffle(self.indices)

    def update_batch(self, states1, states2, actions, action_log_probs, rewards):
        self.buffer.extend(zip(states1, states2, actions, action_log_probs, rewards))

    def __getitem__(self, index):
        return self.buffer[index]

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
        self.start_index = 0
        self.indices = np.arange(self.max_buffer_size)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def restart(self):
        self.start_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def iterate_sample(self, mini_batch_size) -> Iterator:
        """
        A mini batch iterator
        """
        for i in range(self.start_index, len(self.buffer), mini_batch_size):
            sampled_indices = self.indices[i:i + mini_batch_size]
            # get sampled batch
            self.start_index = i + mini_batch_size
            yield itemgetter(*sampled_indices)(self.buffer)
