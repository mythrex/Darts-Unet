import numpy as np
import torch
from tqdm import tqdm
import os


class DataLoader():
    """Loads the data from given path

    Methods:
        make_queue : returns a zipped queue of data (batch_size, X, Y)
    """

    def __init__(self, x_path, y_path, batch_size=1, shuffle=True):
        """Initializes the Dataloader
        X are of shape (batch_size, W, H, channels)
        Y are of shape (batch_size, W, H, channels)

        Args:
            x_path (string): Path for X
            y_path (string): Path for Y
            batch_size (int, optional): Batch Size. Defaults to 1.
            shuffle (bool, optional): Shuffle. Defaults to True.
        """
        self.batch_size = batch_size
        self._x_path = x_path
        self._y_path = y_path
        self.shuffle = shuffle

    def _load_data(self):
        x = np.load(self._x_path)
        x = x.swapaxes(2, 3).swapaxes(1, 2)
        x = x.astype('float')
        # because mask is B/W
        # y = np.load(self._y_path)
        # y = y.astype('float')
        # y = y.swapaxes(2, 3).swapaxes(1, 2)
        # y = y / 255

        indices = list(range(len(x)))
        y = np.random.randint(0, 10, x.shape[0])
        # shuffle the array
        if self.shuffle:
            np.random.shuffle(indices)

        self.X = []
        self.Y = []
        # for i in tqdm(range(len(x) - self.batch_size + 1)):
        for i in tqdm(range(10)):
            self.X.append(x[indices[i: i + self.batch_size]])
            self.Y.append(y[indices[i: i + self.batch_size]])

    def make_queue(self):
        """Make the queue for data

        Returns:
            [zip]: zip object (X, Y)
        """
        # * data.npy contains [(X, Y)......]
        # where X has shape (batch_size, W, H, C)
        # where Y has shape (batch_size, W, H, C)
        # if not os.path.exists('./data/sac/data.npy'):
        self._load_data()
        #     data = list(zip(self.X, self.Y))
        #     if not os.path.exists('./data/sac'):
        #         os.mkdir('./data/sac')
        #     np.save('./data/sac/data.npy', np.array(data))
        # else:
        #     data = np.load('./data/sac/data.npy')
        return list(zip(self.X, self.Y))
