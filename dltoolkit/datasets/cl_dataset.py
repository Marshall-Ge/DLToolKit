import numpy as np
from torch.utils.data import Dataset

class CLDataManager(object):
    """
    init_cls: number of classes in the first task
    increment: number of classes in the incremental tasks
    """
    def __init__(self, dataset, init_cls, increment, strategy):
        self.strategy = strategy
        self.dataset = dataset
        self._setup_data(dataset, strategy.config.data.cls_shuffle, strategy.config.seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
            
    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_dataset(self, indices):
        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                self._data, self._targets, low_range=idx, high_range=idx + 1
            )
            data.append(class_data)
            targets.append(class_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        return DummyDataset(data, targets)

    def _setup_data(self, dataset, shuffle, seed):

        # Data
        self._data, self._targets = dataset.data, dataset.label

        # Order
        order = [i for i in range(len(np.unique(self._targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = np.arange(len(order)).tolist()
        self._class_order = order

        # Map indices
        self._targets = _map_new_class_index(
            self._targets, self._class_order
        )

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return [x[i] for i in idxes.flatten().astype(int)], y[idxes]

    def getlen(self, index):
        y = self._targets
        return np.sum(np.where(y == index))

def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


class DummyDataset(Dataset):
    def __init__(self, datas, labels):
        assert len(datas) == len(labels), "Data size error!"
        self.datas = datas
        self.labels = labels

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        label = self.labels[idx]

        return {
            'data': data,
            'label': label,
        }