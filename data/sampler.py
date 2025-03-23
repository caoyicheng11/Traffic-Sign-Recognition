import math
import random
import torch
import torch.utils.data as tordata

class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

    def __iter__(self):
        while True:
            sample_indices = []
            pid_list = random.sample(self.dataset.label_set, self.batch_size[0])

            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]
                indices = random.sample(indices, k=self.batch_size[1])
                sample_indices += indices

            if self.batch_shuffle:
                sample_indices = random.sample(sample_indices, len(sample_indices))

            yield sample_indices

    def __len__(self):
        return len(self.dataset)

class InferenceSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        indices = list(range(self.size))

        if batch_size != 1:
            complement_size = math.ceil(self.size / batch_size) * \
                batch_size
            indices += indices[:(complement_size - self.size)]
            self.size = complement_size

        self.batches = [
            indices[i*self.batch_size : (i+1)*self.batch_size] 
            for i in range(self.size // self.batch_size)
        ]

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.dataset)