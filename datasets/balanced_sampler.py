import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler


def class_balanced_sampler(labels, num_samples_per_class=10):
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    sample_weights = np.zeros_like(labels).astype(np.float32)
    for idx, label in enumerate(labels):
        sample_weights[idx] = total_samples / class_counts[label]
    # return sample_weights
    # sampler = WeightedRandomSampler(weights=sample_weights,
    #     num_samples=total_samples)

    # mimic test distribution
    num_samples = len(class_counts) * num_samples_per_class
    sampler = WeightedRandomSampler(weights=sample_weights,
        num_samples=num_samples)
    return sampler