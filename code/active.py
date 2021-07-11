import random
import math


def random_active(candidate_dataset, aim_dataset, active_ratio, totality):
    length = len(candidate_dataset.samples)
    print(length)
    index = random.sample(range(length), round(totality * active_ratio))
    print(index)
    aim_dataset.add_item(candidate_dataset.samples[index])
    candidate_dataset.remove_item(index)


def uncertainty_active(candidate_dataset, aim_dataset, uncertainty_rank, current_acc, active_ratio, totality):
    length = len(uncertainty_rank)
    num_active = math.ceil(totality * active_ratio)

    print('current_acc: {}'.format(current_acc))
    start = round(current_acc * length)
    if length - start < num_active:
        start = length - num_active
    index = random.sample(range(start, length), num_active)
    print('range = {},  {}'.format(start, length))
    print(index)

    active_samples = uncertainty_rank[index, 0:2, ...]
    candidate_ds_index = uncertainty_rank[index, 2, ...]

    aim_dataset.add_item(active_samples)
    candidate_dataset.remove_item(candidate_ds_index)

    return active_samples
