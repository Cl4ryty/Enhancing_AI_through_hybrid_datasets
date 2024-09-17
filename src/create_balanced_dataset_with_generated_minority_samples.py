import numpy as np

from utils import (load_tfrecord, undersample_majority_classes)

# Define the default values
minority_cutoff = 35  # Remove classes with fewer samples than this
majority_cutoff = 200  # Undersample classes with more samples than this
datasets_to_use = ['../processed_datasets/train1.tfrecord',
                   '../processed_datasets/validation1.tfrecord']

dataset = load_tfrecord(datasets_to_use.pop(0))
# concatenate the datasets to train the GAN on all the data
while len(datasets_to_use) > 0:
    concatenate_dataset = load_tfrecord(datasets_to_use.pop(0))
    dataset = dataset.concatenate(concatenate_dataset)

# undersample classes with too many samples
dataset = undersample_majority_classes(dataset, majority_cutoff=majority_cutoff)

# get counts for each class:
labels, counts = np.unique(np.fromiter(dataset.map(lambda x, y: y), np.int32),
                           return_counts=True)

print(list(zip(labels, counts)))

# exclude classes with too few samples
mask = counts >= minority_cutoff
labels = labels[mask]
counts = counts[mask]

# get the maximum number of samples for a class - this is the number of samples all classes should have in the end
samples_to_reach = np.max(counts)
print("samples_to_reach", samples_to_reach)

counts_to_generate = samples_to_reach - counts

generate_dictionary = dict(zip(labels, counts_to_generate))
print("average_to_generate", np.mean(counts_to_generate))

print("use to following dictionary to create synthetic data with the correct counts")
print(generate_dictionary)
