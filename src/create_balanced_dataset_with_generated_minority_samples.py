import numpy as np
from utils import (load_tfrecord, get_latest_checkpoint,
                       undersample_majority_classes)

import argparse
import ast

# Define the default values
minority_cutoff = 35  # Remove classes with fewer samples than this
majority_cutoff = 200  # Undersample classes with more samples than this
datasets_to_use = ['../processed_datasets/train1.tfrecord', '../processed_datasets/validation1.tfrecord']

# Set up the argument parser
parser = argparse.ArgumentParser(description="Script for creating a balanced dataset by removing classes with fewer samples than the minority cutoff, undersampling classes with more sampled than the majority cuttoff and generating synthetic examples to fill up classes with fewer samples.")
parser.add_argument('--minority_cutoff', type=int, default=minority_cutoff,
                    help='Cutoff to remove minority classes (default: {})'.format(minority_cutoff))
parser.add_argument('--majority_cutoff', type=int, default=majority_cutoff,
                    help='Cutoff to undersample majority classes (default: {})'.format(majority_cutoff))
parser.add_argument('--datasets_to_use', type=str, default=str(datasets_to_use),
                    help='List of datasets to use for training (default: {})'.format(datasets_to_use))

# Parse the command-line arguments
args = parser.parse_args()

# Convert the datasets_to_use string back to a list
datasets_to_use = ast.literal_eval(args.datasets_to_use)

print("Batch Size:", args.batch_size)
print("Number of Channels:", args.num_channels)
print("Number of Classes:", args.num_classes)
print("Image Size:", args.image_size)
print("Latent Dimension:", args.latent_dim)
print("Image Directory:", args.image_dir)
print("Checkpoint Directory:", args.checkpoint_dir)
print("Number of Epochs:", args.epochs)
print("Minority Cutoff:", args.minority_cutoff)
print("Majority Cutoff:", args.majority_cutoff)
print("Datasets to Use:", datasets_to_use)


dataset = load_tfrecord(datasets_to_use.pop(0))
# concatenate the datasets to train the GAN on all the data
while len(datasets_to_use)>0:
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

counts_to_generate = samples_to_reach-counts

generate_dictionary = dict(zip(labels, counts_to_generate))
print("average_to_generate",np.mean(counts_to_generate))

print("use to following dictionary to create synthetic data with the correct counts")
print(generate_dictionary)