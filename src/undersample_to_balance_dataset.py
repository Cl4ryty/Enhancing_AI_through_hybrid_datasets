import argparse

import numpy as np

from utils import load_tfrecord, write_tfrecord, undersample_dataset


# Define the default values
minority_threshold = 35
save_to = '../processed_datasets/train_balanced_undersampled.tfrecord'
train_dataset_path = '../processed_datasets/train1.tfrecord'
validation_dataset_path = '../processed_datasets/validation1.tfrecord'

# Set up the argument parser
parser = argparse.ArgumentParser(description="Script for processing datasets. Combines training and validation dataset, removes classes with fewer samples than the threshold and undersamples remaining classes to the minority sample count")
parser.add_argument('--minority_threshold', type=int, default=minority_threshold,
                    help='Threshold for minority classes. Classes with fewer samples are excluded (default: {})'.format(minority_threshold))
parser.add_argument('--save_to', type=str, default=save_to,
                    help='Path to save the undersampled (balanced) dataset to. This should be a file with the .tfrecord extension (default: {})'.format(save_to))
parser.add_argument('--train_dataset_path', type=str, default=train_dataset_path,
                    help='Path to the training dataset. This should be a file with the .tfrecord extension (default: {})'.format(train_dataset_path))
parser.add_argument('--validation_dataset_path', type=str, default=validation_dataset_path,
                    help='Path to the validation dataset. This should be a file with the .tfrecord extension (default: {})'.format(validation_dataset_path))

# Parse the command-line arguments
args = parser.parse_args()

print("Minority Threshold:", args.minority_threshold)
print("Path to Save Balanced Dataset:", args.save_to)
print("Training Dataset Path:", args.train_dataset_path)
print("Validation Dataset Path:", args.validation_dataset_path)

train_dataset = load_tfrecord(train_dataset_path)
validation_dataset = load_tfrecord(validation_dataset_path)

# concatenate the datasets to train the GAN on all the data
dataset = train_dataset.concatenate(validation_dataset)

final_dataset = undersample_dataset(dataset, minority_threshold)

# batch dataset as the write function expects a batched dataset
final_dataset = final_dataset.batch(32)

labels, counts = np.unique(np.fromiter(final_dataset.unbatch().map(lambda x, y: y), np.int32),
                           return_counts=True)
print("real dataset counts", np.sum(counts))
print(dict(zip(labels, counts)))

# Inspect the shapes of the final balanced dataset
for x, y in final_dataset.take(1):
    print(x.shape)
    print(y.shape)

# Save the balanced dataset for future use
# write_tfrecord(final_dataset, save_to, scale_image_back=False)
