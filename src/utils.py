import os
import re

import matplotlib.pyplot as plt
import tensorflow as tf


# save the combined datasets so that they can be directly loaded without having to do the preprocessing again
def write_tfrecord(dataset, filename, scale_image_back=True):
    writer = tf.io.TFRecordWriter(filename)
    i = 0

    for batch_features, batch_labels in dataset:
        # Loop through each image in the batch
        for features, label in zip(batch_features, batch_labels):
            # Scale the float image to the range [0, 255] and convert to uint8
            scaled_image = features
            if scale_image_back:
                scaled_image = tf.cast(features * 255, tf.uint8)
            if i < 5:
                plt.imshow(scaled_image)
                plt.show()
                i += 1
            # Encode the image as JPEG
            encoded_image = tf.io.encode_jpeg(scaled_image)

            # Create a tf.train.Example
            example = tf.train.Example(features=tf.train.Features(feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(
                            value=[encoded_image.numpy()])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[label.numpy()]))}))
            writer.write(example.SerializeToString())

    writer.close()


def parse_tfrecord(example_proto):
    feature_description = {'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64), }
    return tf.io.parse_single_example(example_proto, feature_description)


def load_tfrecord(filename):
    raw_dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = raw_dataset.map(parse_tfrecord)
    parsed_dataset = parsed_dataset.map(lambda x: (
    tf.image.decode_jpeg(x['image']), x['label']))  # Decode the image
    return parsed_dataset


def get_latest_checkpoint(checkpoint_dir):
    # List all files in the checkpoint directory
    files = os.listdir(checkpoint_dir)

    # Regular expression to match the checkpoint pattern and extract the epoch number
    pattern = re.compile(r"checkpoint-(\d+).weights.h5")

    # List to hold matching files with extracted epoch number
    checkpoint_files = []

    for file in files:
        match = pattern.match(file)
        if match:
            epoch = int(match.group(1))
            checkpoint_files.append((epoch, os.path.join(checkpoint_dir, file)))

    # Sort the checkpoint files by epoch number
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        # Get the latest checkpoint file path
        latest_checkpoint_path = checkpoint_files[0][1]
        return latest_checkpoint_path
    else:
        return None


from collections import defaultdict
import numpy as np
def undersample_dataset(dataset, minority_threshold):
    # Step 1: Count samples for each class
    class_counts = defaultdict(int)
    for x, y in dataset:
        label = y.numpy()
        class_counts[label] += 1

    print("Class distribution:", dict(class_counts))

    # Step 2: Set a threshold and discard classes below it
    classes_to_keep = [k for k, v in class_counts.items() if
                       v >= minority_threshold]

    print("Classes to keep:", sorted(classes_to_keep))

    # Step 3: Create a balanced dataset
    # Remember that the dataset is already parsed, we need to filter
    balanced_samples = defaultdict(list)

    # Load the dataset again and filter
    for x, y in dataset:
        label = y.numpy()
        if label in classes_to_keep:
            balanced_samples[label].append((x, y))

    # Step 4: Undersample each class to the minimum class size
    min_samples = min(len(samples) for samples in balanced_samples.values())
    print("number of samples to keep:", min_samples)
    balanced_dataset = []

    for samples in balanced_samples.values():
        # Shuffle samples and take `min_samples`
        np.random.shuffle(samples)
        balanced_dataset.extend(samples[:min_samples])

    # You can now shuffle the balanced_dataset before creating the final tf.data.Dataset
    np.random.shuffle(balanced_dataset)

    # Convert to a TensorFlow dataset
    final_dataset = tf.data.Dataset.from_tensor_slices(([x for (x, y) in
                                                         balanced_dataset],
                                                        [y for (x, y) in
                                                         balanced_dataset]))
    return final_dataset


import numpy as np
import tensorflow as tf
from collections import defaultdict


def undersample_majority_classes(dataset, majority_cutoff):
    # Step 1: Count samples for each class
    class_counts = defaultdict(int)
    for x, y in dataset:
        label = y.numpy()
        class_counts[label] += 1

    print("Class distribution:", dict(class_counts))

    # Step 2: Create samples dictionary for undersampling
    samples_to_keep = defaultdict(list)

    # Load the dataset again and filter
    for x, y in dataset:
        label = y.numpy()
        samples_to_keep[label].append((x, y))

    balanced_dataset = []

    for label, samples in samples_to_keep.items():
        if len(samples) > majority_cutoff:
            # Step 3: Undersample this class to the majority_cutoff
            print(f"undersampled class {label} which had {len(samples)} samples")
            np.random.shuffle(samples)
            balanced_dataset.extend(samples[:majority_cutoff])
        else:
            # Keep all samples for those below or equal to majority_cutoff
            balanced_dataset.extend(samples)

    # Shuffle the balanced dataset to mix classes
    np.random.shuffle(balanced_dataset)

    # Convert to a TensorFlow dataset
    final_dataset = tf.data.Dataset.from_tensor_slices(([x for (x, y) in
                                                         balanced_dataset],
                                                        [y for (x, y) in
                                                         balanced_dataset]))
    return final_dataset

