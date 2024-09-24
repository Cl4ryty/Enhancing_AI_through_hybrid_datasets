import numpy as np
import tensorflow as tf
from utils import load_tfrecord, write_tfrecord



datasets_to_use = ['../processed_datasets/train1.tfrecord',
                   '../processed_datasets/validation1.tfrecord',
                   '../processed_datasets/test1.tfrecord']

new_paths = ['../processed_datasets/train_real_filtered.tfrecord',
               '../processed_datasets/validation_real_filtered.tfrecord',
               '../processed_datasets/test_real_filtered.tfrecord']

classes_to_exclude = [0, 6, 7, 10, 12, 13, 16, 17, 22, 25]

# batching is done because image_dataset_from_directory batches automatically
# and this ensures proper shapes also for later use
batch_size = 32


# Define a filter function
def filter_function(image, label):
    # Convert exclusion_labels from a Python set to a tensor
    exclusion_tensor = tf.constant(list(classes_to_exclude), dtype=tf.int64)  # Match the dtype to int64
    # Check if the label is not in the exclusion tensor
    return tf.reduce_all(tf.math.logical_not(tf.equal(label, exclusion_tensor)))


for i, dataset_path in enumerate(datasets_to_use):
    # load real dataset
    dataset = load_tfrecord(dataset_path)

    # get labels and counts for all classes
    labels, counts = np.unique(
            np.fromiter(dataset.map(lambda x, y: y), np.int32),
            return_counts=True)

    print("before filtering")
    print(list(zip(labels, counts)))
    print("total number of samples", np.sum(counts))
    print()

    # use the filter method
    filtered_dataset = dataset.filter(filter_function)


    # get labels and counts for all classes
    labels, counts = np.unique(
        np.fromiter(filtered_dataset.map(lambda x, y: y), np.int32), return_counts=True)

    print("new dataset:", new_paths[i])
    print(dict(zip(labels, counts)))
    print("total number of samples", np.sum(counts))
    print()


    filtered_dataset = filtered_dataset.batch(batch_size)

    # write to new file
    write_tfrecord(filtered_dataset, new_paths[i],
                   scale_image_back=False)
