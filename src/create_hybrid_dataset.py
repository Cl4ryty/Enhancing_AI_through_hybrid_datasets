import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf

from utils import load_tfrecord, write_tfrecord

# Define your default dataset paths
real_dataset_path = ('../processed_datasets/real/train_real_filtered.tfrecord')
synthetic_dataset_path = 'synthetic_100b'
new_hybrid_path = '../processed_datasets/0_100/train_0_100_balanced.tfrecord'

# batching is done because image_dataset_from_directory batches automatically
# and this ensures proper shapes also for later use
batch_size = 32
# load real dataset
real_dataset = load_tfrecord(real_dataset_path)

labels, counts = np.unique(np.fromiter(real_dataset.map(lambda x, y: y), np.int32),
                           return_counts=True)
print("original real dataset counts", np.sum(counts))
print(dict(zip(labels, counts)))

label_counts = {1: 134, 2: 160, 3: 160, 4: 60, 5: 156, 8: 41, 9: 109, 11: 34, 14: 160, 15: 116, 18: 160, 19: 161, 20: 38, 21: 161, 23: 161, 24: 161, 26: 80, 27: 66, 28: 37, 29: 40, 30: 38, 31: 59, 32: 53, 33: 122, 34: 132, 35: 102, 36: 47, 37: 66, 38: 33, 39: 46, 40: 72, 41: 35, 42: 62, 43: 65, 44: 102, 45: 142, 46: 77, 47: 42, 48: 161, 49: 101, 50: 50, 51: 82, 52: 71, 54: 94}

only_synthetic = True
should_sample_real_dataset = True
# Define the threshold
threshold = 200
invert_label_counts = False

if invert_label_counts:
    # Update the counts based on the threshold
    for label, count in label_counts.items():
        if count < threshold:
            label_counts[label] = threshold - count  # Change to threshold - current value



def sample_dataset(dataset, label_counts):
    # Create a dictionary to store the datasets for each class
    sampled_datasets = []

    for label, count in label_counts.items():
        # Filter the dataset to get only the samples for the current label
        filtered_dataset = dataset.filter(
            lambda image, lbl: tf.equal(lbl, label))

        # Randomly sample the specified count from the filtered dataset
        sampled = filtered_dataset.shuffle(buffer_size=1000).take(count)

        sampled_datasets.append(sampled)

    # Concatenate the sampled datasets for each class into a new dataset
    final_dataset = sampled_datasets[0]  # Start with the first dataset
    for sampled in sampled_datasets[1:]:
        final_dataset = final_dataset.concatenate(sampled)

    return final_dataset



# 3. Sample the dataset
if should_sample_real_dataset:
    real_dataset = sample_dataset(real_dataset, label_counts)

    labels, counts = np.unique(
        np.fromiter(real_dataset.map(lambda x, y: y), np.int32),
        return_counts=True)
    print("real dataset counts", np.sum(counts))
    print(dict(zip(labels, counts)))
    print(label_counts)

# 4. (Optional) Inspect the sampled dataset
for image, label in real_dataset.take(5):
    print("Label:", label.numpy())  # Print the sampled labels

real_dataset = real_dataset.batch(batch_size)


labels, counts = np.unique(np.fromiter(real_dataset.unbatch().map(lambda x, y: y), np.int32),
                           return_counts=True)
print("real dataset counts", np.sum(counts))
print(dict(zip(labels, counts)))

for x, y in real_dataset.take(1):
    print(x.shape, y.shape)


def load_image(file_path):
    # Load and preprocess the image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image,
                                  channels=3)  # Assuming 3 channels (RGB)
    return image


def create_dataset_from_directory(dataset_dir):
    class_labels = []
    image_paths = []
    labels = []

    # Walk through the directory to gather image paths and corresponding labels
    for class_dir in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_dir)
        if os.path.isdir(class_path):
            # Use the class directory name as label
            label = int(class_dir)  # Convert directory name to integer label

            # Find images in the directory
            image_files = [f for f in os.listdir(class_path) if
                           f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if image_files:
                # Add image paths and their corresponding labels
                for image_file in image_files:
                    image_paths.append(os.path.join(class_path, image_file))
                    labels.append(label)  # Collect labels separately

                # Record the class label for non-empty directories
                class_labels.append(label)


    print("1")
    # Create a TensorFlow Dataset from the image paths and labels
    image_paths_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    print("2")
    # Load images and labels
    image_label_ds = image_paths_ds.map(
        lambda file_path, label: (load_image(file_path), label),
        num_parallel_calls=tf.data.AUTOTUNE)

    return image_label_ds, sorted(set(class_labels))



# Create the dataset
synthetic_dataset, class_labels = create_dataset_from_directory(synthetic_dataset_path)

# Print class labels
print("Class Labels:", class_labels)

# Optional: inspect the dataset
for image, label in synthetic_dataset.take(5):  # Take the first 5 samples
    print("Label:", label.numpy(), "Image Shape:", image.shape)

labels, counts = np.unique(
    np.fromiter(synthetic_dataset.map(lambda x, y: y), np.int32),
    return_counts=True)
print("synthetic dataset counts", np.sum(counts))
print(dict(zip(labels, counts)))


synthetic_dataset = synthetic_dataset.map(
        lambda x, y: (tf.cast(x, tf.uint8), tf.cast(y, tf.int64)))

synthetic_dataset = synthetic_dataset.batch(batch_size)


for x, y in real_dataset.take(1):
    print(x.shape, y.shape)

dataset = real_dataset.concatenate(synthetic_dataset)

if only_synthetic:
    dataset = synthetic_dataset

labels, counts = np.unique(np.fromiter(dataset.unbatch().map(lambda x, y: y), np.int32),
                           return_counts=True)

print("full dataset counts", np.sum(counts))
print(dict(zip(labels, counts)))

# merge them and write them to another file
write_tfrecord(dataset, new_hybrid_path, scale_image_back=False)
