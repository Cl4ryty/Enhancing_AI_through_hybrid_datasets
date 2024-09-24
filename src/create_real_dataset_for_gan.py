from utils import load_tfrecord, write_tfrecord, undersample_majority_classes

datasets_to_use = ['../processed_datasets/real/train_real_filtered.tfrecord',
                   '../processed_datasets/real/validation_real_filtered.tfrecord']

# batching is done because image_dataset_from_directory batches automatically
# and this ensures proper shapes also for later use
batch_size = 32
majority_cutoff = 200

# load real dataset
dataset = load_tfrecord(datasets_to_use.pop(0))

# concatenate the datasets to train the GAN on all the data
while len(datasets_to_use) > 0:
    concatenate_dataset = load_tfrecord(datasets_to_use.pop(0))
    dataset = dataset.concatenate(concatenate_dataset)
    print("concatenated dataset")


dataset = undersample_majority_classes(dataset, majority_cutoff=majority_cutoff)

real_dataset = dataset.batch(batch_size)

# merge them and write them to another file
write_tfrecord(real_dataset, '../processed_datasets/gan_real_filtered.tfrecord',
               scale_image_back=False)
