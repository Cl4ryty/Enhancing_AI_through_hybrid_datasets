import tensorflow as tf

from pipeline import finetuning_pipeline

batch_size = 32

print("loading dataset")
(train_ds, validation_ds) = tf.keras.utils.image_dataset_from_directory(
        'datasets/PlantVillage', seed=123, batch_size=batch_size,
        validation_split=0.3, subset="both")
print("dataset loaded")

# Get validation batches (validation + test)
val_batches = tf.data.experimental.cardinality(validation_ds)

# Calculate the number of test batches (10% of original dataset)
test_ds = validation_ds.take(val_batches // 3)

# Use the remaining batches for the validation set (20% of the dataset)
validation_ds = validation_ds.skip(val_batches // 3)

base_model = tf.keras.applications.ResNet50(include_top=False,
        weights='imagenet', input_shape=(224, 224, 3), input_tensor=None,
        pooling=None, )

finetuning_pipeline(base_model, train_ds, validation_ds, test_ds)
