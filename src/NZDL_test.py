import tensorflow as tf

from pipeline import finetuning_pipeline

batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory('datasets/NZDL/train',
                                                       seed=123,
                                                       batch_size=batch_size)

validation_ds = tf.keras.utils.image_dataset_from_directory(
        'datasets/NZDL/valid', seed=123, batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory('datasets/NZDL/test',
                                                      seed=123,
                                                      batch_size=batch_size)

base_model = tf.keras.applications.MobileNet(include_top=False,
                                             weights='imagenet',
                                             input_shape=(224, 224, 3),
                                             input_tensor=None, pooling=None, )

finetuning_pipeline(base_model, train_ds, validation_ds, test_ds)
