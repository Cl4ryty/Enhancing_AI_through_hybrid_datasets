import tensorflow as tf

import tensorflow_datasets as tfds


# create data pipeline

def prepare_dataset(data):
    # convert data from uint8 to float32
    data = data.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    data = data.map(lambda img, target: ((img / 128.) - 1., target))
    # create one-hot targets
    data = data.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    # cache this progress in memory, as there is no need to redo it; it is deterministic after all
    data = data.cache()
    # shuffle, batch, prefetch
    data = data.shuffle(1000)
    data = data.batch(256)
    data = data.prefetch(tf.data.AUTOTUNE)
    # return preprocessed dataset
    return data

# load the dataset
(train_ds, validation_ds, test_ds), ds_info = tfds.load('plant_village',
                                             split=['train[:70%]',
                                                    'train[70%:90%]',
                                                    'train[90%:]'],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True, )

# # preprocessing the data
# train_ds = train_ds.apply(prepare_dataset)
# test_ds = test_ds.apply(prepare_dataset)

resize_fn = tf.keras.layers.Resizing(150, 150)

train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
validation_ds = validation_ds.map(lambda x, y: (resize_fn(x), y))
test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))

augmentation_layers = [
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
]


def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

batch_size = 64

train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
validation_ds = validation_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()


base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(150, 150, 3),
    pooling=None,
)

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = tf.keras.Input(shape=(150, 150, 3))

# Pre-trained Xception weights requires that input be scaled
# from (0, 255) to a range of (-1., +1.), the rescaling layer
# outputs: `(inputs * scale) + offset`
scale_layer = tf.keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(inputs)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)

model.summary(show_trainable=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

epochs = 2
print("Fitting the top layer of the model")
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)


# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
model.summary(show_trainable=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

epochs = 1
print("Fitting the end-to-end model")
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

print("Test dataset evaluation")
model.evaluate(test_ds)