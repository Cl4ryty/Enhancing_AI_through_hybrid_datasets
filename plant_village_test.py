import tensorflow as tf
import tensorflow_datasets as tfds


# load the dataset
(train_ds, validation_ds, test_ds), ds_info = tfds.load('plant_village',
                                             split=['train[:70%]',
                                                    'train[70%:90%]',
                                                    'train[90%:]'],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True,
                                             #data_dir="/tmp/tfds/", download=True,
                                                        )

batch_size = 32

# seems to be the maximum size supported by mobile net
resize_fn = tf.keras.layers.Resizing(224, 224)

train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
validation_ds = validation_ds.map(lambda x, y: (resize_fn(x), y))
test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))

train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
validation_ds = validation_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()

base_model = tf.keras.applications.MobileNet(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    input_tensor=None,
    pooling=None,
)


# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = tf.keras.Input(shape=(224, 224, 3))
# preprocess inputs (scaling to expected range of -1 to 1)
x = tf.keras.applications.mobilenet.preprocess_input(inputs)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = tf.keras.layers.Dense(38)(x)  # 28 units for classifying
model = tf.keras.Model(inputs, outputs)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
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
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
)

epochs = 1
print("Fitting the end-to-end model")
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

print("Test dataset evaluation")
model.evaluate(test_ds)