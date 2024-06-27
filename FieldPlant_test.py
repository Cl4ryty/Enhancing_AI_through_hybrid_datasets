import tensorflow as tf

batch_size = 32

import tensorflow as tf
import json
from google.protobuf.json_format import MessageToJson
import numpy as np

raw_dataset = tf.data.TFRecordDataset("datasets/FieldPlant/train/leaves.tfrecord")
for d in raw_dataset.take(1):
    ex = tf.train.Example()
    ex.ParseFromString(d.numpy())
    m = json.loads(MessageToJson(ex))
    print(m['features']['feature'].keys())

    result = {}
    # example.features.feature is the dictionary
    for key, feature in ex.features.feature.items():
        # The values are the Feature objects which contain a `kind` which contains:
        # one of three fields: bytes_list, float_list, int64_list
        print(key, feature.WhichOneof('kind'))
        kind = feature.WhichOneof('kind')
        result[key] = np.array(getattr(feature, kind).value)

    #print(result)

# Create a description of the features.
feature_description = {
    'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'image/object/class/label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/object/class/text': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)
parsed_dataset


# seems to be the maximum size supported by mobile net
resize_fn = tf.keras.layers.Resizing(224, 224)

train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
validation_ds = validation_ds.map(lambda x, y: (resize_fn(x), y))
test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))


train_ds = train_ds.prefetch(tf.data.AUTOTUNE).cache()
validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE).cache()
test_ds = test_ds.prefetch(tf.data.AUTOTUNE).cache()

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
outputs = tf.keras.layers.Dense(28)(x)  # 28 units for classifying
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