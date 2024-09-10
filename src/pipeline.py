import time
from datetime import datetime

import tensorflow as tf


def finetuning_pipeline(base_model, train_ds, validation_ds, test_ds, number_of_classes=None, top_layer_epochs=2, end_to_end_epochs=1):
    if number_of_classes is None:
        class_names = train_ds.class_names
        print("number of classes:", len(class_names), "classes:", class_names)
        number_of_classes = len(class_names)

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
    outputs = tf.keras.layers.Dense(number_of_classes)(
        x)  # units for classifying
    model = tf.keras.Model(inputs, outputs)

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True), metrics=['accuracy'], )

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

    checkpoint_filepath = 'ckpt/checkpoint.model.keras'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, monitor='val_accuracy', mode='max',
            save_best_only=True)

    print("Fitting the top layer of the model")
    start_time = time.time()

    model.fit(train_ds, epochs=top_layer_epochs, validation_data=validation_ds,
              callbacks=[tensorboard_callback, model_checkpoint_callback])
    print("Time taken: %.2fs" % (time.time() - start_time))

    # Unfreeze the base_model. Note that it keeps running in inference mode
    # since we passed `training=False` when calling it. This means that
    # the batchnorm layers will not update their batch statistics.
    # This prevents the batchnorm layers from undoing all the training
    # we've done so far.
    base_model.trainable = True
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True), metrics=['accuracy'], )

    print("Fitting the end-to-end model")
    start_time = time.time()
    model.fit(train_ds, epochs=end_to_end_epochs, validation_data=validation_ds,
              callbacks=[tensorboard_callback, model_checkpoint_callback])
    print("Time taken: %.2fs" % (time.time() - start_time))

    print("Test dataset evaluation")
    model.evaluate(test_ds)