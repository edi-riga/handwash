#
# This is the core code that does the actual training and classification.
# 

import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.layers import Layer

# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define parameters for the dataset loader.
# Adjust batch size according to the memory volume of your GPU;
batch_size = 32
img_width = 320
img_height = 240
IMG_SIZE = (img_height, img_width)
N_CHANNELS = 3
IMG_SHAPE = IMG_SIZE + (N_CHANNELS,)

N_CLASSES = 7

# get environmental variables that control the execution
model_name = os.getenv("HANDWASH_NN", "MobileNetV2")
num_trainable_layers = int(os.getenv("HANDWASH_NUM_LAYERS", 0))
num_epochs = int(os.getenv("HANDWASH_NUM_EPOCHS", 20))
# how many frames to concatenate as input to the TimeDistributed network?
num_frames = int(os.getenv("HANDWASH_NUM_FRAMES", 5))
suffix = os.getenv("HANDWASH_SUFFIX", "")
pretrained_model_path = os.getenv("HANDWASH_PRETRAINED_MODEL", "")
num_extra_layers = int(os.getenv("HANDWASH_EXTRA_LAYERS", 0))

# data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

def freeze_model(model):
    if num_trainable_layers == 0:
        for layer in model.layers:
            layer.trainable = False
        return False
    elif num_trainable_layers > 0:
        for layer in model.layers[:-num_trainable_layers]:
            layer.trainable = False
        for layer in model.layers[-num_trainable_layers:]:
            layer.trainable = True
        return True
    else:
        # num_trainable_layers negative, set all to trainable
        for layer in model.layers:
            layer.trainable = True
        return True


def get_preprocessing_function():
    if model_name == "MobileNetV2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    elif model_name == "InceptionV3":
        return tf.keras.applications.inception_v3.preprocess_input
    elif model_name == "Xception":
        return tf.keras.applications.xception.preprocess_input
    return None


def get_default_model():
    if model_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
    elif model_name == "InceptionV3":
        base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
    elif model_name == "Xception":
        base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet')
    else:
        print("Unknown model name", model_name)
        exit(-1)

    training = freeze_model(base_model)

    # Build the model
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = inputs
    x = data_augmentation(x)
    x = get_preprocessing_function()(x)
    x = base_model(x, training=training)
    x = tf.keras.layers.Flatten()(x)
    if num_extra_layers:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    for i in range(num_extra_layers):
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    print(model.summary())

    return model


# This also fits to Xception and InceptionV3! But maybe not MobileNetV3
class MobileNetPreprocessingLayer(Layer):
    def __init__(self, **kwargs):
        super(MobileNetPreprocessingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MobileNetPreprocessingLayer, self).build(input_shape)

    def call(self, x):
        return (x / 127.5) - 1.0

    def compute_output_shape(self, input_shape):
        return input_shape


def get_time_distributed_model():
    if model_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       pooling='avg',
                                                       weights='imagenet')
    elif model_name == "InceptionV3":
        base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       pooling='avg',
                                                       weights='imagenet')
    elif model_name == "Xception":
        base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    pooling='avg',
                                                    weights='imagenet')
    else:
        print("Unknown model name", model_name)
        exit(-1)

    training = freeze_model(base_model)


    # Build the base model
    single_frame_inputs = tf.keras.Input(IMG_SHAPE)
    x = single_frame_inputs
    x = data_augmentation(x)
    # use a custom layer because otherwise cannot be converted to tflite
    x = MobileNetPreprocessingLayer()(x)
    single_frame_outputs = base_model(x, training=training)
    single_frame_model = tf.keras.Model(single_frame_inputs, single_frame_outputs)

    # Build the time distributed model
    INPUT_SHAPE = (num_frames,) + IMG_SHAPE
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = inputs
    x = tf.keras.layers.TimeDistributed(single_frame_model)(x)
    x = tf.keras.layers.GRU(256)(x)
    for i in range(num_extra_layers):
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    print(model.summary())

    return model



def get_merged_model():
    if model_name == "MobileNetV2":
        rgb_base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                           include_top=False,
                                                           weights='imagenet')
        of_base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                          include_top=False,
                                                          weights='imagenet')
    elif model_name == "InceptionV3":
        rgb_base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                                           include_top=False,
                                                           weights='imagenet')
        of_base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                                          include_top=False,
                                                          weights='imagenet')
    elif model_name == "Xception":
        rgb_base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                                           include_top=False,
                                                           weights='imagenet')
        of_base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                                          include_top=False,
                                                          weights='imagenet')
    else:
        print("Unknown model name", model_name)
        exit(-1)

    training = freeze_model(rgb_base_model)
    freeze_model(of_base_model)

    # Build the model
    rgb_network_input = tf.keras.Input(shape=IMG_SHAPE)
    rgb_network = data_augmentation(rgb_network_input)
    rgb_network = get_preprocessing_function()(rgb_network)
    rgb_network = rgb_base_model(rgb_network, training=training)
    rgb_network = tf.keras.layers.Flatten()(rgb_network)
    rgb_network = tf.keras.Model(rgb_network_input, rgb_network)

    for layer in rgb_network.layers:
        layer._name = "rgb_" + layer.name

    of_network_input = tf.keras.Input(shape=IMG_SHAPE)
    of_network = data_augmentation(of_network_input)
    of_network = get_preprocessing_function()(of_network)
    of_network = of_base_model(of_network, training=training)
    of_network = tf.keras.layers.Flatten()(of_network)
    of_network = tf.keras.Model(of_network_input, of_network)

    for layer in of_network.layers:
        layer._name = "of_" + layer.name

    merged = tf.keras.layers.concatenate([rgb_network.output, of_network.output], axis=1)
    merged = tf.keras.layers.Flatten()(merged)
    # XXX: should add a pooling layer here?!
    for i in range(num_extra_layers):
        merged = tf.keras.layers.Dense(128, activation='relu')(merged)
        merged = tf.keras.layers.Dropout(0.2)(merged)
    merged = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(merged)

    model = tf.keras.Model([rgb_network.input, of_network.input], merged)
    print(model.summary())

    return model


def fit_model(name, model, train_ds, val_ds, test_ds, weights_dict):
    # callbacks to implement early stopping and saving the model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(monitor='val_accuracy', mode='max',
                         verbose=1, save_freq='epoch',
                         filepath=name+'.{epoch:02d}-{val_accuracy:.2f}.h5')

    print("fitting the model...")
    history = model.fit(train_ds,
                        epochs=num_epochs,
                        validation_data=val_ds,
                        class_weight=weights_dict,
                        callbacks=[es]) # add mc to save after each epoch

    model.save(name + "final-model")

    # visualise accuracy
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.figure(figsize=(8, 8))
    plt.grid(True, axis="y")
    plt.subplot(2, 1, 1)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')
    plt.savefig("accuracy-{}.pdf".format(name), format="pdf")

    measure_performance("validation", name, model, val_ds)
    del val_ds

    test_loss, test_accuracy = model.evaluate(test_ds)
    result_str = 'Test loss: {} accuracy: {}\n'.format(test_loss, test_accuracy)
    print(result_str)
    with open("results-{}.txt".format(name), "a+") as f:
        f.write(result_str)

    measure_performance("test", name, model, test_ds)


def measure_performance(ds_name, name, model, ds, num_classes=N_CLASSES):
    matrix = [[0] * num_classes for i in range(num_classes)]

    y_predicted = []
    y_true = []
    n = 0
    for images, labels in ds:
        predicted = model.predict(images)
        for y_p, y_t in zip(predicted, labels):
            y_predicted.append(int(np.argmax(y_p)))
            y_true.append(int(np.argmax(y_t)))
            n += 1
        gc.collect()

    for y_p, y_t in zip(y_predicted, y_true):
        matrix[y_t][y_p] += 1

    print("Confusion matrix:")
    for row in matrix:
        print(row)

    f1_scores = []
    for i in range(num_classes):
        total = sum(matrix[i])
        true_predictions = matrix[i][i]
        total_predictions = sum([matrix[j][i] for j in range(num_classes)])
        if total:
            precision = true_predictions / total
        else:
            precision = 0
        if total_predictions:
            recall = true_predictions / total_predictions
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        print("{} precision={:.2f}% recall={:.2f}% f1={:.2f}".format(i, 100 * precision, 100 * recall, f1))
        f1_scores.append(f1)
    s = "Average {} F1 score: {:.2f}\n".format(ds_name, np.mean(f1_scores))
    print(s)
    with open("results-{}.txt".format(name), "a+") as f:
       f.write(s)


def evaluate(name, train_ds, val_ds, test_ds, weights_dict={}, model=None):
    name_with_suffix = name + suffix

    if len(pretrained_model_path):

        # load and use a pre-trained model
        custom_objects = {"MobileNetPreprocessingLayer": MobileNetPreprocessingLayer}
        base_model = tf.keras.models.load_model(pretrained_model_path, custom_objects)
        print("pretrained model loaded!")
        training = freeze_model(base_model)
        inputs = tf.keras.Input(shape=base_model.layers[0].get_output_at(0).get_shape().as_list()[1:])
        # run in inference mode
        outputs = base_model(inputs, training=training)
        model = tf.keras.Model(inputs, outputs)
        # always train the top layer
        model.layers[-1].trainable = True

        if "kaggle" in pretrained_model_path:
            name_with_suffix += "-pretrained-kaggle"
        elif "mitc" in pretrained_model_path:
            name_with_suffix += "-pretrained-mitc"
        elif "pskus" in pretrained_model_path:
            name_with_suffix += "-pretrained-pskus"
    else:
        # create a new model
        if model is None:
            model = get_default_model()

    model.compile(optimizer='Adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    if num_extra_layers:
        name += "-extralayers" + str(num_extra_layers)

    # clear the results file
    with open("results-{}.txt".format(name), "a+") as f:
        pass

    if len(pretrained_model_path):
        test_loss, test_accuracy = model.evaluate(test_ds)
        result_str = 'Test loss: {} accuracy: {}\n'.format(test_loss, test_accuracy)

        # evaluate the pre-trained model before the additional training
        measure_performance("test-before-retraining", name_with_suffix, model, test_ds)

    fit_model(name_with_suffix, model, train_ds, val_ds, test_ds, weights_dict)
