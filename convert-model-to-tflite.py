#!/usr/bin/env python3

#
# This script shows how to export the model to a TFLite format, with the custom preprocessing object.
# 

import tensorflow as tf
from tensorflow.keras.layers import Layer
import sys

MODEL_NAME = "test"

if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]

if MODEL_NAME.endswith(".h5"):
    MODEL_NAME = MODEL_NAME[:-3]
    

class MobileNetPreprocessingLayer(Layer):
    def __init__(self, **kwargs):
        super(MobileNetPreprocessingLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(MobileNetPreprocessingLayer, self).build(input_shape)
        
    def call(self, x):
        return (x / 127.5) - 1.0
    
    def compute_output_shape(self, input_shape):
        return input_shape


custom_objects = {"MobileNetPreprocessingLayer": MobileNetPreprocessingLayer}


# convert to tensorflow lite format
model = tf.keras.models.load_model(MODEL_NAME + ".h5", custom_objects)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
   tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.float16]
tfmodel = converter.convert()
with open (MODEL_NAME + ".tflite" , "wb")  as f:
    f.write(tfmodel)
