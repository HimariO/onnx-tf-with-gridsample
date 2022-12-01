import onnx
from onnx_tf.backend import prepare

onnx_model_path = 'hdr_flownet.onnx'
onnx_model = onnx.load(onnx_model_path)

tf_model_path = 'tf_hdr_flownet'
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_path)

import tensorflow as tf

model = tf.saved_model.load(tf_model_path)
model.trainable = False

tensor_1 = tf.random.uniform([1, 3, 480, 640])
tensor_2 = tf.random.uniform([1, 3, 480, 640])
tensor_3 = tf.random.uniform([1, 3, 480, 640])
out = model(**{'frame_1': tensor_1, 'frame_2': tensor_2, 'frame_3': tensor_3})

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = []
converter.allow_custom_ops = True
tflite_model = converter.convert()

with open('hdr_flownet_dum.tflite', mode='wb') as f:
    f.write(tflite_model)
