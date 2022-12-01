import os
import tensorflow as tf
import tensorflow_addons as tfa

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func

lib = os.path.join(os.path.dirname(__file__), '../../../resampler.so')
module = tf.load_op_library(lib)
resampler = module.resampler

@onnx_op("GridSample")

class GridSample(BackendHandler):

  @classmethod
  def _version_16(cls, node, **kwargs):

    input_dict = kwargs["tensor_dict"]
    x = input_dict[node.inputs[0]]
    grid = input_dict[node.inputs[1]]
    
    x = tf.transpose(x, perm=[0, 2, 3, 1])
    shape_4d = tf.shape(grid)
    grid_3d = tf.reshape(grid, [shape_4d[0], shape_4d[1] * shape_4d[2], shape_4d[3]])
    
    y = tfa.image.interpolate_bilinear(x, grid_3d)
    y = tf.reshape(y, [shape_4d[0], shape_4d[1], shape_4d[2], -1])
    y = tf.transpose(y, perm=[0, 3, 1, 2])
    
    # print('-' * 100)
    # print(x)
    # print(grid)
    # print(y)
    # # print(shape_4d)
    # print('-' * 100)
    return [y]
  
  @classmethod
  def version_16(cls, node, **kwargs):
    """
    Use dummy node as placeholder, make manual edit tflite file easier
    """

    input_dict = kwargs["tensor_dict"]
    x = input_dict[node.inputs[0]]
    grid = input_dict[node.inputs[1]]
    
    x = tf.transpose(x, perm=[0, 2, 3, 1])
    # y = tf.multiply(x, tf.reduce_mean(grid, axis=-1, keepdims=True, name='gridsample_dum_1'), name='gridsample_dum_2')
    y = resampler(x, grid)
    y = tf.transpose(y, perm=[0, 3, 1, 2])
    
    return [y]
