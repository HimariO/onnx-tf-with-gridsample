import tensorflow as tf
import tensorflow_addons as tfa

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("GridSample")

class GridSample(BackendHandler):

  @classmethod
  def version_16(cls, node, **kwargs):

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
