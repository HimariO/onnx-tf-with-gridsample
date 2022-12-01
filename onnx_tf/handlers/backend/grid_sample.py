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
    
    print('-' * 100)
    print(type(x))
    print(x)
    print(grid)
    # print(shape_4d)
    print('-' * 100)

    # HACK: not sure how to determint input data format is been transpose outside of node or not
    c_last = 'transpose' in x.name.lower()
    if not c_last:
      x = tf.transpose(x, perm=[0, 2, 3, 1])
    shape_4d = tf.shape(grid)
    grid_3d = tf.reshape(grid, [shape_4d[0], shape_4d[1] * shape_4d[2], shape_4d[3]])
    
    y = tfa.image.interpolate_bilinear(x, grid_3d)
    y = tf.reshape(y, [shape_4d[0], shape_4d[1], shape_4d[2], -1])
    if not c_last:
      y = tf.transpose(y, perm=[0, 3, 1, 2])
    
    return [y]
