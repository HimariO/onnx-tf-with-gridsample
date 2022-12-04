from typing import *
from collections import defaultdict, namedtuple

import tensorflow as tf
import onnx_tf.common as common
from onnx_tf.pb_wrapper import OnnxNode


FakeNode = namedtuple("FakeNode", ["op_type", "name"])

class AxisMapping:
  
  def __init__(self, tensor_dict: Dict[str, tf.Tensor]) -> None:
    self.tensor_dict = tensor_dict
    self.tensor_mapping = {
      'AveragePool': lambda a, b: b,
      'Pad': lambda a, b: b,
      'Concat': self.cat_tensor,
      'Transpose': self.transp_tensor,
      'Slice': lambda a, b: b,
    }
    self.attr_mapping = {
      'AveragePool': self.avg_pool_attr,
      'Pad': self.pad_attr,
      'Concat': self.cat_attr,
      'Transpose': self.transp_attr,
      'Slice': self.slice_attr,
    }
    self.visit = set()
  
  def mapping(self, node: OnnxNode, src_perm: List[int]) -> List[int]:
    if node.name in self.visit:
      return src_perm
    self.visit.add(node.name)
    if node.op_type not in self.attr_mapping:
      return src_perm
    self.attr_mapping[node.op_type](node, src_perm)
    return self.tensor_mapping[node.op_type](node, src_perm)
  
  def avg_pool_attr(self, node: OnnxNode, src_perm: List[int]):
    pads = node.attrs['pads']
    pads = [pads[ax] for ax in src_perm]
    node.attrs['pads'] = pads
  
  def pad_attr(self, node: OnnxNode, src_perm: List[int]):
    # pads = node.attrs['pads']
    pads = self.tensor_dict[node.inputs[1]]  # "pads" should be a 1D constant
    _pads = []
    for i, ax in enumerate(src_perm):
      _pads.append(pads[ax * 2: ax * 2 + 2])
    self.tensor_dict[node.inputs[1]] = tf.concat(pads, axis=0)
  
  def cat_attr(self, node: OnnxNode, src_perm: List[int]):
    # print('1233333333333333333333333333333333333333333333', node.name, node.attrs['axis'])
    node.attrs['axis'] = src_perm.index(node.attrs['axis'])
  
  def cat_tensor(self, node: OnnxNode, src_perm: List[int]):
    # TODO
    return src_perm
  
  def transp_attr(self, node: OnnxNode, src_perm: List[int]):
    node.attrs['perm'] = [src_perm[ax] for ax in node.attrs['perm']]
  
  def transp_tensor(self, node: OnnxNode, src_perm: List[int]):
    # NOTE: assume transp_attr is already apply on node
    perm = node.attrs['perm']
    return [src_perm[ax] for ax in perm]
  
  def slice_attr(self, node: OnnxNode, src_perm: List[int]):
    if len(node.inputs) > 3:  # NOTE: no "axes" parameter
      axes = self.tensor_dict[node.inputs[3]]
      py_map = [0] * len(src_perm)
      for i, v in enumerate(src_perm):
        py_map[v] = i
      table = tf.constant(py_map)
      mapped = tf.gather(table, axes)
      self.tensor_dict[node.inputs[3]] = mapped
      
      # vals_tensor = tf.constant(list(range(len(src_perm))))
      # keys_tensor = tf.constant(src_perm)
      # table = tf.lookup.StaticHashTable(
      #     tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
      #     default_value=-1)
      # mapped = table.lookup(tf.cast(axes, tf.int32))
      # self.tensor_dict[node.inputs[3]] = mapped
    else:
      starts = self.tensor_dict[node.inputs[1]]
      shape = starts.shape.as_list()
      axes_key = f"{node.name}_axes"
      self.tensor_dict[axes_key] = tf.constant(src_perm[:len(shape)])
      node.inputs.append(axes_key)


class AxisGraph:

  def __init__(self, onnx_nodes: List[OnnxNode], input_tensors: List[str]):
    self.nodes = onnx_nodes
    self.tensor_axis = {}
    self.input_tensors = input_tensors
    self.graph = defaultdict(set)  # {node name, tensor name}: [{node name, tensor name}, ...]
    self.name2node = {node.name: node for node in onnx_nodes}
    self._build_graph()

  def _build_graph(self):
    for input in self.input_tensors:
      self.tensor_axis[input] = common.sys_config.input_perum
    
    for node in self.nodes:
      if node.op_type != 'Constant':
        for in_ten in node.inputs:
          self.graph[in_ten].add(node.name)
        for out_ten in node.outputs:
          self.graph[node.name].add(out_ten)
      else:
        pass
    
  def infer_perm(self, tensor_dict):
    topo_order = []
    visit = set()
    
    def _dfs(node: str):
      nonlocal topo_order, visit
      visit.add(node)
      for neb in self.graph[node]:
        if neb not in visit:
          _dfs(neb)
      topo_order.append(node)
    
    for in_ten in self.input_tensors:
      if in_ten not in visit:
        _dfs(in_ten)
    topo_order = topo_order[::-1]

    ax_map = AxisMapping(tensor_dict)
    for parent in topo_order:
      src_perm = self.tensor_axis[parent]
      for child in self.graph[parent]:
        if child in self.name2node:
          cnode = self.name2node[child]
          self.tensor_axis[child] = ax_map.mapping(cnode, src_perm)
        else:
          # child is a tensor
          cnode = FakeNode('Tensor', '1')
          self.tensor_axis[child] = ax_map.mapping(cnode, src_perm)
    return [self.name2node[n] for n in topo_order if n in self.name2node]


class TFModuleHelper(object):
  """ Helper class for BackendTFModule and TFModule
  """

  # create tf.Variable for handlers that required to use variable in handler
  @classmethod
  def _create_handlers_variables_for_graph(cls,
                                           handlers,
                                           graph,
                                           init_dict,
                                           var_dict=None):
    var_dict = dict() if var_dict is None else var_dict
    for node in graph.node:
      var_dict = cls._create_handler_variables_for_node(handlers,
                                                        OnnxNode(node),
                                                        init_dict, var_dict)
    return var_dict

  @classmethod
  def _create_handler_variables_for_node(cls,
                                         handlers,
                                         node,
                                         init_dict=None,
                                         var_dict=None):
    init_dict = dict() if init_dict is None else init_dict
    var_dict = dict() if var_dict is None else var_dict
    handler = handlers[node.domain].get(
        node.op_type, None) if node.domain in handlers else None
    var_dict = handler.create_variables(
        handlers, node, init_dict, var_dict,
        cls._create_handlers_variables_for_graph) if handler else var_dict
    return var_dict


class BackendTFModule(tf.Module):
  """ BackendTFModule is the tf.Module class used in backend.prepare,
  tf_rep.export_graph and tf_rep.run
  """

  def __init__(self, handlers, opset, strict, graph_def, backend):
    super(BackendTFModule, self).__init__()
    self.handlers = handlers
    self.opset = opset
    self.strict = strict
    self.graph_def = graph_def
    self.backend = backend
    self.outputs = []
    self.initializer_dict = self._get_initializer_from_graph_and_subgraphs(
        graph_def)
    self.handler_variables = TFModuleHelper._create_handlers_variables_for_graph(
        handlers, graph_def, self.initializer_dict)
    self.is_export = False
    self.clast_input = True

  # get initializer from the main graph and all subgraphs in loop or if or scan
  # into tensor_dict
  def _get_initializer_from_graph_and_subgraphs(self, graph, init_dict=None):
    init_dict = dict() if init_dict is None else init_dict
    if graph.initializer:
      init_dict.update(
          self.backend._onnx_initializer_to_input_dict_items(graph.initializer))
    for node in graph.node:
      handler = self.handlers[node.domain].get(
          node.op_type, None) if node.domain in self.handlers else None
      init_dict = handler.get_initializer_from_subgraph(
          OnnxNode(node), init_dict, self.
          _get_initializer_from_graph_and_subgraphs) if handler else init_dict
    return init_dict

  @tf.function
  def gen_tensor_dict(self, input_dict):
    tensor_dict = dict(input_dict)
    tensor_dict.update(self.initializer_dict)
    tensor_dict.update(self.handler_variables)

    for node in self.graph_def.node:
      onnx_node = OnnxNode(node)
      output_ops = self.backend._onnx_node_to_tensorflow_op(onnx_node,
                                                            tensor_dict,
                                                            self.handlers,
                                                            opset=self.opset,
                                                            strict=self.strict)
      curr_node_output_map = dict(zip(onnx_node.outputs, output_ops))
      tensor_dict.update(curr_node_output_map)

    return tensor_dict

  @tf.function(autograph=True)
  def __call__(self, **kwargs):
    tensor_dict = kwargs
    tensor_dict.update(self.initializer_dict)
    tensor_dict.update(self.handler_variables)

    if self.clast_input:
      _input_perum = common.sys_config.input_perum
      common.sys_config.input_perum = [0, 2, 3, 1]
    
      for in_node in self.graph_def.input:
        in_ten = tensor_dict[in_node.name]
        tensor_dict[in_node.name] = tf.transpose(in_ten, perm=[0, 2, 3, 1])
    
    onnx_nodes = [OnnxNode(node) for node in self.graph_def.node]

    """
    Load constants to tensor_dict
    """
    for onnx_node in [n for n in onnx_nodes if n.op_type == 'Constant']:
      output_ops = self.backend._onnx_node_to_tensorflow_op(onnx_node,
                                                            tensor_dict,
                                                            self.handlers,
                                                            opset=self.opset,
                                                            strict=self.strict)
      curr_node_output_map = dict(zip(onnx_node.outputs, output_ops))
      tensor_dict.update(curr_node_output_map)

    if self.clast_input:
      agraph = AxisGraph(onnx_nodes, [n.name for n in self.graph_def.input])
      agraph.infer_perm(tensor_dict)
      # breakpoint()
      node_left = [n for n in agraph.nodes if n.op_type != 'Constant']
    else:
      node_left = [n for n in onnx_nodes if n.op_type != 'Constant']
    """
    handle remaining nodes
    """
    for onnx_node in node_left:
      output_ops = self.backend._onnx_node_to_tensorflow_op(onnx_node,
                                                            tensor_dict,
                                                            self.handlers,
                                                            opset=self.opset,
                                                            strict=self.strict)
      curr_node_output_map = dict(zip(onnx_node.outputs, output_ops))
      tensor_dict.update(curr_node_output_map)

    outputs = dict()
    for output in self.outputs:
      if not self.is_export or tensor_dict[output].shape.is_fully_defined():
        outputs[output] = tensor_dict[output]
      else:
        # Restore the output shape if not fully defined during export
        for o in self.graph_def.output:
          if o.name == output:
            o_shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
            outputs[
                output] = tensor_dict[output] if 0 in o_shape else tf.reshape(
                    tensor_dict[output], o_shape)
            break
    
    if self.clast_input:
      common.sys_config.input_perum = _input_perum
    return outputs


class TFModule(tf.Module):
  """ TFModule is the tf.Module class used in backend.run_node.
  """

  def __init__(self, node, backend):
    super(TFModule, self).__init__()
    self.node = node
    self.backend = backend
    self.handlers = backend._get_handlers(opset=None)
    self.handler_variables = TFModuleHelper._create_handler_variables_for_node(
        self.handlers, node)

  @tf.function
  def __call__(self, **input_dict):
    input_dict.update(self.handler_variables)
    outputs = self.backend._onnx_node_to_tensorflow_op(self.node, input_dict,
                                                       self.handlers)
    return outputs
