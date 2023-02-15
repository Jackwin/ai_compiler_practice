from onnx import TensorProto

from onnx.helper import (
  make_model, make_node, make_graph,
  make_tensor_value_info)
from onnx.checker import check_model


# Build nodes to a graph; build graphs to a model
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])

Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

graph = make_graph([node1, node2],
                   'lr',
                   [X, A, B],
                   [Y])
onnx_model = make_model(graph)
check_model(onnx_model)
print(onnx_model)


#Tranverse the model
def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

print('--- inputs ---')
print(onnx_model.graph.input)
print('--- outputs ---')
for obj in onnx_model.graph.input:
    print("name=%r" "dtype=%r" "shape=%r" % (
        obj.name, obj.type.tensor_type.elem_type,
        shape2tuple(obj.type.tensor_type.shape)))


