# Installation
```bash
pip install onnx -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## data sturcture

ONNX uses the protocol buffer to represetn the model, containing the following protos:
- ModelProto: model description and GraphProto
- GraphProto: the node information, node initializerm and tensors
- NodeProto: input and output tensor names, node initializers and node attr
- TensorProto: node initializer, data type, shape and values
- ValueInfoProto: IO tensor where only the data type and shape are defined

