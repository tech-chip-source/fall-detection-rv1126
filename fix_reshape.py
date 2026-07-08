import onnx
from onnx import helper, numpy_helper
model = onnx.load("rknn_convert/best_final.onnx")
target_node = None
shape_init = None
for node in model.graph.node:
    if node.name == "/model.22/Reshape":
        target_node = node
        for init in model.graph.initializer:
            if init.name == node.input[1]:
                shape_init = init
                break
        break
shape_vals = numpy_helper.to_array(shape_init).tolist()
new_node = helper.make_node(
    "Reshape",
    inputs=[target_node.input[0]],
    outputs=target_node.output,
    name=target_node.name,
    shape=shape_vals
)
new_nodes = []
for node in model.graph.node:
    if node.name == "/model.22/Reshape":
        new_nodes.append(new_node)
    else:
        new_nodes.append(node)
model.graph.node[:] = new_nodes
onnx.save(model, "rknn_convert/best_manual.onnx")
