import onnx
from onnx import helper
model = onnx.load("rknn_convert/best_fixed.onnx")
new_nodes = []
for node in model.graph.node:
    if node.op_type == "Resize":
        new_node = helper.make_node(
            "Upsample",
            inputs=[node.input[0]],
            outputs=node.output,
            name=node.name + "_upsample",
            mode="nearest"
        )
        new_nodes.append(new_node)
    else:
        new_nodes.append(node)
new_model = helper.make_model(
    helper.make_graph(
        new_nodes,
        model.graph.name,
        list(model.graph.input),
        list(model.graph.output),
        list(model.graph.initializer)
    )
)
new_model.opset_import[:] = model.opset_import
onnx.save(new_model, "rknn_convert/best_no_resize.onnx")
