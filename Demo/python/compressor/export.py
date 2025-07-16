import torch
from module import LidarGuidedCompressor

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower


#state_dict = torch.load("checkpoint_1.pth.tar")['state_dict']
state_dict = torch.load("compressor-cb93b5fb.pth.tar")

eb_quantiles = state_dict['entropy_bottleneck.quantiles'].cpu()
eb_quantized_cdf = state_dict['entropy_bottleneck._quantized_cdf'].cpu()
eb_cdf_length = state_dict['entropy_bottleneck._cdf_length'].cpu()
eb_offset = state_dict['entropy_bottleneck._offset'].cpu()
scale_table = state_dict['gaussian_conditional.scale_table'].cpu()
gc_quantized_cdf = state_dict['gaussian_conditional._quantized_cdf'].cpu()
gc_cdf_length = state_dict['gaussian_conditional._cdf_length'].cpu()
gc_offset = state_dict['gaussian_conditional._offset'].cpu()

model = LidarGuidedCompressor(eb_quantiles, eb_quantized_cdf, eb_cdf_length, eb_offset, scale_table, gc_quantized_cdf, gc_cdf_length, gc_offset).cpu().eval()

# Load weights here
model.load_state_dict(state_dict, strict=False)

dummy_image_input = torch.randn(1, 3, 256, 256)
dummy_lidar_input = torch.randn(1, 1, 256, 256)
lambda_encoder = torch.ones(1, 1, 256, 256)

sample_inputs = (dummy_image_input, lambda_encoder, dummy_lidar_input)

et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[XnnpackPartitioner()]
).to_executorch()

with open("model.pte", "wb") as f:
    f.write(et_program.buffer)
