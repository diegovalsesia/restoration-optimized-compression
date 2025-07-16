import torch
from module import LidarGuidedCompressor
import compressai.ans as ans


def entropy_encode(symbols, indexes, quantized_cdf, cdf_length, offset):
    """
    Compress input tensors to char strings.

    Args:
        inputs (torch.Tensor): input tensors
        indexes (torch.IntTensor): tensors CDF indexes
        means (torch.Tensor, optional): optional tensor means
    """
    
    strings = []
    for i in range(symbols.size(0)):
        rv = ans.RansEncoder().encode_with_indexes(
            symbols[i].reshape(-1).int().tolist(),
            indexes[i].reshape(-1).int().tolist(),
            quantized_cdf.tolist(),
            cdf_length.reshape(-1).int().tolist(),
            offset.reshape(-1).int().tolist(),
        )
        strings.append(rv)
    return strings





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

#dummy_image_input = torch.randn(1, 3, 256, 256)
#dummy_lidar_input = torch.randn(1, 1, 256, 256)

from PIL import Image
import numpy as np

dummy_image_input = np.array(Image.open("blur.png").convert("RGB")).astype(np.float32)
dummy_image_input = torch.tensor(dummy_image_input).permute(2, 0, 1).unsqueeze(0).float() / 255.0 
dummy_image_input = torch.nn.functional.interpolate(dummy_image_input, size=(256, 256), mode='bilinear', align_corners=False)
dummy_lidar_input = np.array(Image.open("depth.png").convert("L")).astype(np.float32)
dummy_lidar_input = torch.tensor(dummy_lidar_input).unsqueeze(0).unsqueeze(0) /255.0 
dummy_lidar_input = torch.nn.functional.interpolate(dummy_lidar_input, size=(256, 256), mode='bilinear', align_corners=False)

lambda_encoder = 0.025*torch.ones(1, 1, 256, 256)

# Test the model with dummy inputs
with torch.no_grad():
    z_symbols, z_indexes, eb_quantized_cdf, eb_cdf_length, eb_offset, y_symbols, y_indexes, means_hat, gc_quantized_cdf, gc_cdf_length, gc_offset = model(dummy_image_input, lambda_encoder, dummy_lidar_input)

    z_strings = entropy_encode(z_symbols, z_indexes, eb_quantized_cdf, eb_cdf_length, eb_offset)
    y_strings = entropy_encode(y_symbols, y_indexes, gc_quantized_cdf, gc_cdf_length, gc_offset)

    # save them to raw file
    with open("z_strings.raw", "wb") as f:
        for s in z_strings:
            f.write(s)
    with open("y_strings.raw", "wb") as f:
        for s in y_strings:
            f.write(s)
