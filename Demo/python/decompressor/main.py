import torch
from decompression_module import LidarGuidedCompressor
from restoration_module import CatRestormer
import argparse
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Lidar Guided Compressor")
    parser.add_argument("--decompressor_model_path", type=str, required=True, help="Path to the decompressor model file")
    parser.add_argument("--restoration_model_path", type=str, required=True, help="Path to the restoration model file")
    parser.add_argument("--z_strings", type=str, required=True, help="Path to the z_strings file")
    parser.add_argument("--y_strings", type=str, required=True, help="Path to the y_strings file")
    
    args = parser.parse_args()

    # Load the decompressor model
    state_dict = torch.load(args.decompressor_model_path)
    eb_quantiles = state_dict['entropy_bottleneck.quantiles'].cpu()
    eb_quantized_cdf = state_dict['entropy_bottleneck._quantized_cdf'].cpu()
    eb_cdf_length = state_dict['entropy_bottleneck._cdf_length'].cpu()
    eb_offset = state_dict['entropy_bottleneck._offset'].cpu()
    scale_table = state_dict['gaussian_conditional.scale_table'].cpu()
    gc_quantized_cdf = state_dict['gaussian_conditional._quantized_cdf'].cpu()
    gc_cdf_length = state_dict['gaussian_conditional._cdf_length'].cpu()
    gc_offset = state_dict['gaussian_conditional._offset'].cpu()
    decompressor_model = LidarGuidedCompressor(eb_quantiles, eb_quantized_cdf, eb_cdf_length, eb_offset, scale_table, gc_quantized_cdf, gc_cdf_length, gc_offset).cpu().eval()
    decompressor_model.load_state_dict(state_dict, strict=False)

    # Load the restoration model
    restoration_state_dict = torch.load(args.restoration_model_path)
    restoration_model = CatRestormer()
    restoration_model.load_state_dict(restoration_state_dict["state_dict"], strict=True)

    # Load the compressed data
    with open(args.z_strings, "rb") as f:
        z_strings = [f.read()]
    with open(args.y_strings, "rb") as f:
        y_strings = [f.read()]

    lidar_input = np.array(Image.open("depth.png").convert("L")).astype(np.float32)
    lidar_input = torch.tensor(lidar_input).unsqueeze(0).unsqueeze(0) /255.0 
    lidar_input = torch.nn.functional.interpolate(lidar_input, size=(256, 256), mode='bilinear', align_corners=False)
    lambda_decoder = 0.001 * torch.ones(1, 1, 256//16, 256//16)

  
    with torch.no_grad():

          # Decompress the data
        inputs = [y_strings, z_strings]
        x_hat = decompressor_model.decompress(inputs, [4,4], lambda_decoder, lidar_input)

        # Restore the decompressed image
        x_hat = restoration_model(x_hat, lidar_input)

    # Save x_hat as PNG image
    x_hat_np = x_hat.permute(0, 2, 3, 1).cpu().numpy().squeeze()  # Convert to HWC format
    x_hat_np = (x_hat_np * 255).astype(np.uint8)
    image = Image.fromarray(x_hat_np)
    image.save("output_image.png")



if __name__ == "__main__":
    main()


