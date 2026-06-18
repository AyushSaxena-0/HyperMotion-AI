import os
import sys
import torch
import torch.nn as nn
import argparse

# Add path to load RIFE model
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, "Practical-RIFE"))

from train_log.RIFE_HDv3 import Model

class RIFEONNXWrapper(nn.Module):
    """Wraps RIFE flownet for ONNX export, supporting dynamic inputs."""
    def __init__(self, flownet):
        super().__init__()
        self.flownet = flownet

    def forward(self, img0, img1, timestep):
        # Concatenate inputs along channel dimension
        imgs = torch.cat((img0, img1), 1)
        # Scale list default (1.0x scale inference)
        scale_list = [32, 16, 8, 4, 1]
        
        # Forward pass on standard IFNet
        flow_list, mask_val, merged = self.flownet(imgs, timestep, scale_list)
        return merged[-1]

def main():
    parser = argparse.ArgumentParser(description="Export RIFE PyTorch weights to ONNX format")
    parser.add_argument("--model_dir", type=str, default="train_log", help="Path to RIFE model checkpoint folder")
    parser.add_argument("--output", type=str, default="models/rife425.onnx", help="Path to save output ONNX model")
    parser.add_argument("--width", type=int, default=1280, help="Base width for dummy input tracing")
    parser.add_argument("--height", type=int, default=720, help="Base height for dummy input tracing")
    args = parser.parse_args()

    # Create directories if missing
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading RIFE model weights from {args.model_dir}...")
    model = Model()
    model.load_model(args.model_dir, -1)
    model.eval()

    # Create ONNX wrapper module
    wrapper = RIFEONNXWrapper(model.flownet)
    wrapper.eval()

    # Dummy inputs
    # Timestep is a 4D tensor with shape [1, 1, 1, 1]
    img0 = torch.randn(1, 3, args.height, args.width)
    img1 = torch.randn(1, 3, args.height, args.width)
    timestep = torch.tensor([[[[0.5]]]], dtype=torch.float32)

    print("Exporting RIFE model to ONNX...")
    # Dynamic axes allow the exported model to process any resolution at runtime
    torch.onnx.export(
        wrapper,
        (img0, img1, timestep),
        args.output,
        opset_version=16, # Opset 16 natively supports grid_sample (used for warping)
        input_names=["img0", "img1", "timestep"],
        output_names=["output"],
        dynamic_axes={
            "img0": {2: "height", 3: "width"},
            "img1": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"}
        }
    )
    print(f"Successfully exported ONNX model to {args.output}!")

if __name__ == "__main__":
    main()
