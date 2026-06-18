import os
import argparse
import tensorrt as trt

def build_engine(onnx_path, engine_path, precision="FP16", max_width=1920, max_height=1080):
    """Builds a TensorRT engine from an ONNX model, optimizing for dynamic resolutions."""
    print(f"Building TensorRT Engine from {onnx_path}...")
    
    # Initialize TensorRT builder, network, parser, and config
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # Set workspace size (VRAM pool for builders) - 2GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    
    # Enable precision modes
    if precision == "FP16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Enabled FP16 half-precision mode.")
        else:
            print("Warning: Fast FP16 not supported on this platform. Falling back to FP32.")
            
    # Load ONNX file
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            print("Failed to parse the ONNX file:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
            
    # Set up optimization profile for dynamic resolution inputs
    # Min resolution: 256x256, Optimal: 1280x720 (720p), Max: 1920x1080 (1080p)
    profile = builder.create_optimization_profile()
    
    # Profile for img0
    profile.set_shape(
        "img0", 
        (1, 3, 256, 256), 
        (1, 3, 720, 1280), 
        (1, 3, max_height, max_width)
    )
    # Profile for img1
    profile.set_shape(
        "img1", 
        (1, 3, 256, 256), 
        (1, 3, 720, 1280), 
        (1, 3, max_height, max_width)
    )
    # Profile for timestep (always shape [1, 1, 1, 1])
    profile.set_shape(
        "timestep", 
        (1, 1, 1, 1), 
        (1, 1, 1, 1), 
        (1, 1, 1, 1)
    )
    
    config.add_optimization_profile(profile)
    
    # Create engine and serialize
    print("Building engine (this can take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Engine build failed.")
        return False
        
    # Write engine to disk
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        
    print(f"Successfully saved TensorRT engine to: {engine_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile an ONNX model into an optimized TensorRT engine")
    parser.add_argument("--onnx", type=str, default="models/rife425.onnx", help="Path to input ONNX model")
    parser.add_argument("--output", type=str, default="engines/rife425_fp16.engine", help="Path to save output TRT engine")
    parser.add_argument("--precision", type=str, default="FP16", choices=["FP32", "FP16"], help="Precision execution mode")
    parser.add_argument("--max_width", type=int, default=1920, help="Maximum image width supported by engine")
    parser.add_argument("--max_height", type=int, default=1080, help="Maximum image height supported by engine")
    args = parser.parse_args()
    
    build_engine(args.onnx, args.output, args.precision, args.max_width, args.max_height)
