from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import time
from typing import Callable

import numpy as np

from config import AppConfig


class RIFEBackend(ABC):
    name = "RIFE"

    @abstractmethod
    def infer(self, first: np.ndarray, second: np.ndarray, timestep: float) -> np.ndarray:
        raise NotImplementedError

    def warmup(self, width: int, height: int, rounds: int = 2) -> float:
        sample = np.zeros((height, width, 3), dtype=np.uint8)
        started = time.perf_counter()
        for _ in range(rounds):
            self.infer(sample, sample, 0.5)
        return (time.perf_counter() - started) * 1000 / rounds


class PyTorchRIFE(RIFEBackend):
    """Built-in RIFE v4.25 backend using the weights shipped with the project."""

    def __init__(self, log: Callable[[str], None]) -> None:
        try:
            import torch
            import torch.nn.functional as functional
            from train_log.RIFE_HDv3 import Model
        except (ImportError, OSError) as exc:
            raise RuntimeError("The built-in RIFE backend requires the installed PyTorch/CUDA runtime.") from exc
        self.torch, self.functional = torch, functional
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("The built-in FP16 backend requires a CUDA GPU.")
        model_dir = Path(__file__).resolve().parent / "train_log"
        weights = model_dir / "flownet.pkl"
        if not weights.is_file():
            raise FileNotFoundError(f"Built-in RIFE weights are missing: {weights}")
        self.model = Model()
        self.model.load_model(str(model_dir), -1)
        self.model.eval()
        self.model.flownet.half()
        torch.backends.cudnn.benchmark = True
        self.name = f"PyTorch RIFE v{self.model.version:g} FP16"
        log(f"RIFE backend: {self.name} on {torch.cuda.get_device_name(0)}")

    def infer(self, first: np.ndarray, second: np.ndarray, timestep: float) -> np.ndarray:
        torch = self.torch
        height, width = first.shape[:2]
        padded_h = (height + 127) // 128 * 128
        padded_w = (width + 127) // 128 * 128
        padding = (0, padded_w - width, 0, padded_h - height)
        # DecodeWorker produces RGB, matching RIFE's training channel order.
        a = torch.from_numpy(np.ascontiguousarray(first.transpose(2, 0, 1))).unsqueeze(0)
        b = torch.from_numpy(np.ascontiguousarray(second.transpose(2, 0, 1))).unsqueeze(0)
        a = self.functional.pad(a.pin_memory().to(self.device, non_blocking=True).half().div_(255), padding)
        b = self.functional.pad(b.pin_memory().to(self.device, non_blocking=True).half().div_(255), padding)
        with torch.inference_mode():
            output = self.model.inference(a, b, timestep, 1.0)[:, :, :height, :width]
        return output[0].mul(255).clamp_(0, 255).byte().cpu().numpy().transpose(1, 2, 0)


def _tensor(frame: np.ndarray, dtype: np.dtype) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = frame.shape[:2]
    padded_h = (height + 31) // 32 * 32
    padded_w = (width + 31) // 32 * 32
    output = np.zeros((1, 3, padded_h, padded_w), dtype=dtype)
    output[:, :, :height, :width] = frame.transpose(2, 0, 1)[None].astype(dtype) / 255.0
    return np.ascontiguousarray(output), (height, width)


def _image_from_output(output: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    value = np.asarray(output)
    if value.ndim == 4:
        value = value[0]
    if value.shape[0] in (3, 4):
        value = value[:3].transpose(1, 2, 0)
    return np.clip(value[:height, :width] * 255.0, 0, 255).astype(np.uint8)


class ONNXRIFE(RIFEBackend):
    def __init__(self, model_path: str, backend: str, config: AppConfig, log: Callable[[str], None]) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install onnxruntime-gpu to use ONNX/TensorRT inference.") from exc

        available = ort.get_available_providers()
        options: list[tuple[str, dict] | str] = []
        if backend == "TensorRT FP16":
            if "TensorrtExecutionProvider" not in available:
                raise RuntimeError(f"TensorRT Execution Provider is unavailable. Available: {available}")
            options.append(("TensorrtExecutionProvider", {
                "device_id": config.ort_device_id,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(config.trt_cache_dir),
                "trt_timing_cache_enable": True,
            }))
        if "CUDAExecutionProvider" in available:
            options.append(("CUDAExecutionProvider", {"device_id": config.ort_device_id, "cudnn_conv_algo_search": "HEURISTIC"}))
        options.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(model_path, providers=options)
        self.inputs = self.session.get_inputs()
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.name = self.session.get_providers()[0]
        log(f"RIFE backend: {self.name}")

    def infer(self, first: np.ndarray, second: np.ndarray, timestep: float) -> np.ndarray:
        image_meta = [item for item in self.inputs if "time" not in item.name.lower() and "t" != item.name.lower()]
        dtype = np.float16 if image_meta and "float16" in image_meta[0].type else np.float32
        a, size = _tensor(first, dtype)
        b, _ = _tensor(second, dtype)
        feed: dict[str, np.ndarray] = {}
        image_inputs = image_meta
        time_inputs = [item for item in self.inputs if item not in image_inputs]
        if len(image_inputs) >= 2:
            feed[image_inputs[0].name], feed[image_inputs[1].name] = a, b
        elif len(image_inputs) == 1:
            feed[image_inputs[0].name] = np.concatenate((a, b), axis=1)
        else:
            raise RuntimeError("Could not identify RIFE image inputs in the ONNX graph.")
        for item in time_inputs:
            item_type = np.float16 if "float16" in item.type else np.float32
            rank = len(item.shape)
            shape = tuple(1 if not isinstance(x, int) else x for x in item.shape)
            feed[item.name] = np.full(shape if rank else (), timestep, dtype=item_type)
        outputs = self.session.run(self.output_names, feed)
        image = next((value for value in outputs if np.asarray(value).ndim == 4), outputs[0])
        return _image_from_output(image, size)


class TensorRTEngine(RIFEBackend):
    """Direct TensorRT plan loader for explicit-batch RIFE engines."""

    def __init__(self, engine_path: str, log: Callable[[str], None]) -> None:
        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
        except ImportError as exc:
            raise RuntimeError("Direct .engine loading requires tensorrt and pycuda.") from exc
        self.trt, self.cuda = trt, cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as handle:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError("TensorRT could not deserialize the engine (version or GPU mismatch).")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.name = "TensorRT Engine FP16"
        log(f"RIFE backend: {self.name}")

    def _names(self) -> list[str]:
        if hasattr(self.engine, "num_io_tensors"):
            return [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        return [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]

    def infer(self, first: np.ndarray, second: np.ndarray, timestep: float) -> np.ndarray:
        trt, cuda = self.trt, self.cuda
        a, size = _tensor(first, np.float16)
        b, _ = _tensor(second, np.float16)
        names = self._names()
        input_names = []
        output_names = []
        for i, name in enumerate(names):
            is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT) if hasattr(self.engine, "get_tensor_mode") else self.engine.binding_is_input(i)
            (input_names if is_input else output_names).append(name)
        arrays: dict[str, np.ndarray] = {}
        image_names = [n for n in input_names if "time" not in n.lower() and n.lower() != "t"]
        time_names = [n for n in input_names if n not in image_names]
        if len(image_names) >= 2:
            arrays[image_names[0]], arrays[image_names[1]] = a, b
        elif len(image_names) == 1:
            arrays[image_names[0]] = np.concatenate((a, b), axis=1)
        for name in time_names:
            arrays[name] = np.array([timestep], dtype=np.float16)

        allocations: dict[str, object] = {}
        bindings = [0] * len(names)
        for name, value in arrays.items():
            if hasattr(self.context, "set_input_shape"):
                self.context.set_input_shape(name, value.shape)
            else:
                self.context.set_binding_shape(names.index(name), value.shape)
            allocations[name] = cuda.mem_alloc(value.nbytes)
            cuda.memcpy_htod_async(allocations[name], value, self.stream)
        host_outputs: dict[str, np.ndarray] = {}
        for name in output_names:
            shape = tuple(self.context.get_tensor_shape(name)) if hasattr(self.context, "get_tensor_shape") else tuple(self.context.get_binding_shape(names.index(name)))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name) if hasattr(self.engine, "get_tensor_dtype") else self.engine.get_binding_dtype(names.index(name)))
            host_outputs[name] = np.empty(shape, dtype=dtype)
            allocations[name] = cuda.mem_alloc(host_outputs[name].nbytes)
        for name in names:
            address = int(allocations[name])
            if hasattr(self.context, "set_tensor_address"):
                self.context.set_tensor_address(name, address)
            else:
                bindings[names.index(name)] = address
        if hasattr(self.context, "execute_async_v3"):
            self.context.execute_async_v3(self.stream.handle)
        else:
            self.context.execute_async_v2(bindings, self.stream.handle)
        for name, host in host_outputs.items():
            cuda.memcpy_dtoh_async(host, allocations[name], self.stream)
        self.stream.synchronize()
        image = next((v for v in host_outputs.values() if v.ndim == 4), next(iter(host_outputs.values())))
        return _image_from_output(image, size)


def load_rife(options, config: AppConfig, log: Callable[[str], None]) -> RIFEBackend:
    if options.backend == "Built-in RIFE FP16":
        return PyTorchRIFE(log)
    path = Path(options.model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError("Select a valid RIFE .onnx or .engine model file.")
    if path.suffix.lower() in {".engine", ".trt", ".plan"}:
        return TensorRTEngine(str(path), log)
    if path.suffix.lower() != ".onnx":
        raise ValueError("RIFE model must be an ONNX model or TensorRT engine.")
    return ONNXRIFE(str(path), options.backend, config, log)
