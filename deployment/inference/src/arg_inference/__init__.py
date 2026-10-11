"""Shared ARG ONNX image inference used by hosted and local callers."""

from arg_inference.runtime import Plan, load_plan, predict_file

__all__ = ["Plan", "load_plan", "predict_file"]
