# ===============================================
# Libraries
# ===============================================

import numpy

import onnx
from onnx import version_converter

from concrete.ml.torch.compile import compile_onnx_model
import torch

# ===============================================
# Variables
# ===============================================

MODEL_FILE = "resnet18-WOGAP.onnx"

# ===============================================
# Model Load
# ===============================================

model = onnx.load(MODEL_FILE)
model = version_converter.convert_version(model, 14)
try:
    onnx.checker.check_model(model)
except onnx.checker.ValidationError as e:
    print("model is not valid")
else:
    print("model is valid")

# ===============================================
# Quantization
# ===============================================

calibration_input_set = torch.FloatTensor(100, 3, 224, 224).uniform_(-100, 100)

qmodel = compile_onnx_model(
    model,
    calibration_input_set
)
