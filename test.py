import tensorflow as tf
from paper_model import build_model as build_2d
from paper_model_1d import build_model as build_1d

# Build models
model_2d = build_2d()
model_1d = build_1d()

# Print summaries
print("=== 2D CNN Model ===")
model_2d.summary()
print("\n=== 1D CNN Model ===")
model_1d.summary()
