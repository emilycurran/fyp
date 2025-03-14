import os
import time
import tensorflow as tf
from tensorflow.keras import backend as K

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def measure_inference_time(model, sample_input, num_trials=100, device='/GPU:0'):
    """Measure the average inference time for a single input."""
    times = []
    with tf.device(device):  # Force execution on specified device
        for _ in range(num_trials):
            start_time = time.time()
            _ = model.predict(sample_input, verbose=0)
            end_time = time.time()
            times.append(end_time - start_time)
    return sum(times) / num_trials

def get_model_size(model_path):
    """Return model file size in MB."""
    size = os.path.getsize(model_path) / (1024 * 1024)  # Convert bytes to MB
    return size

def count_flops(model):
    """Compute FLOPs using TensorFlow 2.x."""
    try:
        # Convert model to a TensorFlow concrete function
        concrete_function = tf.function(model).get_concrete_function(
            tf.TensorSpec(model.input_shape, model.input.dtype)
        )
        
        # Get the computational graph
        frozen_func = concrete_function.graph

        # Compute FLOPs
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(frozen_func, run_meta=run_meta, options=opts)

        return flops.total_float_ops if flops is not None else None
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return None
def compare_models():
    model_1d_path = "1D_CNN.h5"
    model_2d_path = "2D_CNN.h5"
    
    # Load models
    model_1d = load_model(model_1d_path)
    model_2d = load_model(model_2d_path)
    
    # Create sample inputs
    sample_1d = tf.random.normal([1, 256, 1])
    sample_2d = tf.random.normal([1, 128, 128, 1])
    
    print("Measuring inference time on CPU...")
    inf_time_1d_cpu = measure_inference_time(model_1d, sample_1d, device='/CPU:0')
    inf_time_2d_cpu = measure_inference_time(model_2d, sample_2d, device='/CPU:0')
    
    print("Measuring inference time on GPU (if available)...")
    inf_time_1d_gpu = measure_inference_time(model_1d, sample_1d, device='/GPU:0')
    inf_time_2d_gpu = measure_inference_time(model_2d, sample_2d, device='/GPU:0')
    
    print("Calculating model size...")
    model_size_1d = get_model_size(model_1d_path)  # Check directly for .h5 file
    model_size_2d = get_model_size(model_2d_path)  # Check directly for .h5 file

        
    print("Counting FLOPs...")
    flops_1d = count_flops(model_1d)
    flops_2d = count_flops(model_2d)
    
    print("Counting parameters...")
    params_1d = model_1d.count_params()
    params_2d = model_2d.count_params()
    
    print("\nComparison Results:")
    print(f"1D CNN - Inference Time (CPU): {inf_time_1d_cpu:.6f} sec")
    print(f"2D CNN - Inference Time (CPU): {inf_time_2d_cpu:.6f} sec")
    print(f"1D CNN - Inference Time (GPU): {inf_time_1d_gpu:.6f} sec")
    print(f"2D CNN - Inference Time (GPU): {inf_time_2d_gpu:.6f} sec")
    print(f"1D CNN - Model Size: {model_size_1d:.2f} MB")
    print(f"2D CNN - Model Size: {model_size_2d:.2f} MB")
    print(f"1D CNN - FLOPs: {flops_1d}")
    print(f"2D CNN - FLOPs: {flops_2d}")
    print(f"1D CNN - Parameters: {params_1d}")
    print(f"2D CNN - Parameters: {params_2d}")

if __name__ == "__main__":
    compare_models()