import cv2
import numpy as np
import time

def test_cuda_availability():
    """Test if OpenCV is built with CUDA support and check available CUDA devices"""
    print("OpenCV version:", cv2.__version__)
    
    # Check if OpenCV is built with CUDA
    print("\nCUDA Support Information:")
    print("CUDA-enabled build:", cv2.cuda.getCudaEnabledDeviceCount() > 0)
    
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"Found {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s)")
        
        # Print device information
        for device_id in range(cv2.cuda.getCudaEnabledDeviceCount()):
            cv2.cuda.setDevice(device_id)
            props = cv2.cuda.getDevice()
            print(f"\nDevice {device_id} properties:")
            print(f"- Name: {cv2.cuda.printCudaDeviceInfo(device_id)}")
            print(f"- Compute capability: {props}")
    else:
        print("No CUDA-capable devices found")
        return False
    
    return True

def benchmark_cuda_operations():
    """Run benchmark tests comparing CPU vs GPU performance"""
    # Create a large test image
    print("\nRunning benchmark tests...")
    image = np.random.randint(0, 255, (1920, 1080), dtype=np.uint8)
    
    # CPU Gaussian Blur
    start_time = time.time()
    cpu_result = cv2.GaussianBlur(image, (7, 7), 0)
    cpu_time = time.time() - start_time
    print(f"CPU Gaussian Blur time: {cpu_time:.4f} seconds")
    
    try:
        # GPU Gaussian Blur
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)
        
        start_time = time.time()
        gpu_result = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC1, cv2.CV_8UC1, (7, 7), 0
        )(gpu_image)
        # Download result to verify
        gpu_result_cpu = gpu_result.download()
        gpu_time = time.time() - start_time
        print(f"GPU Gaussian Blur time: {gpu_time:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"GPU Speedup: {speedup:.2f}x")
        
        # Verify results match
        if np.allclose(cpu_result, gpu_result_cpu, atol=1.0):
            print("Results match between CPU and GPU")
        else:
            print("Warning: Results differ between CPU and GPU")
            
    except cv2.error as e:
        print(f"Error during GPU operations: {e}")

def main():
    print("Starting OpenCV CUDA Test\n" + "="*30)
    
    if test_cuda_availability():
        benchmark_cuda_operations()
    else:
        print("\nSkipping benchmarks due to no CUDA support")

if __name__ == "__main__":
    main()