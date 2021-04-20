#include <cuda_runtime.h>
#include <time.h>
#include <torch/cuda.h>
#include <torch/script.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
  //   clock_t start, end;

  torch::DeviceType device_type = at::kCPU;
  std::cout << "cuda::is_available():" << torch::cuda::is_available()
            << std::endl;

  if (torch::cuda::is_available()) {
    device_type = at::kCUDA;
  }
  torch::jit::script::Module model =
      torch::jit::load("../cpu_al.pth", device_type);
  //   model.to(device_type);
  //   start = clock();
  while (1) {
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cv::Mat input_image = cv::imread("../275.jpg");
    cv::resize(input_image, input_image, cv::Size(224, 224));

    torch::Tensor image_tensor =
        torch::from_blob(input_image.data,
                         {input_image.rows, input_image.cols, 3}, torch::kByte);

    image_tensor = image_tensor.permute({2, 0, 1});

    image_tensor = image_tensor.toType(torch::kFloat);

    image_tensor = image_tensor.div(255);

    image_tensor = image_tensor.unsqueeze(0);

    image_tensor = image_tensor.to(device_type);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image_tensor.to(device_type));

    torch::Tensor pred = model.forward(inputs).toTensor();

    //   end = clock();
    //   std::cout << "gpu运行时间" << ((double)(end - start) / CLOCKS_PER_SEC)
    //   * 1000
    //             << " ms" << std::endl;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time used:%.2f ms\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // std::cout << pred << std::endl;
  }
}