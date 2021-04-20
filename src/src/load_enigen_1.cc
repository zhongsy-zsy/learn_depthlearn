#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "argsParser.h"
#include "common.h"
#include "common/logger.h"

// using namespace nvinfer1;
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1

const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int CLASS_NUM = 2;
// static const int OUTPUT_SIZE =
//     Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) +
//     1;
static const int OUTPUT_SIZE = 2;

class Logger_1 : public nvinfer1::ILogger {
  //  public:
  //   Logger() {}
  //   ~Logger() {}
  void log(Severity severity, const char* msg) override {
    // suppress info-level messages
    if (severity == Severity::kERROR) std::cout << msg << std::endl;
  }
};

void doInference(nvinfer1::IExecutionContext& context, cudaStream_t& stream,
                 void** buffers, float* input, float* output, int batchSize) {
  // DMA input batch data to device, infer on the batch asynchronously, and DMA
  // output back to host
  cudaMemcpyAsync(buffers[0], input,
                  batchSize * 3 * INPUT_H * INPUT_W * sizeof(float),
                  cudaMemcpyHostToDevice, stream);
  context.enqueue(batchSize, buffers, stream, nullptr);
  cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
}

cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
  int w, h, x, y;
  float r_w = input_w / (img.cols * 1.0);
  float r_h = input_h / (img.rows * 1.0);
  if (r_h > r_w) {
    w = input_w;
    h = r_w * img.rows;
    x = 0;
    y = (input_h - h) / 2;
  } else {
    w = r_h * img.cols;
    h = input_h;
    x = (input_w - w) / 2;
    y = 0;
  }
  cv::Mat re(h, w, CV_8UC3);
  cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
  cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
  re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
  return out;
}

class a {
  void init() {}
};

int main() {
  Logger_1 gLogger;

  cudaSetDevice(DEVICE);

  std::string engine_name = "./ghost.trt";  // load engine name
  std::cout << "hello" << std::endl;
  // start load engine
  char* trtModelStream{nullptr};
  size_t size{0};

  std::ifstream engine_file(engine_name.c_str(), std::ios::binary);

  if (engine_file.good()) {
    engine_file.seekg(0,
                      engine_file.end);  // 定位输入流结束位置地址偏移量为0初
    size = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);
    trtModelStream = new char[size];
    engine_file.read(trtModelStream, size);
    engine_file.close();
  }

  static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
  static float prob[BATCH_SIZE * OUTPUT_SIZE];

  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);

  nvinfer1::ICudaEngine* engine =
      runtime->deserializeCudaEngine(trtModelStream, size);

  nvinfer1::IExecutionContext* context = engine->createExecutionContext();
  delete[] trtModelStream;

  void* buffers[2];
  const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
  const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
  std::cout << "input: " << inputIndex << " output " << outputIndex
            << std::endl;

  //  使用上面的indices，在GPU上创建一个指向input和output缓冲区的buffer数组

  // Create GPU buffers on device
  cudaMalloc(&buffers[inputIndex],
             BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float));
  cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

  // Create stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  while (1) {
    // auto start = std::chrono::system_clock::now();

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cv::Mat img = cv::imread("../3.jpg");
    cv::Mat pre_img = preprocess_img(img, INPUT_W, INPUT_H);

    int i = 0;
    int fcount = 0;

    // 分离BGR并变化成RGB
    for (int row = 0; row < INPUT_H; ++row) {
      uchar* uc_pixel = pre_img.data + row * pre_img.step;
      for (int col = 0; col < INPUT_W; ++col) {
        data[fcount * 3 * INPUT_H * INPUT_W + i] =
            static_cast<float>(uc_pixel[2]) / 255.0;
        data[fcount * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] =
            static_cast<float>(uc_pixel[1]) / 255.0;
        data[fcount * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] =
            static_cast<float>(uc_pixel[0]) / 255.0;
        uc_pixel += 3;
        ++i;
      }
    }
    // Run inference
    // doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
    cudaMemcpyAsync(buffers[0], data,
                    BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    context->enqueue(BATCH_SIZE, buffers, stream, nullptr);
    cudaMemcpyAsync(prob, buffers[1], BATCH_SIZE * OUTPUT_SIZE * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    auto end = std::chrono::system_clock::now();
    // std::cout << "prob: " << std::endl;
    // std::cout << prob[0] << " " << prob[1] << std::endl;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time used:%.2f ms\n", time);
    // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end -
    //                                                                    start)
    //                  .count()
    //           << "ms" << std::endl;
  }
}