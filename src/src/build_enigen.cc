#include <cuda_runtime.h>

#include <iostream>
#include <string>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "argsParser.h"
#include "common.h"
#include "common/logger.h"

static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 2;
samplesCommon::Args gArgs;

const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";

// static Logger
// gLogger;  // 创建全局变量Logger，作为TensorRT各种调用的方法的输入参数

const std::string gSampleName = "TensorRT.sample_onnx_image";

// bool OnnxToTRTModel(const std::string& )

int main() {
  Logger gLogger;

  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  // 创建模型容器
  std::cout << "创建模型容器" << std::endl;
  nvinfer1::INetworkDefinition* network =
      builder->createNetworkV2(explicitBatch);

  // 开始填充模型
  std::cout << "开始填充模型" << std::endl;
  auto parser = nvonnxparser::createParser(*network, gLogger);
  const char* onnx_filename = "../gpu.onnx";
  parser->parseFromFile(onnx_filename,
                        static_cast<int>(Logger::Severity::kWARNING));

  // 开始构建enigen
  std::cout << "开始构建enigen" << std::endl;
  builder->setMaxBatchSize(1);  // 设置batch size
  //   builder->setMaxWorkspaceSize(1600 * (1 << 20));  // 最大占用显存1600M

  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1000 * (1 << 20));
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  std::cout << "Building engine, please wait for a while..." << std::endl;

  nvinfer1::ICudaEngine* enigen =
      builder->buildEngineWithConfig(*network, *config);
  std::cout << "Build engine successfully!" << std::endl;

  // 序列化模型保存
  std::cout << "序列化模型保存" << std::endl;
  nvinfer1::IHostMemory* giemodelstream = enigen->serialize();
  std::string serialize_str;
  std::ofstream serialize_output_stream;
  serialize_str.resize(giemodelstream->size());
  //   memcpy((void*)serialize_str.data(), giemodelstream->data(),
  //          giemodelstream->size());
  serialize_output_stream.open("./ghost.trt", std::ios::binary | std::ios::out);
  serialize_output_stream.write(
      reinterpret_cast<const char*>(giemodelstream->data()),
      giemodelstream->size());
  std::cout << "writing engine file..." << std::endl;
  //   serialize_output_stream << serialize_str;
  serialize_output_stream.close();
}
