#pragma once

#ifdef __CUDACC__

#include <cuda_runtime.h>
#define hipError_t            cudaError_t
#define hipSuccess            cudaSuccess
#define hipMalloc             cudaMalloc
#define hipFree               cudaFree
#define hipMemcpy             cudaMemcpy
#define hipMemset             cudaMemset
#define hipDeviceSynchronize  cudaDeviceSynchronize
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMallocPitch        cudaMallocPitch
#define hipMemcpy2D           cudaMemcpy2D
#define hipMemset2D           cudaMemset2D
#define hipExtent             cudaExtent
#define hipPitchedPtr         cudaPitchedPtr
#define hipMalloc3D           cudaMalloc3D
#define hipMemcpy3D           cudaMemcpy3D
#define hipMemset3D           cudaMemset3D
#define make_hipExtent        make_cudaExtent

#else
#include <hip/hip_runtime.h>
#endif
