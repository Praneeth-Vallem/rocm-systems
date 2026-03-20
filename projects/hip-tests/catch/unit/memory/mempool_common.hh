/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <resource_guards.hh>
#include <utils.hh>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <tchar.h>
#include <tlhelp32.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <sys/socket.h>
#include <memory.h>
#include <sys/un.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {
constexpr auto wait_ms = 500;
}  // anonymous namespace

#define checkMempoolSupported(device) {\
  int deviceSupportsMemoryPools = 0;\
  HIP_CHECK(hipDeviceGetAttribute(&deviceSupportsMemoryPools,\
        hipDeviceAttributeMemoryPoolsSupported, device));\
  if (0 == deviceSupportsMemoryPools) {\
    HipTest::HIP_SKIP_TEST("Memory Pool not supported. Skipping Test..");\
    return;\
  }\
}

#define checkIfMultiDev(numOfDev) {\
  if (numOfDev < 2) {\
    HipTest::HIP_SKIP_TEST("Multiple GPUs not available. Skipping Test..");\
    return;\
  }\
}

template <typename T> __global__ void kernel_500ms(T* host_res, int clk_rate) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  host_res[tid] = tid + 1;
  __threadfence_system();
  // expecting that the data is getting flushed to host here!
  uint64_t start = clock64() / clk_rate, cur;
  if (clk_rate > 1) {
    do {
      cur = clock64() / clk_rate - start;
    } while (cur < wait_ms);
  } else {
    do {
      cur = clock64() / start;
    } while (cur < wait_ms);
  }
}

template <typename T> __global__ void kernel_500ms_gfx11(T* host_res, int clk_rate) {
#if HT_AMD
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  host_res[tid] = tid + 1;
  __threadfence_system();
  // expecting that the data is getting flushed to host here!
  uint64_t start = clock_function() / clk_rate, cur;
  if (clk_rate > 1) {
    do {
      cur = clock_function() / clk_rate - start;
    } while (cur < wait_ms);
  } else {
    do {
      cur = clock_function() / start;
    } while (cur < wait_ms);
  }
#endif
}

template <typename T> __global__ void notifiedKernel(T* host_res, volatile unsigned int* notified) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  host_res[tid] = tid + 1;
  __threadfence_system();
  while (*notified == 0) { }
}

template <typename F> void MallocMemPoolAsync_OneAlloc(F malloc_func, const MemPools mempool_type) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }
  unsigned int *notified = nullptr;
  HIP_CHECK(hipHostMalloc(&notified, sizeof(unsigned int)));
  *notified = 0;
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);
  MemPoolGuard mempool(mempool_type, device_id);

  int* alloc_mem;
  StreamGuard stream(Streams::created);

  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem), allocation_size, mempool.mempool(),
                        stream.stream()));

  int blocks = 16;
  hipMemPoolAttr attr;
  notifiedKernel<<<blocks, 32, 0, stream.stream()>>>(alloc_mem, notified);

  const auto element_count = allocation_size / sizeof(int);
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int expected_value = 17;
  VectorSet<<<block_count, thread_count, 0, stream.stream()>>>(alloc_mem, expected_value,
                                                               element_count);

  HIP_CHECK(hipMemcpyAsync(host_alloc.host_ptr(), alloc_mem, allocation_size, hipMemcpyDeviceToHost,
                           stream.stream()));

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem), stream.stream()));

  attr = hipMemPoolAttrReservedMemCurrent;
  std::uint64_t res_before_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_before_sync));
  *notified = 1;
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::uint64_t res_after_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_after_sync));
  // Sync must release memory to OS
  REQUIRE(res_after_sync <= res_before_sync);

  std::uint64_t used_mem = 10;
  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  REQUIRE(0 == used_mem);

  ArrayFindIfNot(host_alloc.host_ptr(), expected_value, element_count);
  HIP_CHECK(hipHostFree(notified));
}

template <typename F>
void MallocMemPoolAsync_TwoAllocs(F malloc_func, const MemPools mempool_type) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }
  unsigned int *notified = nullptr;
  HIP_CHECK(hipHostMalloc(&notified, sizeof(unsigned int)));
  *notified = 0;
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);
  MemPoolGuard mempool(mempool_type, device_id);

  int* alloc_mem1;
  int* alloc_mem2;
  StreamGuard stream(Streams::created);

  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem1), allocation_size, mempool.mempool(),
                        stream.stream()));
  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem2), allocation_size, mempool.mempool(),
                        stream.stream()));

  int blocks = 16;
  hipMemPoolAttr attr;
  notifiedKernel<<<blocks, 32, 0, stream.stream()>>>(alloc_mem1, notified);

  const auto element_count = allocation_size / sizeof(int);
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int expected_value = 17;
  VectorSet<<<block_count, thread_count, 0, stream.stream()>>>(alloc_mem1, expected_value,
                                                               element_count);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpyAsync(alloc_mem2, alloc_mem1, allocation_size, hipMemcpyDeviceToDevice,
                           stream.stream()));

  HIP_CHECK(hipMemcpyAsync(host_alloc.host_ptr(), alloc_mem2, allocation_size,
                           hipMemcpyDeviceToHost, stream.stream()));

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream.stream()));

  attr = hipMemPoolAttrReservedMemCurrent;
  std::uint64_t res_before_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_before_sync));
  *notified = 1;
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::uint64_t res_after_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_after_sync));
  // Sync must release memory to OS
  REQUIRE(res_after_sync <= res_before_sync);

  std::uint64_t used_mem = 0;
  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  // Make sure the current usage query works - just second buffer is left
  REQUIRE(allocation_size == used_mem);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  // Make sure the high watermark usage works - both buffers must be reported
  REQUIRE((2 * allocation_size) == used_mem);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  // Make sure the current usage query works - none of the buffers are used
  REQUIRE(0 == used_mem);

  ArrayFindIfNot(host_alloc.host_ptr(), expected_value, element_count);
  HIP_CHECK(hipHostFree(notified));
}

template <typename F> void MallocMemPoolAsync_Reuse(F malloc_func, const MemPools mempool_type) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }
  unsigned int *notified = nullptr;
  HIP_CHECK(hipHostMalloc(&notified, sizeof(unsigned int)));
  *notified = 0;
  MemPoolGuard mempool(mempool_type, device_id);

  int *alloc_mem1, *alloc_mem2, *alloc_mem3;
  StreamGuard stream(Streams::created);

  size_t allocation_size1 = kPageSize * kPageSize * 2;
  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem1), allocation_size1, mempool.mempool(),
                        stream.stream()));

  size_t allocation_size2 = kPageSize;
  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem3), allocation_size2, mempool.mempool(),
                        stream.stream()));

  int blocks = 2;

  notifiedKernel<<<blocks, 32, 0, stream.stream()>>>(alloc_mem1, notified);

  hipMemPoolAttr attr;
  // Not a real free, since kernel isn't done
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream.stream()));

  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem2), allocation_size1, mempool.mempool(),
                        stream.stream()));
  // Runtime must reuse the pointer
  REQUIRE(alloc_mem1 == alloc_mem2);

  // Make a sync before the second kernel launch to make sure memory B isn't gone
  *notified = 1;
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  *notified = 0;
  // Second kernel launch with new memory
  notifiedKernel<<<blocks, 32, 0, stream.stream()>>>(alloc_mem2, notified);
  *notified = 1;
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  attr = hipMemPoolAttrUsedMemCurrent;
  std::uint64_t value64 = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the current usage reports the both buffers
  REQUIRE((allocation_size1 + allocation_size2) == value64);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the high watermark usage works - the both buffers must be reported
  REQUIRE((allocation_size1 + allocation_size2) == value64);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream.stream()));
  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the current usage reports just one buffer, because the above free doesn't hold memory
  REQUIRE(allocation_size2 == value64);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem3), stream.stream()));
  HIP_CHECK(hipHostFree(notified));
}

// definitions
#define THREADS_PER_BLOCK 512
#define LAUNCH_ITERATIONS 5
#define NUMBER_OF_THREADS 5
#define NUM_OF_STREAM 3

enum eTestValue {
  testdefault,
  testMaximum,
  testDisabled,
  testEnabled
};

class streamMemAllocTest {
  int *A_h, *B_h, *C_h;
  int *A_d, *B_d, *C_d;
  int size;
  size_t byte_size;
  hipMemPool_t mem_pool;

 public:
  explicit streamMemAllocTest(int N) : size(N) {
    byte_size = N*sizeof(int);
  }
  // Create host buffers and initialize them with input data
  void createHostBufferWithData() {
    A_h = reinterpret_cast<int*>(malloc(byte_size));
    REQUIRE(A_h != nullptr);
    B_h = reinterpret_cast<int*>(malloc(byte_size));
    REQUIRE(B_h != nullptr);
    C_h = reinterpret_cast<int*>(malloc(byte_size));
    REQUIRE(C_h != nullptr);
    // set data to host
    for (int i = 0; i < size; i++) {
      A_h[i] = 2*i + 1;  // Odd
      B_h[i] = 2*i;      // Even
      C_h[i] = 0;
    }
  }
  // Instead of creating a mempool in class use the global mempool.
  void useCommonMempool(hipMemPool_t mempool) {
    mem_pool = mempool;
  }
  // Create the mempool
  void createMempool(hipMemPoolAttr attr, enum eTestValue testtype,
                    int dev) {
    // Create mempool in current device
    hipMemPoolProps pool_props{};
    pool_props.allocType = hipMemAllocationTypePinned;
    pool_props.location.id = dev;
    pool_props.location.type = hipMemLocationTypeDevice;
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
    if (attr == hipMemPoolAttrReleaseThreshold) {
      uint64_t setThreshold = 0;
      if (testtype == testMaximum) {
        setThreshold = UINT64_MAX;
      }
      HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &setThreshold));
    } else if ((attr == hipMemPoolReuseFollowEventDependencies) ||
              (attr == hipMemPoolReuseAllowOpportunistic) ||
              (attr == hipMemPoolReuseAllowInternalDependencies)) {
      int value = 0;
      if (testtype == testEnabled) {
        value = 1;
      }
      HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));
    }
  }
  // allocate device memory from mempool.
  void allocFromMempool(hipStream_t stream) {
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d),
              byte_size, mem_pool, stream));
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B_d),
              byte_size, mem_pool, stream));
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&C_d),
              byte_size, mem_pool, stream));
  }
  // Transfer data from host to device asynchronously.
  void transferToMempool(hipStream_t stream) {
    HIP_CHECK(hipMemcpyAsync(A_d, A_h, byte_size, hipMemcpyHostToDevice,
              stream));
    HIP_CHECK(hipMemcpyAsync(B_d, B_h, byte_size, hipMemcpyHostToDevice,
              stream));
  }
  // allocate from default mempool.
  void allocFromDefMempool(hipStream_t stream) {
    HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&A_d),
              byte_size, stream));
    HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&B_d),
              byte_size, stream));
    HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&C_d),
              byte_size, stream));
  }
  // Execute Kernel to process input data and wait for it.
  void runKernel(hipStream_t stream) {
    int blocks = (size % THREADS_PER_BLOCK == 0) ? (size / THREADS_PER_BLOCK)
                                                 : ((size / THREADS_PER_BLOCK) + 1);
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(THREADS_PER_BLOCK), 0, stream,
                       static_cast<const int*>(A_d), static_cast<const int*>(B_d), C_d, size);
    HIP_CHECK(hipGetLastError());
  }
  // Transfer data from device to host asynchronously.
  void transferFromMempool(hipStream_t stream) {
    HIP_CHECK(hipMemcpyAsync(C_h, C_d, byte_size, hipMemcpyDeviceToHost,
                        stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  // Validate the data returned from device.
  bool validateResult() {
    for (int i = 0; i < size; i++) {
      auto res = A_h[i] + B_h[i];
      REQUIRE(res == C_h[i]);
    }
    return true;
  }
  // Free device memory
  void freeDevBuf(hipStream_t stream) {
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d), stream));
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B_d), stream));
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(C_d), stream));
  }
  // Free mempool if not using global mempool
  void freeMempool() {
    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
  // Free all host buffers
  void freeHostBuf() {
    free(A_h);
    free(B_h);
    free(C_h);
  }
};

#define checkSysCallErrors(result)                                                                 \
  if (result == -1) {                                                                              \
    fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__); exit(EXIT_FAILURE);                 \
  }

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
typedef PROCESS_INFORMATION Process;
typedef HANDLE hipShareableHdl;
#else
typedef pid_t Process;
#ifdef HT_AMD
typedef int64_t hipShareableHdl;
#else
typedef int hipShareableHdl;
#endif
#endif

struct sharedMemoryInfo {
  void *addr;
  size_t size;
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  HANDLE shmHandle;
#else
  int shmFd;
#endif
};

inline int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  info->size = sz;
  info->shmHandle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL,
                                      PAGE_READWRITE, 0, (DWORD)sz, name);
  if (info->shmHandle == 0) return GetLastError();
  info->addr = MapViewOfFile(info->shmHandle, FILE_MAP_ALL_ACCESS, 0, 0, sz);
  if (info->addr == NULL) return GetLastError();
  return 0;
#else
  info->size = sz;
  info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
  if (info->shmFd < 0) return errno;
  if (ftruncate(info->shmFd, sz) != 0) return errno;
  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == MAP_FAILED) return errno;
  return 0;
#endif
}

inline int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  info->size = sz;
  info->shmHandle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, name);
  if (info->shmHandle == 0) return GetLastError();
  info->addr = MapViewOfFile(info->shmHandle, FILE_MAP_ALL_ACCESS, 0, 0, sz);
  if (info->addr == NULL) return GetLastError();
  return 0;
#else
  info->size = sz;
  info->shmFd = shm_open(name, O_RDWR, 0777);
  if (info->shmFd < 0) return errno;
  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == MAP_FAILED) return errno;
  return 0;
#endif
}

inline void sharedMemoryClose(sharedMemoryInfo *info) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  if (info->addr) UnmapViewOfFile(info->addr);
  if (info->shmHandle) CloseHandle(info->shmHandle);
#else
  if (info->addr) munmap(info->addr, info->size);
  if (info->shmFd) close(info->shmFd);
#endif
}

inline int spawnProcess(Process *process, const char *app, char *const *args) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  STARTUPINFO si = {};
  std::string arg_string;
  memset(process, 0, sizeof(*process));
  while (*args) {
    arg_string.append(*args).append(1, ' ');
    args++;
  }
  BOOL status = CreateProcess(app, LPSTR(arg_string.c_str()), NULL, NULL, FALSE,
                              0, NULL, NULL, &si, process);
  return status ? 0 : GetLastError();
#else
  *process = fork();
  if (*process == 0) {
    if (0 > execvp(app, args)) return errno;
  } else if (*process < 0) {
    return errno;
  }
  return 0;
#endif
}

inline int waitProcess(Process *process) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  DWORD exitCode;
  WaitForSingleObject(process->hProcess, INFINITE);
  GetExitCodeProcess(process->hProcess, &exitCode);
  CloseHandle(process->hProcess);
  CloseHandle(process->hThread);
  return (int)exitCode;
#else
  int status = 0;
  do {
    if (0 > waitpid(*process, &status, 0)) return errno;
  } while (!WIFEXITED(status));
  return WEXITSTATUS(status);
#endif
}

inline void barrierWait(volatile int *barrier, volatile int *sense, unsigned int n) {
  int count;
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  count = InterlockedIncrement((volatile LONG *)barrier);
#else
  count = __sync_add_and_fetch((int *)barrier, 1);
#endif
  if ((unsigned int)count == n) {
    *barrier = 0;
    *sense = 1 - *sense;
  } else {
    int old_sense = *sense;
    while (*sense == old_sense) { }
  }
}

inline std::string getSelfExePath() {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  char path[MAX_PATH];
  DWORD len = GetModuleFileName(NULL, path, MAX_PATH);
  return std::string(path, len);
#else
  char path[4096];
  ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (len < 0) return "";
  path[len] = '\0';
  return std::string(path);
#endif
}

inline unsigned long getParentProcessId() {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  DWORD pid = GetCurrentProcessId();
  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snapshot == INVALID_HANDLE_VALUE) return 0;
  PROCESSENTRY32 pe = {};
  pe.dwSize = sizeof(pe);
  if (Process32First(snapshot, &pe)) {
    do {
      if (pe.th32ProcessID == pid) {
        CloseHandle(snapshot);
        return pe.th32ParentProcessID;
      }
    } while (Process32Next(snapshot, &pe));
  }
  CloseHandle(snapshot);
  return 0;
#else
  return static_cast<unsigned long>(getppid());
#endif
}

struct mempoolIpcShmStruct {
  hipMemPoolPtrExportData ptrExportData;
  hipMemAllocationHandleType handleType;
  int device;
  int barrier;
  int sense;
};

struct ipcHdl {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    HANDLE mailslot;
#else
    int socket;
#endif
    char *name;
};

class ipcSocketCom {
  ipcHdl *handle;

  int createSocket() {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    handle = new ipcHdl;
    if (nullptr == handle) {
      perror("Socket failure: Handle memory allocation failed");
      return -1;
    }
    handle->mailslot = INVALID_HANDLE_VALUE;
    handle->name = NULL;
    return 0;
#else
    int server_fd;
    struct sockaddr_un servaddr;

    char name[16];
    sprintf(name, "%u", getpid());

    handle = new ipcHdl;
    if (nullptr == handle) {
      perror("Socket failure: Handle memory allocation failed");
      return -1;
    }

    memset(handle, 0, sizeof(*handle));
    handle->socket = -1;
    handle->name = NULL;

    if ((server_fd = socket(AF_UNIX, SOCK_DGRAM, 0)) == 0) {
      perror("Socket failure: Socket creation failed");
      return -1;
    }

    unlink(name);
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sun_family = AF_UNIX;

    size_t len = strlen(name);
    if (len > (sizeof(servaddr.sun_path) - 1)) {
      perror("Socket failure: Cannot bind provided name to socket. Name too large");
      return -1;
    }

    strncpy(servaddr.sun_path, name, len);

    if (bind(server_fd, (struct sockaddr *)&servaddr, SUN_LEN(&servaddr)) < 0) {
      perror("Socket failure: Binding socket failed");
      return -1;
    }

    handle->name = new char[strlen(name) + 1];
    strcpy(handle->name, name);
    handle->socket = server_fd;
    return 0;
#endif
  }

  int openSocket() {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    handle = new ipcHdl;
    if (nullptr == handle) {
      perror("Socket failure: Handle memory allocation failed");
      return -1;
    }
    char name[128];
    sprintf(name, "\\\\.\\mailslot\\hipMemPoolIPC_%lu",
            (unsigned long)GetCurrentProcessId());
    handle->mailslot = CreateMailslot(name, 0, MAILSLOT_WAIT_FOREVER, NULL);
    if (handle->mailslot == INVALID_HANDLE_VALUE) {
      fprintf(stderr, "CreateMailslot failed (%lu)\n", GetLastError());
      return -1;
    }
    handle->name = new char[strlen(name) + 1];
    strcpy(handle->name, name);
    return 0;
#else
    int sock = 0;
    struct sockaddr_un cliaddr;

    handle = new ipcHdl;
    if (nullptr == handle) {
      perror("Socket failure: Handle memory allocation failed");
      return -1;
    }
    memset(handle, 0, sizeof(*handle));

    if ((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
      perror("IPC failure:Socket creation error");
      return -1;
    }

    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    char name[16];

    sprintf(name, "%u", getpid());

    strcpy(cliaddr.sun_path, name);
    if (bind(sock, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
      perror("Socket failure: Binding socket failed");
      return -1;
    }

    handle->socket = sock;
    handle->name = new char[strlen(name) + 1];
    strcpy(handle->name, name);

    return 0;
#endif
  }

  int closeSocket() {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    if (!handle) return -1;
    if (handle->mailslot != INVALID_HANDLE_VALUE) {
      CloseHandle(handle->mailslot);
    }
    if (handle->name) delete[] handle->name;
    delete handle;
    return 0;
#else
    if (!handle) {
      return -1;
    }

    if (handle->name) {
      unlink(handle->name);
      delete[] handle->name;
    }
    close(handle->socket);
    delete handle;
    return 0;
#endif
  }

public:
  ipcSocketCom() = default;
  ipcSocketCom(bool isServer) {
    if (isServer) {
      checkSysCallErrors(createSocket());
    } else {
      checkSysCallErrors(openSocket());
    }
  }
  ~ipcSocketCom() {
  }
  int closeThisSock() {
    return closeSocket();
  }

  int recvShareableHdl(hipShareableHdl *shHandle) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    DWORD cbRead = 0;
    if (!ReadFile(handle->mailslot, shHandle, sizeof(*shHandle), &cbRead, NULL)) {
      fprintf(stderr, "ReadFile failed (%lu)\n", GetLastError());
      return -1;
    }
    return 0;
#else
    struct msghdr msg = {};
    struct iovec iov[1];

    union {
      struct cmsghdr cm;
      char control[CMSG_SPACE(sizeof(int))];
    } control_un = {};

    struct cmsghdr *cmptr;
    ssize_t n;
    int receivedfd;
    int dummy_data;

    msg.msg_name = NULL;
    msg.msg_namelen = 0;
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);
    iov[0].iov_base = &dummy_data;
    iov[0].iov_len = sizeof(dummy_data);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    if ((n = recvmsg(handle->socket, &msg, 0)) <= 0) {
      perror("Socket failure: Receiving data over socket failed");
      return -1;
    }

    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) &&
       (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
      if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        return -1;
      }

      memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
      *(int *)shHandle = receivedfd;
    } else {
      return -1;
    }

    return 0;
#endif
  }

  int sendShareableHdl(hipShareableHdl shareableHdl, Process process) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    HANDLE hProcess = OpenProcess(PROCESS_DUP_HANDLE, FALSE, process.dwProcessId);
    if (hProcess == NULL) {
      fprintf(stderr, "OpenProcess failed (%lu)\n", GetLastError());
      return -1;
    }
    HANDLE hDup = INVALID_HANDLE_VALUE;
    if (!DuplicateHandle(GetCurrentProcess(), shareableHdl, hProcess,
                         &hDup, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
      CloseHandle(hProcess);
      fprintf(stderr, "DuplicateHandle failed (%lu)\n", GetLastError());
      return -1;
    }
    CloseHandle(hProcess);

    char slotName[128];
    sprintf(slotName, "\\\\.\\mailslot\\hipMemPoolIPC_%lu",
            (unsigned long)process.dwProcessId);
    HANDLE hFile = CreateFile(slotName, GENERIC_WRITE, FILE_SHARE_READ,
                              NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
      fprintf(stderr, "CreateFile for mailslot failed (%lu)\n", GetLastError());
      return -1;
    }
    DWORD cbWritten;
    if (!WriteFile(hFile, &hDup, sizeof(hDup), &cbWritten, NULL)) {
      CloseHandle(hFile);
      fprintf(stderr, "WriteFile failed (%lu)\n", GetLastError());
      return -1;
    }
    CloseHandle(hFile);
    return 0;
#else
    struct msghdr msg = {};
    struct iovec iov[1];
    int dummy_data = 0;

    union {
      struct cmsghdr cm;
      char control[CMSG_SPACE(sizeof(int))];
    } control_un = {};

    struct cmsghdr *cmptr;
    struct sockaddr_un cliaddr;

    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    strcpy(cliaddr.sun_path, std::to_string(process).c_str());

    int sendfd = (int)shareableHdl;

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;

    memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));

    msg.msg_name = (void *)&cliaddr;
    msg.msg_namelen = sizeof(struct sockaddr_un);
    iov[0].iov_base = &dummy_data;
    iov[0].iov_len = sizeof(dummy_data);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    ssize_t sendResult = sendmsg(handle->socket, &msg, 0);
    if (sendResult <= 0) {
      perror("Socket failure: Sending data over socket failed");
      return -1;
    }
    return 0;
#endif
  }
};
