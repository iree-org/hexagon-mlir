//===- HexagonThreadPool.h - Hexagon multi-threading threadpool manager ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_MULTITHREADING_HEXAGON_THREAD_POOL_H
#define HEXAGON_BIN_RUNTIME_MULTITHREADING_HEXAGON_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class HexagonThreadPool {
public:
  HexagonThreadPool(size_t numThreads = defaultThreadCount);
  ~HexagonThreadPool();

  void enqueueTask(std::function<void()> task);
  void wait();
  size_t getMaxConcurrency() const;

  // The calling thread runs queued tasks until done() returns true. It checks
  // done() before each task and, while the queue is empty, sleeps until a task
  // is enqueued or finishes, then checks again. Hence done() must only become
  // true as an effect of a task of this pool.
  void runTasksUntil(const std::function<bool()> &done);

  template <typename F> void async(F &&f);

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;

  std::mutex queueMutex;
  std::condition_variable condition;
  // Notified whenever a task is enqueued or finishes; threads in
  // runTasksUntil() wait on it.
  std::condition_variable runTasksUntilCondition;
  bool stop;

  std::atomic<int> activeTasks;
  std::condition_variable allTasksDone;
  std::mutex allTasksDoneMutex;

  void workerThread();

  static constexpr size_t defaultThreadCount = 8; // Default number of threads
};

template <typename F> inline void HexagonThreadPool::async(F &&f) {
  enqueueTask(std::forward<F>(f));
}

#endif // HEXAGON_BIN_RUNTIME_MULTITHREADING_HEXAGON_THREAD_POOL_H
