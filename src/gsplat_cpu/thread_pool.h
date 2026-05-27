#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace gsplat_cpu {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t num_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void submit(std::function<void()> task);
    void wait();
    void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn);
    std::size_t size() const noexcept;

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::size_t num_threads_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::queue<std::function<void()>> tasks_;
    std::size_t in_flight_{0};
    bool stop_{false};
};

}  // namespace gsplat_cpu
