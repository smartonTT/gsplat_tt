#include "gsplat_cpu/thread_pool.h"

#include <condition_variable>
#include <mutex>
#include <queue>

namespace gsplat_cpu {

ThreadPool::ThreadPool(std::size_t num_threads)
    : num_threads_(num_threads == 0
                       ? std::max<std::size_t>(1, std::thread::hardware_concurrency())
                       : num_threads) {
    workers_.reserve(num_threads_);
    for (std::size_t i = 0; i < num_threads_; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::submit(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
}

void ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this] { return tasks_.empty() && in_flight_ == 0; });
}

void ThreadPool::parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn) {
    if (n == 0) {
        return;
    }

    const std::size_t workers = num_threads_;
    for (std::size_t w = 0; w < workers; ++w) {
        submit([w, workers, n, &fn]() {
            for (std::size_t i = w; i < n; i += workers) {
                fn(i);
            }
        });
    }
    wait();
}

std::size_t ThreadPool::size() const noexcept {
    return num_threads_;
}

void ThreadPool::worker_loop() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
            ++in_flight_;
        }

        task();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            --in_flight_;
            if (tasks_.empty() && in_flight_ == 0) {
                done_cv_.notify_all();
            }
        }
    }
}

}  // namespace gsplat_cpu
