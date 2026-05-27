#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "gsplat_cpu/thread_pool.h"

TEST_CASE("ThreadPool: every task runs exactly once", "[thread_pool]") {
    gsplat_cpu::ThreadPool pool(4);
    constexpr std::size_t n = 10000;
    std::vector<std::atomic<int>> counters(n);
    for (auto& counter : counters) {
        counter.store(0);
    }

    pool.parallel_for(n, [&counters](std::size_t i) {
        counters[i].fetch_add(1);
    });

    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE(counters[i].load() == 1);
    }
}

TEST_CASE("ThreadPool: parallel_for n=0 is a no-op", "[thread_pool]") {
    gsplat_cpu::ThreadPool pool(4);
    bool called = false;
    pool.parallel_for(0, [&called](std::size_t) { called = true; });
    REQUIRE_FALSE(called);
}

TEST_CASE("ThreadPool: parallel_for n=1 still runs the task", "[thread_pool]") {
    gsplat_cpu::ThreadPool pool(4);
    std::atomic<int> count{0};
    pool.parallel_for(1, [&count](std::size_t i) {
        REQUIRE(i == 0);
        count.fetch_add(1);
    });
    REQUIRE(count.load() == 1);
}

TEST_CASE("ThreadPool: destructor joins cleanly without leaks", "[thread_pool]") {
    for (int i = 0; i < 100; ++i) {
        gsplat_cpu::ThreadPool pool(4);
        pool.parallel_for(1000, [](std::size_t) {});
    }
}

TEST_CASE("ThreadPool: hardware_concurrency default", "[thread_pool]") {
    gsplat_cpu::ThreadPool pool(0);
    REQUIRE(pool.size() > 0);
}

TEST_CASE("ThreadPool: tasks see distinct work units", "[thread_pool]") {
    gsplat_cpu::ThreadPool pool(4);
    constexpr std::size_t n = 1'000'000;
    std::atomic<std::uint64_t> sum{0};

    pool.parallel_for(n, [&sum](std::size_t i) {
        sum.fetch_add(static_cast<std::uint64_t>(i));
    });

    const std::uint64_t expected = static_cast<std::uint64_t>(n) * (n - 1) / 2;
    REQUIRE(sum.load() == expected);
}
