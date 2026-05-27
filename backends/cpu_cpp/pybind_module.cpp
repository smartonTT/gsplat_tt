#include <pybind11/pybind11.h>

#include "gsplat_cpu/thread_pool.h"

namespace py = pybind11;

PYBIND11_MODULE(_gsplat_cpu, m) {
    m.doc() = "gsplat_cpu — C++ rendering backend (iter-001 scaffolding)";
    m.def("hello", []() { return std::string("hello from gsplat_cpu"); });

    py::class_<gsplat_cpu::ThreadPool>(m, "ThreadPool")
        .def(py::init<std::size_t>(), py::arg("num_threads") = 0)
        .def("size", &gsplat_cpu::ThreadPool::size)
        .def("wait", &gsplat_cpu::ThreadPool::wait);
}
