// bindings.cpp — pybind11 wrapper around BatchedGridWorld.
//
// Numpy interop: we use py::array_t<T> with c_style + forcecast and request
// buffers directly. No copies on the hot path. Output arrays are allocated
// once on the Python side and passed in by reference, matching the SB3
// VecEnv style.
//
// Build: see CMakeLists.txt — produces `gridworld_fast.<ext>` importable from Python.

#include "env_fast.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <stdexcept>

namespace py = pybind11;
using namespace gridworld_fast;

namespace {

template <typename T>
T* request_ptr(py::array_t<T, py::array::c_style | py::array::forcecast>& arr,
               const char* name, std::ptrdiff_t expected) {
    auto info = arr.request();
    if (info.size != expected) {
        throw std::invalid_argument(std::string(name) +
            ": expected size " + std::to_string(expected) +
            ", got " + std::to_string(info.size));
    }
    return static_cast<T*>(info.ptr);
}

}  // namespace

PYBIND11_MODULE(gridworld_fast, m) {
    m.doc() = "Batched GridWorld with AVX2 SIMD over the batch dimension.";

    py::class_<GridWorldConfig>(m, "GridWorldConfig")
        .def(py::init<>())
        .def_readwrite("size",           &GridWorldConfig::size)
        .def_readwrite("max_steps",      &GridWorldConfig::max_steps)
        .def_readwrite("hidden_walls_p", &GridWorldConfig::hidden_walls_p)
        .def_readwrite("goal_reward",    &GridWorldConfig::goal_reward)
        .def_readwrite("wall_penalty",   &GridWorldConfig::wall_penalty)
        .def_readwrite("step_penalty",   &GridWorldConfig::step_penalty);

    py::class_<BatchedGridWorld>(m, "BatchedGridWorld")
        .def(py::init<int32_t, GridWorldConfig, uint64_t>(),
             py::arg("num_envs"),
             py::arg("config") = GridWorldConfig{},
             py::arg("seed")   = 0xC0FFEEull)
        .def_property_readonly("num_envs", &BatchedGridWorld::num_envs)
        .def_property_readonly("size",     &BatchedGridWorld::size)
        .def("reset_all",
             [](BatchedGridWorld& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> obs) {
                 float* obs_ptr = request_ptr<float>(obs, "obs", self.num_envs() * 5);
                 self.reset_all(obs_ptr);
             },
             py::arg("obs"),
             "Reset every env. obs must be shape [num_envs, 5] float32.")
        .def("reset_masked",
             [](BatchedGridWorld& self,
                py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask,
                py::array_t<float,   py::array::c_style | py::array::forcecast> obs) {
                 const uint8_t* m_ptr   = request_ptr<uint8_t>(mask, "mask", self.num_envs());
                 float*         obs_ptr = request_ptr<float>(obs, "obs", self.num_envs() * 5);
                 self.reset_masked(m_ptr, obs_ptr);
             },
             py::arg("mask"), py::arg("obs"))
        .def("step",
             [](BatchedGridWorld& self,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> actions,
                py::array_t<float,   py::array::c_style | py::array::forcecast> obs,
                py::array_t<float,   py::array::c_style | py::array::forcecast> rewards,
                py::array_t<uint8_t, py::array::c_style | py::array::forcecast> dones,
                py::array_t<uint8_t, py::array::c_style | py::array::forcecast> hit_wall) {
                 const int32_t n = self.num_envs();
                 const int32_t* a_ptr   = request_ptr<int32_t>(actions, "actions", n);
                 float*         obs_ptr = request_ptr<float>(obs, "obs", n * 5);
                 float*         r_ptr   = request_ptr<float>(rewards, "rewards", n);
                 uint8_t*       d_ptr   = request_ptr<uint8_t>(dones, "dones", n);
                 uint8_t*       w_ptr   = request_ptr<uint8_t>(hit_wall, "hit_wall", n);
                 self.step(a_ptr, obs_ptr, r_ptr, d_ptr, w_ptr);
             },
             py::arg("actions"),
             py::arg("obs"),
             py::arg("rewards"),
             py::arg("dones"),
             py::arg("hit_wall"));
}
