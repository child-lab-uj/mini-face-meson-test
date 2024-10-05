#pragma once

#include <cstdint>

#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::tuple point3f_to_tuple(const cv::Point3f &p);
cv::Point3f tuple_to_point3f(const py::tuple &t);
py::tuple vec2d_to_tuple(const cv::Vec2d &v);
cv::Vec2d tuple_to_vec2d(const py::tuple &t);
cv::Mat numpy_to_mat(const py::array_t<uint8_t> &arr);
cv::Rect_<float> tuple_to_rect(const py::tuple &t);
