#include "conversions.h"

namespace py = pybind11;

py::tuple point3f_to_tuple(const cv::Point3f &p) { return py::make_tuple(p.x, p.y, p.z); }

cv::Point3f tuple_to_point3f(const py::tuple &t) {
    return cv::Point3f(t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>());
}

py::tuple vec2d_to_tuple(const cv::Vec2d &v) { return py::make_tuple(v[0], v[1]); }

cv::Vec2d tuple_to_vec2d(const py::tuple &t) { return cv::Vec2d(t[0].cast<double>(), t[1].cast<double>()); }

cv::Mat numpy_to_mat(const py::array_t<uint8_t> &arr) {
    py::buffer_info buf_info = arr.request();

    size_t height = buf_info.shape[0];
    size_t width = buf_info.shape[1];
    size_t channels = buf_info.shape[2];

    // Very important - Python equivalent of cv::Mat uses 3 channels
    return cv::Mat(height, width, CV_8UC3, (uint8_t *) buf_info.ptr);
}

cv::Rect_<float> tuple_to_rect(const py::tuple &t) {
    return cv::Rect_<float>(t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>(), t[3].cast<float>());
}
