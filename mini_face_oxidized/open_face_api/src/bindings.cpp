#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "action_units.h"
#include "conversions.h"
#include "gaze.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(_bindings, handle) {
    py::class_<Gaze>(handle, "Gaze")
        .def(py::init<cv::Point3f, cv::Point3f, cv::Point3f, cv::Point3f, cv::Vec2d>())
        .def_property(
            "eye1",
            [](const Gaze &self) { return point3f_to_tuple(self.eye1); },
            [](Gaze &self, const py::tuple &t) { self.eye1 = tuple_to_point3f(t); }
        )
        .def_property(
            "direction1",
            [](const Gaze &self) { return point3f_to_tuple(self.direction1); },
            [](Gaze &self, const py::tuple &t) { self.direction1 = tuple_to_point3f(t); }
        )
        .def_property(
            "eye2",
            [](const Gaze &self) { return point3f_to_tuple(self.eye2); },
            [](Gaze &self, const py::tuple &t) { self.eye2 = tuple_to_point3f(t); }
        )
        .def_property(
            "direction2",
            [](const Gaze &self) { return point3f_to_tuple(self.direction2); },
            [](Gaze &self, const py::tuple &t) { self.direction2 = tuple_to_point3f(t); }
        )
        .def_property(
            "angle",
            [](const Gaze &self) { return vec2d_to_tuple(self.angle); },
            [](Gaze &self, const py::tuple &t) { self.angle = tuple_to_vec2d(t); }
        )
        .def("__repr__", [](const Gaze &gaze) -> std::string { return gaze.toString(); });

    py::class_<GazeExtractor>(handle, "GazeExtractor")
        .def(
            py::init<
                std::string,
                bool,
                std::optional<bool>,
                std::optional<bool>,
                std::optional<bool>,
                std::optional<int>,
                std::optional<float>,
                std::optional<float>>(),
            py::arg("model_loc"),
            py::arg("ld_video_mode") = false,
            py::arg("wild") = std::nullopt,
            py::arg("multi_view") = std::make_optional(true),
            py::arg("limit_pose") = std::nullopt,
            py::arg("n_iter") = std::nullopt,
            py::arg("reg_factor") = std::nullopt,
            py::arg("weight_factor") = std::nullopt,
            "Constructor for GazeExtractor.\n\n"
            ":param ld_video_mode: decides to use DetectLandmarksInVideo() or "
            "DetectLandmarksInImage() function.\n"
            ":param wild: a specyfic set of parameters for tough conditions (various lighting, "
            "incomplete view of the face).\n"
            ":param multi_view: (Recommended) decides whether to consider multiple views during "
            "model reinit.\n"
            ":param limit_pose: should pose be limited to 180 degrees frontal.\n"
            ":param n_iter: number of optimization iterations.\n"
            ":param reg_factor: regularization parameter.\n"
            ":param weight_factor: refers to how much weight is applied to certain constraints "
            "during the optimization process."
        )
        .def(
            "set_camera_calibration",
            &GazeExtractor::setCameraCalibration,
            py::arg("fx"),
            py::arg("fy"),
            py::arg("cx"),
            py::arg("cy")
        )
        .def(
            "detect_gaze",
            [](GazeExtractor &self,
               const py::array_t<uint8_t> &frame,
               double timestamp,
               const py::tuple &roi) -> std::optional<Gaze> {
                Frame frameMat = numpy_to_mat(frame);
                BoundingBox bbox = tuple_to_rect(roi);
                return self.detectGaze(frameMat, timestamp, bbox);
            },
            py::arg("frame"),
            py::arg("timestamp"),
            py::arg("roi")
        );

    py::class_<AUExtractor>(handle, "AUExtractor")
        .def(
            py::init<
                std::string,
                bool,
                bool,
                std::optional<bool>,
                std::optional<bool>,
                std::optional<bool>,
                std::optional<int>,
                std::optional<float>,
                std::optional<float>>(),
            py::arg("model_loc"),
            py::arg("ld_video_mode") = false,
            py::arg("fa_video_mode") = false,
            py::arg("wild") = std::nullopt,
            py::arg("multi_view") = std::make_optional(true),
            py::arg("limit_pose") = std::nullopt,
            py::arg("n_iter") = std::nullopt,
            py::arg("reg_factor") = std::nullopt,
            py::arg("weight_factor") = std::nullopt,
            "Constructor for AUExtractor.\n\n"
            ":param ld_video_mode: decides to use DetectLandmarksInVideo() or "
            "DetectLandmarksInImage() function.\n"
            ":param fa_video_mode: decides to use AddNextFrame() or "
            "PredictStaticAUsAndComputeFeatures() function.\n"
            ":param wild: a specyfic set of parameters for tough conditions (various lighting, "
            "incomplete view of the face).\n"
            ":param multi_view: (Recommended) decides whether to consider multiple views during "
            "model reinit.\n"
            ":param limit_pose: should pose be limited to 180 degrees frontal.\n"
            ":param n_iter: number of optimization iterations.\n"
            ":param reg_factor: regularization parameter.\n"
            ":param weight_factor: refers to how much weight is applied to certain constraints "
            "during the optimization process."
        )
        .def(
            "detect_au_presence",
            [](AUExtractor &self,
               const py::array_t<uint8_t> &frame,
               double timestamp,
               const py::tuple &roi) -> std::vector<std::pair<std::string, double>> {
                Frame frameMat = numpy_to_mat(frame);
                BoundingBox bbox = tuple_to_rect(roi);
                return self.detectActionUnitPresence(frameMat, timestamp, bbox);
            },
            py::arg("frame"),
            py::arg("timestamp"),
            py::arg("roi")
        )
        .def(
            "detect_au_intensity",
            [](AUExtractor &self,
               const py::array_t<uint8_t> &frame,
               double timestamp,
               const py::tuple &roi) -> std::vector<std::pair<std::string, double>> {
                Frame frameMat = numpy_to_mat(frame);
                BoundingBox bbox = tuple_to_rect(roi);
                return self.detectActionUnitIntensity(frameMat, timestamp, bbox);
            },
            py::arg("frame"),
            py::arg("timestamp"),
            py::arg("roi")
        );

#ifdef VERSION_INFO
    handle.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    handle.attr("__version__") = "dev";
#endif
}
