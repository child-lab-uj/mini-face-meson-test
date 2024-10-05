#include <iostream>

#include <pybind11/pybind11.h>

#include "open_face.h"


PYBIND11_MODULE(_bindings, handle) {
    handle.doc() = "pybind11 example module";
    handle.def("add", &add, "A function that adds two numbers");
}
