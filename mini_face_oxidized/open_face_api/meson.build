vcpkg = find_program(
    'vcpkg',
    dirs: [
        '/Users/runner/work/mini-face-meson-test/mini-face-meson-test/vcpkg/',
        '/project/vcpkg'
    ],
    required: true
)

run_command(
    vcpkg,
    'install',
    '--x-wait-for-lock',
    '--x-manifest-root=' + meson.current_source_dir()
)

py = import('python').find_installation(pure: false)

vcpkg_include = [
    include_directories('vcpkg_installed/arm64-osx/include'),
    include_directories('vcpkg_installed/arm64-osx/include/openblas'),
    include_directories('vcpkg_installed/arm64-osx/include/opencv2'),
    include_directories('vcpkg_installed/arm64-osx/include/OpenCL'),
]

vcpkg_lib = [
    'vcpkg_installed/arm64-osx/lib/libdlib.a',
    'vcpkg_installed/arm64-osx/lib/libopenblas.a',
    'vcpkg_installed/arm64-osx/lib/libOpenCL.a',
    'vcpkg_installed/arm64-osx/lib/libopencv_core4.a',
#    'vcpkg_installed/arm64-osx/lib/libopencv_imgcodecs4.a',
#    'vcpkg_installed/arm64-osx/lib/libopencv_videoio4.a',
#    'vcpkg_installed/arm64-osx/lib/libopencv_ml4.a',
#    'vcpkg_installed/arm64-osx/lib/libopencv_calib3d4.a',
#    'vcpkg_installed/arm64-osx/lib/libopencv_flann4.a',
#    'vcpkg_installed/arm64-osx/lib/libopencv_photo4.a',
#    'vcpkg_installed/arm64-osx/lib/libopencv_features2d4.a',
#    'vcpkg_installed/arm64-osx/lib/libopencv_stitching4.a',
#    'vcpkg_installed/arm64-osx/lib/libopencv_video4.a',
#    'vcpkg_installed/arm64-osx/lib/libopencv_objdetect4.a',
#    'vcpkg_installed/arm64-osx/lib/libopencv_dnn4.a',
#    'vcpkg_installed/arm64-osx/lib/libopencv_imgproc4.a',
#    'vcpkg_installed/arm64-osx/lib/libopencv_highgui4.a',
]

bindings_source = [
    'src/action_units.cpp',
    'src/bindings.cpp',
    'src/conversions.cpp',
    'src/gaze.cpp',
    'src/landmarks.cpp',
]

bindings_include = [include_directories('include')]

open_face_source = [
    'open_face/src/SVM_dynamic_lin.cpp',
    'open_face/src/CEN_patch_expert.cpp',
    'open_face/src/SVM_static_lin.cpp',
    'open_face/src/SVR_dynamic_lin_regressors.cpp',
    'open_face/src/Face_utils.cpp',
    'open_face/src/LandmarkDetectorUtils.cpp',
    'open_face/src/CNN_utils.cpp',
    'open_face/src/Patch_experts.cpp',
    'open_face/src/GazeEstimation.cpp',
    'open_face/src/FaceAnalyserParameters.cpp',
    'open_face/src/LandmarkDetectorModel.cpp',
    'open_face/src/CCNF_patch_expert.cpp',
    'open_face/src/SVR_patch_expert.cpp',
    'open_face/src/PAW.cpp',
    'open_face/src/LandmarkDetectorParameters.cpp',
    'open_face/src/LandmarkDetectionValidator.cpp',
    'open_face/src/FaceAnalyser.cpp',
    'open_face/src/PDM.cpp',
    'open_face/src/FaceDetectorMTCNN.cpp',
    'open_face/src/LandmarkDetectorFunc.cpp',
    'open_face/src/SVR_static_lin_regressors.cpp',
]

open_face_include = [include_directories('open_face/include')]

py.extension_module(
    '_bindings',
    bindings_source + open_face_source,
    include_directories:
        bindings_include
        + open_face_include
        + vcpkg_include,
    objects: vcpkg_lib,
    install: true,
    subdir: 'mini_face_oxidized/open_face_api'
)

py.install_sources(
    [
        '__init__.py',
    ],
    subdir: 'mini_face_oxidized/open_face_api'
)
