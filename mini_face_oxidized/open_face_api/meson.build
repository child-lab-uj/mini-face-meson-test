vcpkg = find_program(
    'vcpkg',
    dirs: ['/Users/runner/work/mini-face-meson-test/mini-face-meson-test/vcpkg/', '/project/vcpkg'],
    required: true
)

run_command(
    vcpkg,
    'install',
    '--x-wait-for-lock',
    '--x-manifest-root=' + meson.current_source_dir()
)

pybind11_dep = dependency('pybind11', required: true)
py = import('python').find_installation(pure: false)

py.extension_module(
    'bindings',
    [
        'src/bindings.cpp',
        'src/open_face.h',
        'src/open_face.cpp',
    ],
    dependencies: pybind11_dep,
    install: true,
    subdir: 'mini_face_oxidized/open_face_api'
)

py.install_sources(
    '__init__.py',
    subdir: 'mini_face_oxidized/open_face_api'
)
