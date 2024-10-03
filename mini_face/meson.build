pybind11_dep = dependency('pybind11', required: true)
py = import('python').find_installation(pure: false)

py.install_sources(
    [
        '__init__.py',
    ],
    subdir: 'mini_face'
)

subdir('open_face_api')
