
set(SOURCES_RS_TEST
    right_hand_side_kernel_test.cu
    ../src/cuda_src/right_hand_side_kernels.cu)

set(SOURCES_VEL_TEST
    velocity_kernels_test.cu
    ../src/cuda_src/velocity_kernels.cu)

set(SOURCES_DT_TEST
    timestep_kernels_test.cu
    ../src/cuda_src/timestep_kernels.cu)

set(SOURCES_DISCR_TEST
    discretization_test.cu
    discretization_host.cpp
    ../src/cuda_src/discretization.cu)


add_executable(rhs_test ${SOURCES_RS_TEST})
add_test(NAME rhs_test COMMAND rhs_test)
add_executable(vel_test ${SOURCES_VEL_TEST})
add_test(NAME vel_test COMMAND vel_test)
add_executable(dt_test ${SOURCES_DT_TEST})
add_test(NAME dt_test COMMAND dt_test)
add_executable(discretization_test ${SOURCES_DISCR_TEST})
add_test(NAME discretization_test COMMAND discretization_test)

target_link_libraries(rhs_test gtest gtest_main cuda)
target_link_libraries(vel_test gtest gtest_main cuda)
target_link_libraries(dt_test gtest gtest_main cuda)
target_link_libraries(discretization_test gtest gtest_main cuda)

set_target_properties(rhs_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61"
)
set_target_properties(vel_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61"
)
set_target_properties(dt_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61"
)
set_target_properties(discretization_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61"
)
target_include_directories(rhs_test PUBLIC ../include ../include/cuda_kernels ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_include_directories(vel_test PUBLIC ../include ../include/cuda_kernels ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_include_directories(dt_test PUBLIC ../include ../include/cuda_kernels ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_include_directories(discretization_test PUBLIC ../include ../include/cuda_kernels ${CUDA_TOOLKIT_ROOT_DIR}/include)