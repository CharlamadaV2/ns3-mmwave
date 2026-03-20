# CMakeLists.txt for scratch/mmwave-sim
#
# The parent scratch/CMakeLists.txt delegates here via add_subdirectory() because
# this file exists. We bypass create_scratch() because that function derives the
# target name by stripping CMAKE_CURRENT_SOURCE_DIR from the source path — when
# called from a subdirectory scope both resolve to scratch/mmwave-sim/, leaving an
# empty dirname and producing target "scratch_sim" instead of the required
# "scratch_mmwave-sim_sim". We construct the target name explicitly instead.

# Add our own directory to include paths so that #include "src/..." resolves.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(MMWAVE_SIM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/sim.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config/config-loader.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sim/topology-builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sim/traffic-setup.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/metrics-writer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/viz-writer.cc
)

# Target name must match the ns3 script convention:
#   scratch/mmwave-sim/sim  ->  scratch_mmwave-sim_sim
add_executable(scratch_mmwave-sim_sim ${MMWAVE_SIM_SOURCES})

if(${NS3_STATIC})
  target_link_libraries(
    scratch_mmwave-sim_sim ${LIB_AS_NEEDED_PRE_STATIC} ${lib-ns3-static}
  )
else()
  target_link_libraries(
    scratch_mmwave-sim_sim ${LIB_AS_NEEDED_PRE} "${ns3-libs}" "${ns3-contrib-libs}"
    "${ns3-external-libs}" ${LIB_AS_NEEDED_POST}
  )
endif()

# ns3-mmwave headers have virtual functions without 'override' — suppress the
# warning so -Werror doesn't fail compilation of included ns3 headers.
target_compile_options(scratch_mmwave-sim_sim PRIVATE -Wno-inconsistent-missing-override)

# set_runtime_outputdirectory (defined in build-support/custom-modules/ns3-executables.cmake)
# does three things that our target requires:
#   1. Sets RUNTIME_OUTPUT_NAME  →  ns3.42-sim-debug  (so the binary is named correctly)
#   2. Sets RUNTIME_OUTPUT_DIRECTORY  →  build/scratch/mmwave-sim/
#   3. Appends build/scratch/mmwave-sim/ns3.42-sim-debug to the ns3-execs CMake cache
#      variable, which is written into .lock-ns3_* so the ns3 Python script can find us.
string(REPLACE "${PROJECT_SOURCE_DIR}" "${CMAKE_OUTPUT_DIRECTORY}"
               mmwave_sim_out_dir "${CMAKE_CURRENT_SOURCE_DIR}"
)
set_runtime_outputdirectory(sim "${mmwave_sim_out_dir}/" scratch_mmwave-sim_)
