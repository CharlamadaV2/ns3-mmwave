# CMakeLists.txt for scratch/mesh-sim
#
# The parent scratch/CMakeLists.txt delegates here via add_subdirectory() because
# this file exists. We bypass create_scratch() because that function derives the
# target name by stripping CMAKE_CURRENT_SOURCE_DIR from the source path — when
# called from a subdirectory scope both resolve to scratch/mesh-sim/, leaving an
# empty dirname and producing target "scratch_sim" instead of the required
# "scratch_mesh-sim_sim". We construct the target name explicitly instead.

set(MESH_SIM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/sim.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config/config-loader.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config/config-validator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/string-utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/ini-parser.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cli/cli-parser.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/setup/topology-builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/eval/link-evaluator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/eval/link-table.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/traffic/traffic-matrix.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/routing/mesh-router.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/viz-writer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/metrics-writer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rl/rl-bridge.cc
)

# Target name must match the ns3 script convention:
#   scratch/mesh-sim/sim  ->  scratch_mesh-sim_sim
add_executable(scratch_mesh-sim_sim ${MESH_SIM_SOURCES})
target_include_directories(scratch_mesh-sim_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(${NS3_STATIC})
  target_link_libraries(
    scratch_mesh-sim_sim ${LIB_AS_NEEDED_PRE_STATIC} ${lib-ns3-static}
  )
else()
  target_link_libraries(
    scratch_mesh-sim_sim ${LIB_AS_NEEDED_PRE} "${ns3-libs}" "${ns3-contrib-libs}"
    "${ns3-external-libs}" ${LIB_AS_NEEDED_POST}
  )
endif()

# ns3-mmwave headers have virtual functions without 'override' — suppress the
# warning so -Werror doesn't fail compilation of included ns3 headers.
target_compile_options(scratch_mesh-sim_sim PRIVATE -Wno-inconsistent-missing-override)

# set_runtime_outputdirectory (defined in build-support/custom-modules/ns3-executables.cmake)
# sets RUNTIME_OUTPUT_NAME, RUNTIME_OUTPUT_DIRECTORY, and registers the binary
# in the ns3-execs cache variable so the ns3 Python script can find it.
string(REPLACE "${PROJECT_SOURCE_DIR}" "${CMAKE_OUTPUT_DIRECTORY}"
               mesh_sim_out_dir "${CMAKE_CURRENT_SOURCE_DIR}"
)
set_runtime_outputdirectory(sim "${mesh_sim_out_dir}/" scratch_mesh-sim_)
