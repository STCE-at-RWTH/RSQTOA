# **************************************************************************** #
# This file is part of the rsqtoa build system. 
#
# Version: 0.1.0
# Author:  Simon Maertens, STCE (info@stce.rwth-aachen.de)
# **************************************************************************** #

if (RSQTOA_BUILD_CASE_STUDIES)
  # ************************************************************************** #
  # Build case studies
  # ************************************************************************** #
  file(GLOB_RECURSE case_study_files "src/*.cpp")

  # Make log directory
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/log")

  # Make directory for epsilons
  file(MAKE_DIRECTORY "${RSQTOA_GEN_DIRECTORY}/case_studies")

  # Build test executables
  foreach(file ${case_study_files})
    # Get test name
    get_filename_component(cs_name ${file} NAME_WE)

    # Output folder
    set(FULL_OUT_DIR "${RSQTOA_GEN_DIRECTORY}/case_studies/${cs_name}_full")

    # Build executable with full model
    add_executable(${cs_name}_full ${file})
    target_include_directories(${cs_name}_full SYSTEM PRIVATE ${extern_includes})
    target_include_directories(${cs_name}_full PRIVATE ${case_study_include_dirs})
    target_compile_definitions(${cs_name}_full PRIVATE ${CXX_DCO_DEFINES})
    target_compile_options(${cs_name}_full PRIVATE ${CXX_WARN_OPTIONS})
    target_link_libraries(${cs_name}_full PRIVATE
      ${TensorFlow_LIBRARY} ${extern_targets})
    if(OpenMP_CXX_FOUND)
      target_link_libraries(${cs_name}_full PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_dependencies(${cs_name}_full tf_api)
    target_compile_definitions(${cs_name}_full PRIVATE
      "RSQTOA_LEARN_ORIGINAL_MODEL"
      "RSQTOA_OUTPUT_FOLDER=\"${FULL_OUT_DIR}\"")

    # Check if epsilon exists
    set(eps_file "${FULL_OUT_DIR}/trained_model/epsilon.txt")
    set(eps_rqstoa_file "${FULL_OUT_DIR}/trained_model/epsilon_rsqtoa.txt")
    watch(${eps_file})

    if(EXISTS ${eps_file})
      # Output folder
      set(OUT_DIR "${RSQTOA_GEN_DIRECTORY}/case_studies/${cs_name}_rsqtoa")

      # Generate reduced model
      rsqtoa_generate_reduced_model_cxx(${file} ${eps_file} ${eps_rqstoa_file})

      # Build executable with reduced model
      add_executable(${cs_name}_rsqtoa ${file})
      target_include_directories(${cs_name}_rsqtoa SYSTEM PRIVATE ${extern_includes})
      target_include_directories(${cs_name}_rsqtoa PRIVATE ${case_study_include_dirs})
      target_compile_definitions(${cs_name}_rsqtoa PRIVATE ${CXX_DCO_DEFINES})
      target_compile_options(${cs_name}_rsqtoa PRIVATE ${CXX_WARN_OPTIONS})
      target_link_libraries(${cs_name}_rsqtoa PRIVATE
        ${TensorFlow_LIBRARY} ${extern_targets})
      if(OpenMP_CXX_FOUND)
        target_link_libraries(${cs_name}_rsqtoa PRIVATE OpenMP::OpenMP_CXX)
      endif()
      add_dependencies(${cs_name}_rsqtoa ${cs_name}_reduced_model tf_api)
      target_compile_definitions(${cs_name}_rsqtoa PRIVATE
        "RSQTOA_OUTPUT_FOLDER=\"${OUT_DIR}\""
        "RSQTOA_EPSILON_FILE=\"${eps_rqstoa_file}\"")
    endif()
  endforeach()

  # Cleanup
  unset(case_study_files)
endif()
