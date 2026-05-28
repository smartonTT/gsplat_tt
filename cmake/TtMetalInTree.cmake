# Link against an in-tree tt-metal build (no install / broken export symlinks).
function(gsplat_link_tt_metal target)
  if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "GSPLAT_WITH_TT requires TT_METAL_HOME")
  endif()
  set(TT_HOME "$ENV{TT_METAL_HOME}")
  set(TT_BUILD "${TT_HOME}/build")

  set(TT_METAL_SO "${TT_BUILD}/tt_metal/libtt_metal.so")
  if(NOT EXISTS "${TT_METAL_SO}")
    message(FATAL_ERROR "Missing ${TT_METAL_SO}; build tt-metal first")
  endif()

  set(_cpm_includes "")
  file(GLOB _cpm_roots LIST_DIRECTORIES true "${TT_HOME}/.cpmcache/*/*")
  foreach(_root IN LISTS _cpm_roots)
    if(EXISTS "${_root}/include")
      list(APPEND _cpm_includes "${_root}/include")
    endif()
    if(EXISTS "${_root}/enchantum/include")
      list(APPEND _cpm_includes "${_root}/enchantum/include")
    endif()
    if(EXISTS "${_root}/reflect")
      list(APPEND _cpm_includes "${_root}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _cpm_includes)

  target_include_directories(${target} SYSTEM PUBLIC
    "${TT_HOME}/tt_metal/api"
    "${TT_BUILD}/tt_metal/api"
    "${TT_HOME}/tt_stl"
    "${TT_HOME}/tt_metal/hostdevcommon/api"
    "${TT_HOME}/tt_metal/hw/inc/hostdev"
    "${TT_HOME}/tt_metal/third_party/umd/device/api"
    "${TT_HOME}/tt_metal/jit_build"
    "${TT_BUILD}/tt_metal/jit_build"
    ${_cpm_includes}
  )

  target_compile_definitions(${target} PRIVATE
    FMT_HEADER_ONLY=1
    NTEST
    SPDLOG_COMPILED_LIB
    SPDLOG_FMT_EXTERNAL
    TT_ENABLE_LIGHT_METAL_TRACE=1
  )

  target_compile_options(${target} PRIVATE
    -Wno-int-to-pointer-cast
    -Wno-deprecated-declarations
    -Wno-unused-parameter
    -Wno-missing-field-initializers
  )

  target_link_libraries(${target} PUBLIC "${TT_METAL_SO}")

  set(_tt_rpath
    "${TT_BUILD}/tt_metal"
    "${TT_BUILD}/tt_stl"
    "${TT_BUILD}/tt_metal/third_party/umd/lib"
    "${TT_BUILD}/lib"
  )
  set_target_properties(${target} PROPERTIES
    BUILD_RPATH "${_tt_rpath}"
    INSTALL_RPATH "${_tt_rpath}"
  )
endfunction()
