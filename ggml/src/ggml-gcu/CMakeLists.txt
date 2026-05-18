# ggml-gcu: Enflame GCU (TOPS) backend
#
# Required SDK packages on the build host (Debian):
#   tops-sdk  - provides /opt/tops with runtime headers + libtopsrt.so
#   topsaten  - provides /usr/lib/libtopsaten.so + headers under
#               /opt/tops/include/gcu/topsaten

if (NOT DEFINED TOPS_INSTALL_DIR)
    if (DEFINED ENV{TOPS_INSTALL_DIR})
        set(TOPS_INSTALL_DIR $ENV{TOPS_INSTALL_DIR})
    else()
        set(TOPS_INSTALL_DIR "/opt/tops")
    endif()
endif()

if (NOT EXISTS "${TOPS_INSTALL_DIR}/include/gcu/topsaten/topsaten.h")
    message(FATAL_ERROR
        "ggml-gcu: topsaten headers not found at ${TOPS_INSTALL_DIR}/include/gcu/topsaten. "
        "Install the tops-sdk and topsaten packages, or pass -DTOPS_INSTALL_DIR=...")
endif()

# Explicit source list (preferred over a glob so a missed file fails the
# build loudly rather than silently dropping a translation unit).
set(GGML_SOURCES_GCU
    ggml-gcu.cpp
    common.cpp
    gcu_pool.cpp
    gcu_buffer.cpp
    gcu_op_elementwise.cpp
    gcu_op_norm.cpp
    gcu_op_matmul.cpp
    gcu_op_attn.cpp
    gcu_op_misc.cpp
)

ggml_add_backend_library(ggml-gcu ${GGML_SOURCES_GCU})

target_include_directories(ggml-gcu PRIVATE
    ${TOPS_INSTALL_DIR}/include
    ${TOPS_INSTALL_DIR}/include/gcu)

# libtopsaten.so lives in /usr/lib (system path) so the linker finds it
# without help. libtopsrt.so is at ${TOPS_INSTALL_DIR}/lib so we add it.
target_link_directories(ggml-gcu PRIVATE ${TOPS_INSTALL_DIR}/lib)
target_link_libraries(ggml-gcu PRIVATE topsaten topsrt)

message(STATUS "ggml-gcu: TOPS_INSTALL_DIR = ${TOPS_INSTALL_DIR}")
