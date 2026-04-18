cmake_minimum_required(VERSION 3.15)
project(Aethelix VERSION 0.2.0 LANGUAGES C CXX)

# Setup dual-core architecture profiles
option(PROFILE_LEON3 "Target the sparc-unknown-none-elf architecture (LEON3 Industry Standard)" ON)
option(PROFILE_SHAKTI "Target the riscv32imac-unknown-none-elf architecture (Shakti/RISC-V New Norm)" OFF)

if(PROFILE_LEON3 AND PROFILE_SHAKTI)
    message(FATAL_ERROR "Cannot select both PROFILE_LEON3 and PROFILE_SHAKTI simultaneously.")
endif()

if(PROFILE_SHAKTI)
    set(RUST_TARGET "riscv32imac-unknown-none-elf")
    message(STATUS "Aethelix-PnP: Configuring for RISC-V Shakti/IRIS architecture (${RUST_TARGET})")
else()
    set(RUST_TARGET "sparc-unknown-none-elf")
    message(STATUS "Aethelix-PnP: Configuring for LEON3 SPARC architecture (${RUST_TARGET})")
endif()

# Fetch Corrosion (Rust <-> CMake bridge)
include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.0
)
FetchContent_MakeAvailable(Corrosion)

# Import the Rust core (aethelix_core)
corrosion_import_crate(MANIFEST_PATH rust_core/Cargo.toml
                       TARGET_CLIMBED aethelix_core
                       FLAGS --target ${RUST_TARGET} --features flight --no-default-features)

# Define the C/C++ interface library
add_library(aethelix INTERFACE)
target_include_directories(aethelix INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aethelix INTERFACE aethelix_core)

# For easy consumption via FetchContent by other projects
add_library(Aethelix::aethelix ALIAS aethelix)

message(STATUS "Aethelix FDIR Framework successfully registered. Target 'Aethelix::aethelix' is ready for linking.")
