cmake_minimum_required(VERSION 3.12...3.31)
project(matrix_service CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(DownloadUserver)

find_package(userver COMPONENTS core QUIET)
if(NOT userver_FOUND)
  # Tries TRY_DIR first, falls back to downloading userver from GitHub using CPM.
  download_userver(TRY_DIR third_party/userver VERSION 2.13)
endif()

userver_setup_environment()

file(GLOB_RECURSE PROJECT_CORE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/routes/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/*.cpp
)
add_library(${PROJECT_NAME}_objs OBJECT ${PROJECT_CORE_SOURCES})
target_include_directories(${PROJECT_NAME}_objs PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(${PROJECT_NAME}_objs PUBLIC userver::core)

file(GLOB PROJECT_MAIN_SOURCE CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)
add_executable(${PROJECT_NAME} ${PROJECT_MAIN_SOURCE})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_objs)

file(GLOB_RECURSE PROJECT_TEST_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cpp
)
add_executable(${PROJECT_NAME}_unittest ${PROJECT_TEST_SOURCES})
target_link_libraries(${PROJECT_NAME}_unittest PRIVATE ${PROJECT_NAME}_objs userver::utest)
add_google_tests(${PROJECT_NAME}_unittest)

file(GLOB_RECURSE PROJECT_BENCH_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/*.cpp
)
add_executable(${PROJECT_NAME}_benchmark ${PROJECT_BENCH_SOURCES})
target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE ${PROJECT_NAME}_objs userver::ubench)
add_google_benchmark_tests(${PROJECT_NAME}_benchmark)

# Functional testing
userver_testsuite_add_simple()

# Install
include(GNUInstallDirs)

if(DEFINED ENV{PREFIX})
  message(STATUS "Set install prefix: $ENV{PREFIX}")
  file(TO_CMAKE_PATH "$ENV{PREFIX}" PREFIX_PATH)
  set(CMAKE_INSTALL_PREFIX "${PREFIX_PATH}")
endif()

file(GLOB CONFIGS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/*.yaml ${CMAKE_CURRENT_SOURCE_DIR}/configs/*.json)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${PROJECT_NAME})
install(FILES ${CONFIGS_FILES} DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/${PROJECT_NAME} COMPONENT ${PROJECT_NAME})
