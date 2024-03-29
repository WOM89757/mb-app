#
# Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


cmake_minimum_required(VERSION 3.10)
 
set(UNIT_NAME "tracke")
set(UNIT_DEVICE "cpu")
set(UNIT_SECT "generic")

if(NOT DEFINED MODELBOX_PROJECT_VERSION_MAJOR)
    # build from flowunit cmakelists, not from project cmakelists
    set(MODELBOX_PROJECT_VERSION_MAJOR 0)
    set(MODELBOX_PROJECT_VERSION_MINOR 0)
    set(MODELBOX_PROJECT_VERSION_PATCH 1)
    set(RELEASE_PACKAGE_DIR_ROOT /opt/modelbox/flowunit/${UNIT_SECT} )
    set(RELEASE_PACKAGE_DIR_LIB ${RELEASE_PACKAGE_DIR_ROOT} ) 
    set(RELEASE_PACKAGE_DIR_BIN ${RELEASE_PACKAGE_DIR_ROOT} ) 
    if(${UNIT_DEVICE} STREQUAL "cuda" )
        find_package(CUDA 10.0)
    endif()
    
endif()

if (NOT OPENCV_FOUND) 
    message(FATAL_ERROR "Not found opencv")
endif()
find_package(Eigen3 3.3 REQUIRED)

if(${UNIT_DEVICE} STREQUAL "cuda" )
    if(NOT CUDA_FOUND)
        message(FATAL_ERROR "cannot find cuda in current environment ,please checkout you flowunit device type!")
    endif()
endif()
        
if(${UNIT_DEVICE} STREQUAL "ascend" )
    if(NOT ACL_FOUND OR NOT DSMI_FOUND)
        message(FATAL_ERROR "cannot find acl or dsmi in current environment ,please checkout you flowunit device type!")
    endif()
endif()
        
if(NOT EIGEN3_FOUND)
    message(FATAL_ERROR "cannot find eigen3 in current environment ,please try install by apt install libeigen3-dev !")
endif()
    
file(GLOB_RECURSE UNIT_SOURCE *.cpp *.cc *.c)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})


set(UNIT_SHARED modelbox-${CMAKE_PROJECT_NAME}-${UNIT_DEVICE}-${UNIT_NAME})
add_library(${UNIT_SHARED} SHARED ${UNIT_SOURCE})
 
set_target_properties(
  ${UNIT_SHARED} PROPERTIES
  SOVERSION ${MODELBOX_PROJECT_VERSION_MAJOR}
  VERSION ${MODELBOX_PROJECT_VERSION_MAJOR}.${MODELBOX_PROJECT_VERSION_MINOR}.${MODELBOX_PROJECT_VERSION_PATCH}
)

target_link_libraries(${UNIT_SHARED} pthread)
target_link_libraries(${UNIT_SHARED} rt)
target_link_libraries(${UNIT_SHARED} dl)
target_link_libraries(${UNIT_SHARED} modelbox)
target_link_libraries(${UNIT_SHARED} modelbox-device-${UNIT_DEVICE})
target_link_libraries(${UNIT_SHARED} ${OpenCV_LIBS})

install(TARGETS ${UNIT_SHARED} 
    COMPONENT ${UNIT_COMPONENT}
    RUNTIME DESTINATION ${RELEASE_PACKAGE_DIR_BIN}
    LIBRARY DESTINATION ${RELEASE_PACKAGE_DIR_LIB}
    ARCHIVE DESTINATION ${RELEASE_PACKAGE_DIR_LIB}
    OPTIONAL
    )

# for test
list(APPEND UNIT_TEST_TARGET ${UNIT_SHARED})
list(APPEND UNIT_TEST_LINK_LIBRARIES ${UNIT_LINK_LIBRARY})
set(UNIT_TEST_TARGET ${UNIT_TEST_TARGET} CACHE INTERNAL "")
set(UNIT_TEST_LINK_LIBRARIES ${UNIT_TEST_LINK_LIBRARIES} CACHE INTERNAL "")

