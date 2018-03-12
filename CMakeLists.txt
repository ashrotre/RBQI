cmake_minimum_required (VERSION 2.6)
project (RBQI)
set (CMAKE_CXX_STANDARD 11)
find_package (CUDA REQUIRED)
find_package (OpenCV REQUIRED)
find_package (Boost COMPONENTS filesystem system REQUIRED)
include_directories ("/usr/local/include")
if(UNIX AND NOT APPLE)
	find_package (OpenMP)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
set (SOURCE_FILES main.cpp
		  calcDistortionMaps.cpp
		  calcSimScore.cpp
		  ssim_CS_search.cpp
		  findBlockType.cpp
		  findAJNCD.cpp
		  calcDR.cpp
		  Utils.cpp)
set (HEADER_FILES Utils.hpp)
add_executable (calcRBQI ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries (calcRBQI ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
