gpu_convolve_test
=================

A C++ test to compare cv::filter2D, cv::gpu::filter2D and cv::gpu::convolve, in OpenCV 2.4.9


To build:

cmake .

make


Usage:

./gpu_convolve_test [path_to_image [filter_wavelength_pixels [filter_orientation_degrees]]]

e.g.

./gpu_convolve_test

./gpu_convolve_test chiguiro.jpg 8

./gpu_convolve_test chiguiro.jpg 64 90
