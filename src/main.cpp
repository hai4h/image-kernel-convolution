#include <opencv2/opencv.hpp>
#include <iostream>
#include "image_processing.h"
#include "opencv2/core.hpp"
#include <omp.h>

int main() {
    omp_set_num_threads(omp_get_max_threads());
    // load the image
    cv::Mat image = cv::imread("/home/haiah/Documents/GitHub/image-kernel-convolution/image/image.jpg");
    if (image.empty()) {
        std::cerr << "There is no such image in the directory" << std::endl;
        return -1;
    }

    // turn the loaded image in to grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    double itime, ftime, exec_time;
    itime = omp_get_wtime();

    // all-in-one edge filtering
    cv::Mat outputImage = cv::Mat::zeros(grayImage.size(), CV_8U);
    edgeFilter(grayImage, outputImage);

    ftime = omp_get_wtime();
    exec_time = ftime - itime;
    printf("\n\nTime taken is %f", exec_time);

    // Create a combined image to display both images side by side
    cv::Mat combinedImage;
    
    // Convert output image to 3 channels to match the original image
    cv::Mat outputImage3C;
    cv::cvtColor(outputImage, outputImage3C, cv::COLOR_GRAY2BGR);
    
    // Concatenate images horizontally
    cv::hconcat(image, outputImage3C, combinedImage);

    // Set the display window name
    std::string window_name = "Original vs Processed Image";

    // Display the combined image
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 2048, 1024); // Double the width to accommodate both images
    cv::imshow(window_name, combinedImage);

    // Add labels to the window
    int baseline = 0;
    cv::Size textSize = cv::getTextSize("Original Image", cv::FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);
    
    // Draw labels on the combined image
    cv::putText(combinedImage, "Original Image", 
                cv::Point(image.cols/2 - textSize.width/2, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    
    cv::putText(combinedImage, "Processed Image", 
                cv::Point(image.cols + image.cols/2 - textSize.width/2, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

    // wait for key press indefinitely
    cv::waitKey(0);

    // destroy the display window
    cv::destroyWindow(window_name);

    return 0;
}
