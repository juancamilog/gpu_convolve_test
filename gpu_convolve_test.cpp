#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <string>
#include <cstdlib>
#include <iostream>
#define M_PI           3.14159265358979323846

int main(int argc, char** argv){
    double lambda = 16.0;
    double theta = 0.0;
    std::string filename("chiguiro.jpg");
    int wflags = CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED;
    if (argc >= 2){
        filename = std::string(argv[1]);
    }
    if (argc >= 3){
        lambda = std::atof(argv[2]);
    }
    if (argc >= 4){
        theta = std::atof(argv[3]);
    }


    cv::Mat image, image_gray, image_f, sin_response, cos_response, temp;

    //load the image
    image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(image, image_gray, CV_BGR2GRAY);
    std::cout<<"Original image size: "<<image.size()<<std::endl;

    // generate gabor kernels
    double sigma = 0.5*lambda;
    cv::Mat sin_gabor = cv::getGaborKernel(cv::Size(), sigma, theta*M_PI/180.0, lambda, 1.0, M_PI/2.0, CV_32F);
    cv::Mat cos_gabor = cv::getGaborKernel(cv::Size(), sigma, theta*M_PI/180.0, lambda, 1.0, 0.0, CV_32F);
    std::cout<<"Kernel size: "<<sin_gabor.size()<<std::endl;

    // get filter responses (CPU)
    image_gray.convertTo(image_f, CV_32F, 1.0/256.0);
    cv::filter2D(image_f,sin_response, -1, sin_gabor, cv::Point(-1,-1));
    cv::filter2D(image_f,cos_response, -1, cos_gabor, cv::Point(-1,-1));
    cv::multiply(sin_response, sin_response, sin_response);
    cv::multiply(cos_response, cos_response, cos_response);
    
    // get filter responses (GPU)
    cv::gpu::GpuMat image_d, image_f_d, sin_gabor_d, cos_gabor_d, padded_image_d;
    cv::gpu::GpuMat sin_response_d, cos_response_d;

    image_d.upload(image_gray);
    image_d.convertTo(image_f_d, CV_32F, 1.0/256.0);
    sin_gabor_d.upload(sin_gabor);
    cos_gabor_d.upload(cos_gabor);

    if (cos_gabor.rows*cos_gabor.cols<=16*16){
        // cv::gpu::filter2D is limited to 16*16 kernels
        cv::gpu::filter2D(image_f_d, sin_response_d, -1, sin_gabor, cv::Point(-1,-1));
        cv::gpu::filter2D(image_f_d, cos_response_d, -1, cos_gabor, cv::Point(-1,-1));
    }
    else{
        int vertical_pad = (cos_gabor.rows-1)/2;
        int horizontal_pad = (cos_gabor.cols-1)/2;
        cv::gpu::copyMakeBorder(image_f_d,padded_image_d,
                                vertical_pad,vertical_pad,
                                horizontal_pad,horizontal_pad,
                                cv::BORDER_DEFAULT, 0.0);
	padded_image_d.download(temp);
	cv::imshow("Padded Image", temp);
	std::cout<<"Padded image size: "<<padded_image_d.size()<<std::endl;

        cv::gpu::convolve(padded_image_d, sin_gabor_d, sin_response_d);
        cv::gpu::convolve(padded_image_d, cos_gabor_d, cos_response_d);
    }
    cv::gpu::multiply(sin_response_d, sin_response_d, sin_response_d);
    cv::gpu::multiply(cos_response_d, cos_response_d, cos_response_d);

    std::cout<<"response size: "<<sin_response_d.size()<<std::endl;

    // Display results
    cv::imshow("Original Image", image);
    cv::imshow("Grayscale Image", image_gray);
    cv::imshow("Sine Gabor Kernel", sin_gabor+0.5);
    cv::imshow("Cosine Gabor Kernel", cos_gabor+0.5);
    cv::imshow("Sine Response (CPU)", sin_response);
    cv::imshow("Cosine Response (CPU)", cos_response);
    sin_response_d.download(temp);
    cv::imshow("Sine Response (GPU)", temp);
    cos_response_d.download(temp);
    cv::imshow("Cosine Response (GPU)", temp);

    
    // Quit
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
