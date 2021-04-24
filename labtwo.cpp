#include <opencv2/opencv.hpp>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>


std::vector<cv::Mat> calculateHistogram(cv::Mat& img);
cv::Mat bgrEqualization(cv::Mat& img);
cv::Mat luminanceEqualization(cv::Mat& img);
void showHistogram(std::vector<cv::Mat>& hists);


int main(int argc, char** argv)
{
    // STEP 1 - Load the original image
	cv::Mat originalImage = cv::imread("data/barbecue.png");
	cv::imshow("Original Image", originalImage);
	cv::waitKey();


	// STEP 2 - Calculate and show BGR histogram of the original image
    std::vector<cv::Mat> originalImageHistogram(3);
    originalImageHistogram = calculateHistogram(originalImage);
    showHistogram(originalImageHistogram);
    cv::waitKey();


    // STEP 3 - Equalize the original image using cv::equalizeHist() function
    cv::Mat equalizedImage = bgrEqualization(originalImage);
    cv::imshow("BGR Equalized Image", equalizedImage);
    cv::waitKey();


    // STEP 4 - Calculate and show the BGR histogram of the equalized image
    std::vector<cv::Mat> equalizedImageHistogram(3);
    equalizedImageHistogram = calculateHistogram(equalizedImage);
    showHistogram(equalizedImageHistogram);
    cv::waitKey();


    // STEP 5 - Equalize the original image using luminance equalization
    cv::Mat luminanceEqualizedImage = luminanceEqualization(originalImage);
    cv::imshow("Luminance Equalized Image", luminanceEqualizedImage);
    cv::waitKey();

    // Calculate and show histogram
    std::vector<cv::Mat> luminanceEqualizedImageHistogram(3);
    luminanceEqualizedImageHistogram = calculateHistogram(luminanceEqualizedImage);
    showHistogram(luminanceEqualizedImageHistogram);


	// wait for a key to be pressed and then close all
	cv::waitKey();
	cv::destroyAllWindows();

	return 0;
}

/**
 * This function will calculate the histogram of a 3 channels (BGR) color image.
 * Then it will return vector containing the B,G,R histograms.
 */
std::vector<cv::Mat> calculateHistogram(cv::Mat& img)
{
    // Separating the image into 3 channels (B, G, and R)
    std::vector<cv::Mat> bgr_channels;
    cv::split(img, bgr_channels);

    // Number of bins
    int histogramSize = 256;

    // Histogram parameter;
    bool uniform = true;
    bool accumulate = false;

    // Setting the range of B,G,R channels
    float range[] = {0, 256};
    const float* histogramRange = {range};

    // Calculating the histogram
    cv::Mat b_histogram;
    cv::Mat g_histogram;
    cv::Mat r_histogram;
    cv::calcHist(&bgr_channels[0], 1, 0, cv::Mat(), b_histogram, 1, &histogramSize, &histogramRange, uniform, accumulate);
    cv::calcHist(&bgr_channels[1], 1, 0, cv::Mat(), g_histogram, 1, &histogramSize, &histogramRange, uniform, accumulate);
    cv::calcHist(&bgr_channels[2], 1, 0, cv::Mat(), r_histogram, 1, &histogramSize, &histogramRange, uniform, accumulate);

    // Creating a 3 channels vector for the BGR Histogram
    std::vector<cv::Mat> bgr_histogram(3);
    bgr_histogram[0] = b_histogram;
    bgr_histogram[1] = g_histogram;
    bgr_histogram[2] = r_histogram;

    return bgr_histogram;
}

/**
 * This function will equalize the BGR histograms of the input image using
 * the cv::equalHist() function.
 */
cv::Mat bgrEqualization(cv::Mat& img)
{
    // Separating the image into 3 channels (B, G, and R)
    std::vector<cv::Mat> bgr_channels;
    cv::split(img, bgr_channels);

    // Applying histogram equalization to the image

    cv::equalizeHist(bgr_channels[0], bgr_channels[0]);
    cv::equalizeHist(bgr_channels[1], bgr_channels[1]);
    cv::equalizeHist(bgr_channels[2], bgr_channels[2]);

    // Merging the BGR equalized images into a single image
    cv::Mat equalizedImage;
    cv::merge(bgr_channels, equalizedImage);

    return equalizedImage;
}

/**
 * This function will equalize the input image using the luminance equalization,
 * where only L channel that is equalized.
 */
cv::Mat luminanceEqualization(cv::Mat& img)
{
    // Convert the input image from BGR color space to Lab color space
    cv::Mat Lab_image;
    cv::cvtColor(img, Lab_image, cv::COLOR_BGR2Lab);

    // Separating the image into 3 channels (L, a, and b)
    std::vector<cv::Mat> Lab_channels;
    cv::split(Lab_image, Lab_channels);

    // Applying histogram equalization to the Luminance channel
    cv::equalizeHist(Lab_channels[0], Lab_channels[0]);

    // Merging the BGR equalized images into a single image
    cv::Mat luminanceEqualizedImage;
    cv::merge(Lab_channels, luminanceEqualizedImage);

    // Convert the image from Lab color space back to BGR color space
    cv::cvtColor(luminanceEqualizedImage,luminanceEqualizedImage, cv::COLOR_Lab2BGR);

    return luminanceEqualizedImage;
}

/**
 * This function will visualize the BGR histograms.
 * @param hists
 */
void showHistogram(std::vector<cv::Mat>& hists)
{
    // Min/Max computation
    double hmax[3] = {0,0,0};
    double min;
    cv::minMaxLoc(hists[0], &min, &hmax[0]);
    cv::minMaxLoc(hists[1], &min, &hmax[1]);
    cv::minMaxLoc(hists[2], &min, &hmax[2]);

    std::string wname[3] = { "blue", "green", "red" };
    cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
                             cv::Scalar(0,0,255) };

    std::vector<cv::Mat> canvas(hists.size());

    // Display each histogram in a canvas
    for (int i = 0, end = hists.size(); i < end; i++)
    {
        canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < hists[0].rows-1; j++)
        {
            cv::line(
                    canvas[i],
                    cv::Point(j, rows),
                    cv::Point(j, rows - (hists[i].at<float>(j) * rows/hmax[i])),
                    hists.size() == 1 ? cv::Scalar(200,200,200) : colors[i],
                    1, 8, 0
            );
        }

        cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
    }
}
