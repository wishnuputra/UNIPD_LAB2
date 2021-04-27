#include <opencv2/opencv.hpp>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>


cv::Mat luminanceEqualizedImage;

// Function Header of PART-1
std::vector<cv::Mat> calculateHistogram(cv::Mat& img);
cv::Mat rgbEqualization(cv::Mat& img);
cv::Mat luminanceEqualization(cv::Mat& img);
void showHistogram(std::vector<cv::Mat>& hists);

// Function Header of PART 2
static void medianFilter( int, void* );
static void gaussianFilter(int, void*);
static void bilateralFilter(int, void*);


int main(int argc, char** argv)
{
    /****************** PART 1 - HISTOGRAM EQUALIZATION ******************/

    // STEP 1 - Load the original image
    cv::Mat originalImage = cv::imread("data/barbecue.png");
    cv::imshow("Original Image", originalImage);

    // STEP 2 - Calculate and show BGR histogram of the original image
    std::vector<cv::Mat> originalImageHistogram(3);
    originalImageHistogram = calculateHistogram(originalImage);
    showHistogram(originalImageHistogram);
    cv::waitKey();

    // STEP 3 - Equalize the original image using RGB Equalization
    cv::Mat equalizedImage = rgbEqualization(originalImage);
    cv::imshow("RGB Equalized Image", equalizedImage);

    // STEP 4 - Calculate and show the BGR histogram of the equalized image
    std::vector<cv::Mat> equalizedImageHistogram(3);
    equalizedImageHistogram = calculateHistogram(equalizedImage);
    showHistogram(equalizedImageHistogram);
    cv::waitKey();

    // STEP 5 - Equalize the original image using luminance equalization
    luminanceEqualizedImage = luminanceEqualization(originalImage);
    cv::imshow("Luminance Equalized Image", luminanceEqualizedImage);

    // Calculate and show histogram of the luminance equalized image
    std::vector<cv::Mat> luminanceEqualizedImageHistogram(3);
    luminanceEqualizedImageHistogram = calculateHistogram(luminanceEqualizedImage);
    showHistogram(luminanceEqualizedImageHistogram);
    cv::waitKey();
    cv::destroyAllWindows();


    /****************** PART 2 - IMAGE FILTERING ******************/
    // Variable for the slider
    int ksize_slider;
    int sigmaX_slider;
    int sigmaRange_slider;
    int sigmaSpace_slider;

    const int MAX_KSIZE_SLIDER = 31;
    const int MAX_SIGMAX_SLIDER = 100;
    const int MAX_SIGMARANGE_SLIDER = 3000;
    const int MAX_SIGMASPACE_SLIDER = 500;

    // STEP 1 - Median Filter
    ksize_slider = 1;
    cv::namedWindow("Median Filter", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("ksize", "Median Filter", &ksize_slider, MAX_KSIZE_SLIDER, medianFilter);
    // Show some image at the beginning
    medianFilter(ksize_slider, 0);
    cv::waitKey();

    // STEP 2 - Gaussian Filter
    ksize_slider = 1;
    sigmaX_slider = 0;
    cv::namedWindow("Gaussian Filter", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("ksize", "Gaussian Filter", &ksize_slider, MAX_KSIZE_SLIDER, gaussianFilter);
    cv::createTrackbar("sigmaX", "Gaussian Filter", &sigmaX_slider, MAX_SIGMAX_SLIDER, gaussianFilter);
    // Show some image at the beginning
    gaussianFilter(ksize_slider, 0);
    cv::waitKey();

    // STEP 3 - Bilateral Filter
    sigmaRange_slider = 0;
    sigmaSpace_slider = 0;
    cv::namedWindow("Bilateral Filter", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("sigma range", "Bilateral Filter", &sigmaRange_slider, MAX_SIGMARANGE_SLIDER, bilateralFilter);
    cv::createTrackbar("sigma space", "Bilateral Filter", &sigmaSpace_slider, MAX_SIGMASPACE_SLIDER, bilateralFilter);
    // Show some image at the beginning
    bilateralFilter(1,0);
    cv::waitKey();

    cv::destroyAllWindows();

	return 0;
}
/**
 * This function will perform a bilateral filter smoothing based on the values of sigma range
 * and sigma space that is set on the trackbar
 */
static void bilateralFilter(int, void*)
{
    cv::Mat filteredImage = luminanceEqualizedImage.clone();

    double sigmaRange = (double) cv::getTrackbarPos("sigma range", "Bilateral Filter");
    double sigmaSpace = (double) cv::getTrackbarPos("sigma space", "Bilateral Filter");
    sigmaRange = sigmaRange / 1;
    sigmaSpace = sigmaSpace / 1;

    cv::bilateralFilter(luminanceEqualizedImage, filteredImage, 5, sigmaSpace, sigmaRange, cv::BORDER_DEFAULT);
    cv::imshow("Bilateral Filter", filteredImage);

    std::cout << "sigma range: " << sigmaRange;
    std::cout << " sigma space: " << sigmaSpace << std::endl;
}

/**
 * This function will perform a gaussian blur based on the value of the kernel size and sigmaX
 * that is selected on the trackbar
 */
static void gaussianFilter(int, void*)
{
    cv::Mat filteredImage = luminanceEqualizedImage.clone();

    int ksize = cv::getTrackbarPos("ksize", "Gaussian Filter");
    double sigmaX = (double) cv::getTrackbarPos("sigmaX", "Gaussian Filter");
    sigmaX = sigmaX / 10;

    // ksize of gaussian filter must be odd and greater than or equal to zero
    if (ksize >= 1) {
        ksize = (ksize * 2) - 1;
    }

    // sigmaX must be greater or equal to zero
    if (ksize > 0 && sigmaX >= 0)
    {
        cv::GaussianBlur(luminanceEqualizedImage, filteredImage, cv::Size(ksize, ksize), sigmaX, 0, 0);
        cv::imshow("Gaussian Filter", filteredImage);

        std::cout << "ksize: " << ksize;
        std::cout << " sigmaX: " << sigmaX << std::endl;
    }
}

/**
 * This function will perform median blur based on the value of the kernel size that
 * is selected on the trackbar.
 */
static void medianFilter(int, void*)
{
    cv::Mat filteredImage = luminanceEqualizedImage.clone();
    int ksize = cv::getTrackbarPos("ksize", "Median Filter");
    
    // ksize of median filter must be odd and greater than or equal to one
    if ((ksize % 2) != 0 && ksize >= 1)
    {
        cv::medianBlur(luminanceEqualizedImage, filteredImage, ksize);
        cv::imshow("Median Filter", filteredImage);

        std::cout << "ksize: " << ksize << std::endl;
    }
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
cv::Mat rgbEqualization(cv::Mat& img)
{
    // Separating the image into 3 channels (B, G, and R)
    std::vector<cv::Mat> bgr_channels;
    cv::split(img, bgr_channels);

    // Applying histogram equalization to the image
    cv::Mat b_equalized;
    cv::Mat g_equalized;
    cv::Mat r_equalized;
    cv::equalizeHist(bgr_channels[0], b_equalized);
    cv::equalizeHist(bgr_channels[1], g_equalized);
    cv::equalizeHist(bgr_channels[2], r_equalized);

    // Creating a 3 channels vector for the BGR equalized images
    std::vector<cv::Mat> bgr_equalized(3);
    bgr_equalized[0] = b_equalized;
    bgr_equalized[1] = g_equalized;
    bgr_equalized[2] = r_equalized;

    // Merging the BGR equalized images into a single image
    cv::Mat equalizedImage;
    cv::merge(bgr_equalized, equalizedImage);

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
    cv::Mat L_equalized;
    cv::equalizeHist(Lab_channels[0], L_equalized);

    // Creating a 3 channels vector for the Lab equalized images
    std::vector<cv::Mat> Lab_equalized(3);
    Lab_equalized[0] = L_equalized;
    Lab_equalized[1] = Lab_channels[1];
    Lab_equalized[2] = Lab_channels[2];

    // Merging the Lab equalized images into a single image
    cv::Mat luminanceEqualizedImage;
    cv::merge(Lab_equalized, luminanceEqualizedImage);

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
