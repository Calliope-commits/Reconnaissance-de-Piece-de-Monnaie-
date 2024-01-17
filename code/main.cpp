// Projet_Images.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>



#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>
#include <vector>
//#include "opencv2/xfeatures2d.hpp"
#include "pathHandler.h"


using std::cout;
using std::endl;

using namespace std;
using namespace cv;
RNG rng(12345);
int min_r = 8.125;
int max_r = 38;
double pi = 3.14159265358979323846;
typedef struct {
    float x;
    float y;
    float radius;
} coin_coordinates;

string W_output = WIN_path_output;
string W_input = WIN_path_input;
cv::Mat coins_edge_detection(cv::Mat coins) {

    if (coins.empty()) {
        cout << "Not a valid image file" << endl;
        return cv::Mat{};
    }
    // defining minimal and maximal radius, specified to the coins.jpg

    int coins_height = coins.size().height;
    int coins_width = coins.size().width;

    cv::Mat coins_resized, coins_blurred, coins_edge;
    cv::resize(coins, coins_resized, Size(int(coins_width / 2), int(coins_height / 2)));
    cv::GaussianBlur(coins_resized, coins_blurred, Size(5, 5), cv::BORDER_DEFAULT);
    cv::Canny(coins_blurred, coins_edge, 127, 255);

    //cv::imwrite("output_images/coins_blurred.jpg", coins_blurred);
    //cv::imwrite("output_images/coins_edge.jpg", coins_edge);
   
    //cv::imwrite("output_images\coins_blurred.jpg", coins_blurred);
    //cv::imwrite("output_images\coins_edge.jpg", coins_edge);
    cv::imwrite(W_output+"\\coins_blurred.jpg", coins_blurred);
    cv::imwrite(W_output+"\\coins_edge.jpg", coins_edge);
    return coins_edge;

}

vector<coin_coordinates> coin_center_detect(cv::Mat coins) {
    cv::Mat coins_edge;
    coins_edge = coins_edge_detection(coins);

    int max_height = coins_edge.size().height;
    int max_width = coins_edge.size().width;
    float edge_threshold = 0.35;
    float intensity_threshold = 255 * 0.123;
    int next_circle_step = 1;
    std::vector<coin_coordinates> coin_detection;

    for (int radius = min_r; radius < max_r; radius++) {
        cv::Mat img_circle = cv::Mat::zeros(cv::Size(radius * 2, radius * 2), CV_8UC1);
        cv::circle(img_circle, cv::Point(radius, radius), radius, 255);
        float circumference = 2 * M_PI * radius;

        std::vector<Point> circle_pixels;
        for (int i = 0; i < img_circle.rows; i++)
            for (int j = 0; j < img_circle.cols; j++) {
                if (img_circle.at<uchar>(i, j) == 255)
                    circle_pixels.push_back(Point(i, j));

            }
        for (int start_y = 0; start_y < max_height - 2 * radius; start_y++)
            for (int start_x = 0; start_x < max_width - 2 * radius; start_x++) {
                int count = 0;
                for (auto it = circle_pixels.begin(); it != circle_pixels.end(); ++it) {
                    int image_x = it->x + start_x;
                    int image_y = it->y + start_y;
                    if (coins_edge.at<uchar>(image_y, image_x) >= intensity_threshold)
                        count++;
                }
                if (count > 50) {
                    float percentage = round(count / circumference * 100);
                    int coor_x = start_x + radius;
                    int coor_y = start_y + radius;
                    cout << "candidate " << coor_x << ", " << coor_y << radius << ", " << percentage << endl;

                }

                if ((count / circumference) > edge_threshold) {
                    int coor_x = start_x + radius;
                    int coor_y = start_y + radius;
                    coin_coordinates cc;
                    cc.x = coor_x;
                    cc.y = coor_y;
                    cc.radius = radius;
                    coin_detection.push_back(cc);
                }
            }


    }
    return coin_detection;
}
void circle_coins(cv::Mat coins) {
    vector<coin_coordinates> coins_circled = coin_center_detect(coins);
    cv::Mat coins_copy;
    coins.copyTo(coins_copy);
    for (auto it = coins_circled.begin(); it != coins_circled.end(); ++it) {
        float x_coor = it->x;
        float y_coor = it->y;
        float radius = it->radius;
        cv::circle(coins_copy, cv::Point(x_coor * 2, y_coor * 2), radius * 2, (0, 0, 255));


    }
    //cv::imwrite("output_images/coins_detected1.jpg", coins_copy);
    cv::imwrite(W_output + "\\coins_detected1.jpg", coins_copy);
}


/*
void hough_try(cv::Mat coins) {
    cv::Mat gray, neg_thresh, thresh, img, copy_coins;
    cv::cvtColor(coins, gray, COLOR_BGR2GRAY);
    cv::threshold(gray, thresh, 120, 255, THRESH_BINARY);
    cv::bitwise_not(thresh, neg_thresh);
    cv::medianBlur(neg_thresh, img, 5);
    vector<Vec3f> circles;
    cout << "hereeee" << endl;
    //imwrite("output_images/negthresh1.jpg", neg_thresh);
    imwrite(W_output+"\negthresh1.jpg", neg_thresh);
    cv::HoughCircles(neg_thresh, circles, cv::HOUGH_GRADIENT, 1, 300, 20, 40, 0, 0);
    coins.copyTo(copy_coins);
    cout << circles.size() << endl;
    for (auto it = circles.begin(); it != circles.end(); ++it) {
        float x_coor = it[0][0];
        float y_coor = it[0][1];
        float detected_radius = it[0][2];
        cv::circle(copy_coins, cv::Point(x_coor, y_coor), detected_radius, (0, 0, 255), 1);

    }
    cout << "here" << endl;
    //cv::imwrite("output_images/coins_detected_Hough4.jpg", copy_coins);
    cv::imwrite(W_output+"coins_detected_Hough4.jpg", copy_coins);

    //cv::GaussianBlur(gray, img, Size(17,17), cv::BORDER_DEFAULT);
    //cv::Canny(img, img, 127, 200);
    /*cv::adaptiveThreshold(img, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);
    vector<Vec3f> circles;
    cv::HoughCircles(thresh, circles, cv::HOUGH_GRADIENT, 1,30, 20,40,0,0);

    cv::imwrite("output_images/thresh.jpg", thresh);

    coins.copyTo(copy_coins);

    for (auto it = circles.begin(); it != circles.end(); ++it){
        float x_coor = it[0][0];
        float y_coor = it[0][1];
        float detected_radius = it[0][2];
        cv::circle(copy_coins, cv::Point(x_coor, y_coor), detected_radius, (0, 0, 255), 1);

    }

    cv::imwrite("output_images/coins_detected_Hough1.jpg", copy_coins);*/
//}
/*
void findcontours_try() {
    cv::Mat coins = cv::imread("input_images/img1.jpeg");
    cv::Mat gray, neg_thresh, thresh, img, copy_coins, closing;
    cv::cvtColor(coins, gray, COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, img, Size(15, 15), 0);
    adaptiveThreshold(img, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 1);
    Mat kernel = Mat::ones(3, 3, 1);
    morphologyEx(thresh, closing, MORPH_CLOSE, kernel);

    vector<vector<Point> > contours;
    findContours(closing, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    vector<RotatedRect> minEllipse(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        if (contourArea(contours[i]) > 2000) {
            cout << contourArea(contours[i]) << endl;
            cout << contours[i].size() << endl;
            if (contours[i].size() > 5 && contours[i].size() < 5000)
            {
                minEllipse[i] = fitEllipse(contours[i]);
            }
        }
        //cout << contourArea(contours[i]) << endl;

    }
    for (int i = 0; i < contours.size(); i++) {
        //drawContours(coins, contours, int(i), color);

        ellipse(coins, minEllipse[i], color, 2);
    }
    //imwrite("output_images/lasttry/contours_image1_1.jpg", coins);
    imwrite("output_images\lasttry\contours_image1_1.jpg", coins);

}
*/

/*
void sift_detector_try() {
    //Mat img_1 = imread("input_images/1euro.jpeg", 1);
    //Mat img_2 = imread("input_images/2euro.jpeg", 1);
    Mat img_1 = imread(W_input+"\1euro.jpeg", 1);
    Mat img_2 = imread(W_input+"\2euro.jpeg", 1);
    cv::Ptr<Feature2D> f2d = SIFT::create();
    Mat descriptors_1, descriptors_2;
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    f2d->compute(img_1, keypoints_1, descriptors_1);

    f2d->compute(img_2, keypoints_2, descriptors_2);


    BFMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    vector<cv::DMatch> good_matches;
    vector<Point2f> points1, points2;
    for (int i = 0; i < matches.size(); i++)
    {
        //-- Get the keypoints from the good matches

        points1.push_back(keypoints_1[matches[i].queryIdx].pt);

        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    /* Find Homography */
    /*cout << points1.size() << endl;
    Mat H = findHomography(Mat(points2), Mat(points1), RANSAC);
    cv::namedWindow("Mon image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Mon image", H); 
    //cv::waitKey(0);
}*/

cv::Mat loadImage(const char* name)
{
    cv::Mat image;
    image = cv::imread(name);
    if (image.data == NULL || image.channels() != 3)
    {
        std::cout << name << " could not be read or is not correct." << std::endl;
        exit(1);
    }
    return image;
}
#define THRESHOLD_BLUE  130
#define THRESHOLD_TYPE_BLUE  cv::THRESH_BINARY_INV
#define THRESHOLD_GREEN 130
#define THRESHOLD_TYPE_GREEN cv::THRESH_BINARY_INV
#define THRESHOLD_RED   140
#define THRESHOLD_TYPE_RED   cv::THRESH_BINARY_INV
#define CLOSE_ITERATIONS 5
cv::Mat createMask(const cv::Mat& image)
{
    cv::Mat channels[3];
    cv::split(image, channels);
    cv::Mat mask[3];
    cv::threshold(channels[0], mask[0], THRESHOLD_BLUE, 255, THRESHOLD_TYPE_BLUE);
    cv::threshold(channels[1], mask[1], THRESHOLD_GREEN, 255, THRESHOLD_TYPE_GREEN);
    cv::threshold(channels[2], mask[2], THRESHOLD_RED, 255, THRESHOLD_TYPE_RED);
    cv::Mat compositeMask;
    cv::bitwise_and(mask[0], mask[1], compositeMask);
    cv::bitwise_and(compositeMask, mask[2], compositeMask);
    cv::morphologyEx(compositeMask, compositeMask, cv::MORPH_CLOSE,
        cv::Mat(), cv::Point(-1, -1), CLOSE_ITERATIONS);

    /// Next three lines only for debugging, may be removed
    cv::Mat filtered;
    image.copyTo(filtered, compositeMask);
    //cv::imwrite("output_images/10cent_mask.jpg", filtered);
    cv::imwrite(W_output+"\10cent_mask.jpg", filtered);

    return compositeMask;
}
#define TEMPLATE_SIZE 128
cv::Mat locate(const cv::Mat& mask)
{
    // Compute center and radius.
    cv::Moments moments = cv::moments(mask, true);
    Point p(moments.m10 / moments.m00, moments.m01 / moments.m00);

    float area = moments.m00;
    float radius = sqrt(area / pi);
    float xCentroid = moments.m10 / moments.m00;
    float yCentroid = moments.m01 / moments.m00;
    vector<vector<float>> m{ { xCentroid, yCentroid, radius} };
    Mat mat(1, 3, CV_32F);
    for (int i = 0; i < m.size(); ++i)
        mat.row(i) = Mat(m[i]).t();

    return mat;


}
void centerAndScale(const cv::Mat& image, const cv::Mat& mask,
    const cv::Mat& characteristics,
    cv::Mat& imageCS, cv::Mat& maskCS)
{
    float radius = characteristics.at<float>(0, 2);
    float xCenter = characteristics.at<float>(0, 0);
    float yCenter = characteristics.at<float>(0, 1);
    int diameter = round(radius * 2);
    int xOrg = round(xCenter - radius);
    int yOrg = round(yCenter - radius);

    if (xOrg > 0 && yOrg > 0 && xOrg + diameter < image.size().width && yOrg + diameter < image.size().height) {
        cv::Rect roiOrg = cv::Rect(xOrg, yOrg, diameter, diameter);
        cv::Mat roiImg = image(roiOrg);

        cv::Mat roiMask = mask(roiOrg);
        cv::Mat centered = cv::Mat::zeros(diameter, diameter, CV_8UC3);
        roiImg.copyTo(centered, roiMask);
        cv::imwrite("centered.bmp", centered); // debug
        imageCS.create(TEMPLATE_SIZE, TEMPLATE_SIZE, CV_8UC3);
        cv::resize(centered, imageCS, cv::Size(TEMPLATE_SIZE, TEMPLATE_SIZE), 0, 0);
        cv::imwrite("scaled.bmp", imageCS); // debug

        roiMask.copyTo(centered);
        cv::resize(centered, maskCS, cv::Size(TEMPLATE_SIZE, TEMPLATE_SIZE), 0, 0);
    }
}
void saveRotatedTemplates(const cv::Mat& image, const cv::Mat& mask, int stepAngle)
{
    char name[1000];
    cv::Mat rotated(TEMPLATE_SIZE, TEMPLATE_SIZE, CV_8UC3);
    for (int angle = 0; angle < 360; angle += stepAngle)
    {
        cv::Point2f center(TEMPLATE_SIZE / 2, TEMPLATE_SIZE / 2);
        cv::Mat r = cv::getRotationMatrix2D(center, angle, 1.0);

        cv::warpAffine(image, rotated, r, cv::Size(TEMPLATE_SIZE, TEMPLATE_SIZE));
        //sprintf_s(name, "output_images/template-%03d.bmp", angle);
        //String path = W_output + "\template - % 03d.bmp";
        //sprintf_s(name,path , angle);
        sprintf_s(name, "F:\\M1 VMI\\s2\\traitement image\\Projet2\\Image\\Projet_Images\\ouput_images\\template-%03d.bmp", angle);
        cv::imwrite(name, rotated);
       
    
        cv::warpAffine(mask, rotated, r, cv::Size(TEMPLATE_SIZE, TEMPLATE_SIZE));
        //sprintf_s(name, "output_images/templateMask-%03d.bmp", angle);
        sprintf_s(name, "F:\\M1 VMI\\s2\\traitement image\\Projet2\\Image\\Projet_Images\\ouput_images\\templateMask-%03d.bmp", angle);
        cv::imwrite(name, rotated);
    }
}



// Celui là pour les candidats
#define LABEL "10c"
#define MATCH_THRESHOLD 0.065
#define TEMPLATE_SIZE 128
#define THRESHOLD_BLUE  130
#define THRESHOLD_TYPE_BLUE  cv::THRESH_BINARY_INV
#define THRESHOLD_GREEN 130
#define THRESHOLD_TYPE_GREEN cv::THRESH_BINARY_INV
#define THRESHOLD_RED   140
#define THRESHOLD_TYPE_RED   cv::THRESH_BINARY
#define CLOSE_ITERATIONS 5
#define ANGLE_STEP 30
#define CANDIDATES_MIN_AREA 2000
// Possible methods are : CV_TM_CCORR_NORMED, CV_TM_CCORR,
// CV_TM_SQDIFF_NORMED, CV_TM_SQDIFF, CV_TM_CCOEFF_NORMED and CV_TM_CCOEFF
// See http://docs.opencv.org/modules/imgproc/doc/object_detection.html?highlight=matchtemplate#matchtemplate
#define MATCH_METHOD TM_SQDIFF_NORMED
void loadTemplates(vector<cv::Mat>& templates, int angleStep)
{
    templates.clear();
    for (int angle = 0; angle < 360; angle += angleStep)
    {
        char name[1000];
        //sprintf_s(name, "output_images/template-%03d.bmp", angle);
        sprintf_s(name, "F:\\M1 VMI\\s2\\traitement image\\Projet2\\Image\\Projet_Images\\ouput_images\\template-%03d.bmp", angle);
        cv::Mat templateImg = cv::imread(name);
        if (templateImg.data == NULL)
        {
            std::cout << "Could not read " << name << std::endl;
            exit(1);
        }
        templates.push_back(templateImg);
    }
}
typedef struct Candidate
{
    cv::Mat image;
    float x;
    float y;
    float radius;
    float score;
} Candidate;

void getCandidates(const cv::Mat& image, const cv::Mat& mask,
    vector<Candidate>& candidates)
{
    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy;
    /// Find contours
    cv::Mat maskCopy;
    mask.copyTo(maskCopy);
    cv::findContours(maskCopy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    cv::Mat maskCS;
    cv::Mat imageCS;
    cv::Scalar white = cv::Scalar(255);
    for (int nContour = 0; nContour < contours.size(); ++nContour)
    {
        /// Draw contour
        cv::Mat drawing = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::drawContours(drawing, contours, nContour, white, -1, 8, hierarchy, 0, cv::Point());

        // Compute center and radius and area.
        // Discard small areas.
        cv::Moments moments = cv::moments(drawing, true);
        float area = moments.m00;
        if (area < CANDIDATES_MIN_AREA)
            continue;
        Candidate candidate;
        candidate.radius = sqrt(area / pi);
        candidate.x = moments.m10 / moments.m00;
        candidate.y = moments.m01 / moments.m00;

        vector<vector<float>> m{
            { candidate.x, candidate.y, candidate.radius}
        };
        cv::Mat characteristics(1, 3, CV_32F);
        for (int i = 0; i < m.size(); ++i)
            characteristics.row(i) = Mat(m[i]).t();
        centerAndScale(image, drawing, characteristics, imageCS, maskCS);
        imageCS.copyTo(candidate.image);
        if (!candidate.image.empty())
            candidates.push_back(candidate);
    }
}

void saveCandidates(const vector<Candidate>& candidates)
{

    for (int n = 0; n < candidates.size(); ++n)
    {
        char name[1000];
        //sprintf_s(name, "output_images/Candidate-%03d.bmp", n);

        sprintf_s(name, "F:\\M1 VMI\\s2\\traitement image\\Projet2\\Image\\Projet_Images\\ouput_images\\Candidate-%03d.bmp", n);
        cout << candidates[n].image.empty() << endl;;
        cv::imwrite(name, candidates[n].image);
    }
}
float singleTemplateMatch(const cv::Mat& templateImg, const cv::Mat& candidateImg)
{
    cv::Mat result(1, 1, CV_8UC1);
    cv::matchTemplate(candidateImg, templateImg, result, MATCH_METHOD);
    return result.at<float>(0, 0);
}
void matchCandidate(const vector<cv::Mat>& templates, Candidate& candidate)
{
    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    candidate.score;
    if (MATCH_METHOD == TM_SQDIFF || MATCH_METHOD == TM_SQDIFF_NORMED)
        candidate.score = FLT_MAX;
    else
        candidate.score = 0;
    for (auto it = templates.begin(); it != templates.end(); ++it)
    {
        float score = singleTemplateMatch(*it, candidate.image);
        if (MATCH_METHOD == TM_SQDIFF || MATCH_METHOD == TM_SQDIFF_NORMED)
        {
            if (score < candidate.score)
                candidate.score = score;
        }
        else
        {
            if (score > candidate.score)
                candidate.score = score;
        }
    }
}
void matchCandidates(const vector<cv::Mat>& templates,
    vector<Candidate>& candidates)
{
    for (auto it = candidates.begin(); it != candidates.end(); ++it)
        matchCandidate(templates, *it);
}
bool selected(const Candidate& candidate, float threshold)
{
    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better

    if (MATCH_METHOD == TM_SQDIFF || MATCH_METHOD == TM_SQDIFF_NORMED) {
        cout << candidate.score << ",   " << threshold << endl;
        return candidate.score <= threshold;
    }
    else
        return candidate.score > threshold;
}

void drawLabel(const Candidate& candidate, const char* label, cv::Mat image)
{
    int x = candidate.x - candidate.radius;
    int y = candidate.y;
    cv::Point point(x, y);
    cv::Scalar blue(255, 128, 128);
    cv::putText(image, label, point, FONT_HERSHEY_SIMPLEX, 1.5f, blue, 2);
}

cv::Mat labelCoins(const cv::Mat& image, const vector<Candidate>& candidates,
    float threshold, bool inverseThreshold, const char* label)
{
    cv::Mat imageLabeled;
    image.copyTo(imageLabeled);

    for (auto it = candidates.begin(); it != candidates.end(); ++it)
    {
        if (selected(*it, threshold)) {
            cout << "gets here it's selected" << endl;
            drawLabel(*it, label, imageLabeled);
        }
    }

    return imageLabeled;
}


// Ce main c'est pour créer un template
/*#define TEMPLATE_IMG "50c.jpg"
#define ANGLE_STEP 30
int main()
{
    //cv::Mat image = loadImage("input_images/10cent.jpeg");
    //string path = W_input + "\10cent.jpeg";
    cv::Mat image = loadImage("F:\\M1 VMI\\s2\\traitement image\\Projet2\\Image\\Projet_Images\\input_images\\10cent.jpeg");
    cv::Mat mask = createMask(image);

    cv::Mat loc = locate(mask);
    cv::Mat imageCS;
    cv::Mat maskCS;
    centerAndScale(image, mask, loc, imageCS, maskCS);
    saveRotatedTemplates(imageCS, maskCS, ANGLE_STEP);
    return 0;
}*/




//#define INPUT_IMAGE "input_images/image4.jpeg"
//#define LABELED_IMAGE "output_images/coins_with50cLabeled.bmp"
#define INPUT_IMAGE "F:\\M1 VMI\\s2\\traitement image\\Projet2\\Image\\Projet_Images\\input_images\\10cent.jpg"
#define LABELED_IMAGE "F:\\M1 VMI\\s2\\traitement image\\Projet2\\Image\\Projet_Images\\output_images\\coins_with50cLabeled.bmp"
#define LABEL "10c"
#define MATCH_THRESHOLD 0.25
#define ANGLE_STEP 30

int main()
{
    printf("début du lancement \n"); 
    vector<cv::Mat> templates;
    loadTemplates(templates, ANGLE_STEP);
    cv::Mat image = loadImage(INPUT_IMAGE);
    cv::Mat mask = createMask(image);
    vector<Candidate> candidates;
    getCandidates(image, mask, candidates);
    saveCandidates(candidates); // debug
    matchCandidates(templates, candidates);

    cv::Mat labeledImg = labelCoins(image, candidates, MATCH_THRESHOLD, true, LABEL);
    cv::imwrite(LABELED_IMAGE, labeledImg);
    cv::namedWindow("img",WINDOW_NORMAL);
    cv::imshow("img", labeledImg);
    cv::waitKey(0);

    printf("fin du lancement");
    return 0;
}
