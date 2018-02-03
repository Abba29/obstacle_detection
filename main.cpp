#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

//! [DBSCAN CONSTANTS DEFINITION]
const int NOISE = -2;
const int NOT_CLASSIFIED = -1;
//! [DBSCAN CONSTANTS DEFINITION]

//! [DBSCAN CLASSES DEFINITION]
class PointDB {
public:
    int frameNumber;
    size_t objectClass;
    float confidence;
    int x, y;
    int pointsCount; // # of adjacent points
    int cluster; // # of cluster

    // CONSTRUCTORS
    PointDB(size_t objClass, float conf, int frameN, int x,int y) {
        this->frameNumber = frameN;
        this->objectClass = objClass;
        this->confidence = conf;
        this->x = x;
        this->y = y;
        this->pointsCount = 0;
        this->cluster = NOT_CLASSIFIED;
    }

    double getDistance(const PointDB& otherPoint) {
        return sqrt((x-otherPoint.x)*(x-otherPoint.x)+(y-otherPoint.y)*(y-otherPoint.y));
    }
};

class DBSCAN {
public:
    int minPts; // Min # of neighbors to define a core point
    double eps; // Range of neighborhood
    vector<PointDB> points;
    int size; // Size of 'points' vector
    vector<vector<int>> adjPoints; // Contains, for each point, a vector of his adjacent points
    vector<vector<int>> cluster; // Contains, for each cluster #, a vector of his corresponding points
    int clusterIdx; // # of clusters

    // CONSTRUCTOR
    DBSCAN(double eps, int minPts, vector<PointDB> points) {
        this->eps = eps;
        this->minPts = minPts;
        this->points = points;
        this->size = (int)points.size(); // Resizes 'adjacentPoints' to the size of 'points' vector
        adjPoints.resize(size);
        this->clusterIdx = -1;
    }

    // GET METHOD
    vector<vector<int>> getCluster() {
        return cluster;
    }

    // DBSCAN CLUSTERING METHOD
    void run () {
        checkNearPoints();

        for(int i = 0; i < size; i++) {
            if(points[i].cluster != NOT_CLASSIFIED) continue;
            if(isCoreObject(i)) {
                dfs(i, ++clusterIdx);
            }
            else {
                points[i].cluster = NOISE;
            }
        }
        cluster.resize(clusterIdx + 1);
        for(int i = 0; i < size; i++) {
            if(points[i].cluster != NOISE) {
                cluster[points[i].cluster].push_back(i);
            }
        }
    }

    // SUPPORT METHOD. Calculates the number ('points[i].pointsCount') and the
    // indices ('adjacentPoints[i]') of the i-th point's adjacent points.
    void checkNearPoints() {
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                if(i == j) continue;
                if(points[i].getDistance(points[j]) <= eps) {
                    points[i].pointsCount++;
                    adjPoints[i].push_back(j);
                }
            }
        }
    }

    // SUPPORT METHOD. Performs a DFS (Depth First Search) from the now-th core point to identify its neighbors.
    // It recursively applies the DFS for each of the identified neighborhood points, until there are no more points
    // that can be added to the set.
    void dfs (int now, int c) {
        points[now].cluster = c;
        if(!isCoreObject(now)) return;

        for(auto&next:adjPoints[now]) {
            if(points[next].cluster != NOT_CLASSIFIED) continue;
            dfs(next, c);
        }
    }

    // SUPPORT METHOD. Checks if idx-th point is a core object.
    bool isCoreObject(int idx) {
        return points[idx].pointsCount >= minPts;
    }
};
//! [DBSCAN CLASSES DEFINITION]

//! [PARAMETERS]
// YOLO
const float confidenceThreshold = 0.24;
const string namesPath = "/home/lorenzo/Scrivania/yolo-9000/darknet/data/coco.names";
const string source = "/home/lorenzo/Scrivania/test/hd/base.avi";
const string tinyModelBinary = "/home/lorenzo/Scrivania/yolo-9000/yolo9000-weights/tiny-yolo.weights";
const string tinyModelConfiguration = "/home/lorenzo/Scrivania/yolo-9000/darknet/cfg/tiny-yolo.cfg";
const string yoloModelBinary = "/home/lorenzo/Scrivania/yolo-9000/yolo9000-weights/yolo.weights";
const string yoloModelConfiguration = "/home/lorenzo/Scrivania/yolo-9000/darknet/cfg/yolo.cfg";

// DBSCAN CLUSTERING
const double eps = 25; // Range of the neighborhood
const int minPoints = 5; // Minimum # of neighbors to define a core point
const int windowSize = 10; //Size of sliding window

// EXPONENTIAL MOVING AVERAGE
const vector<float> alpha5  = {0.6, 0.2, 0.15, 0.035, 0.015};
const vector<float> alpha6  = {0.6, 0.2, 0.1,  0.06,  0.025, 0.015};
const vector<float> alpha7  = {0.6, 0.2, 0.1,  0.055, 0.025, 0.015, 0.005};
const vector<float> alpha8  = {0.6, 0.2, 0.1,  0.05,  0.02,  0.015, 0.01,  0.005};
const vector<float> alpha9  = {0.6, 0.2, 0.1,  0.045, 0.02,  0.015, 0.01,  0.008, 0.002};
const vector<float> alpha10 = {0.6, 0.2, 0.1,  0.04,  0.02,  0.015, 0.011, 0.008, 0.005, 0.001};

//RISK OF IMPACT
const float proximityThreshold = 0.3;
const float movementThreshold = 0.4;
const Rect proximityZone = Rect(0, 400, 1280, 320);
const Rect movementZone = Rect(400, 0, 450, 720);
//! [PARAMETERS]

int main() {
    char choice;
    String modelConfiguration; // The path to the .cfg file with text description of the network architecture.
    String modelBinary;        // The path to the .weights file with learned network.

    cout << "Supported YOLO versions: " << endl << endl;
    cout << "A) Tiny-YOLO \t B) YOLO" << endl << endl;
    cout << "What version do you want to use? (A/B) ";
    cin >> choice;
    cout << endl;

    if (choice == 'a' || choice == 'A') {
        modelBinary = tinyModelBinary;
        modelConfiguration = tinyModelConfiguration;
    }
    else if (choice == 'b' || choice == 'B') {
        modelBinary = yoloModelBinary;
        modelConfiguration = yoloModelConfiguration;
    }

    //! [Initialize network]
    dnn::Net net = readNetFromDarknet(modelConfiguration, modelBinary); // Read a network model stored in Darknet model files
    //! [Initialize network]

    if (net.empty()) {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "cfg-file:     " << modelConfiguration << endl;
        cerr << "weights-file: " << modelBinary << endl;
        exit(-1);
    }

    //! [Copy the content of coco.names into a vector]
    vector<string> classNamesVec;

    ifstream classNamesFile(namesPath);
    if (classNamesFile.is_open()) {
        string className = "";
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }
    else {
        cout << "Couldn't open the following file: " << namesPath << endl;
        return -1;
    }
    //! [Copy the content of coco.names into a vector]4

    VideoCapture cap(source);

    if (!cap.isOpened()) {
        cout << "Couldn't open image or video: " << source << endl;
        return -1;
    }

    int frameWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    // int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form

    //Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
    //              (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));

    //VideoWriter outputVideo("/home/lorenzo/Scrivania/output.avi", ex, cap.get(CV_CAP_PROP_FPS), S, true);

    Mat frame;

    int frameNum;
    int windowStart = 1;

    vector<PointDB> leftBottomPoints, rightTopPoints;
    vector<vector<int>> leftBottomCluster, rightTopCluster;

    int index = -1; // Index for the corresponding cluster
    vector<int>::iterator it; // Iterator used for searching the corresponding cluster

    vector<float> alpha;
    vector<PointDB> tempLBPoints, tempRTPoints;
    float xLB = 0, yLB = 0, xRT = 0, yRT = 0; // Variables for bBoxes' points mean values

    Rect movemIntersection, proximIntersection;
    float objArea, movemIntersecArea, proximIntersecArea;

    vector<double> clusteringTimings, otherTimings, readingTimings, yoloTimings; // Vectors for timings

    // ofstream myfile;
    // myfile.open("/home/lorenzo/Scrivania/timings.txt");
    // myfile << std::fixed << std::setprecision(8);
    // myfile << "frame \t frameReading \t yoloDetection \t clustering \t other" << endl;

    for (;;) {

        auto startReading = std::chrono::high_resolution_clock::now();

        cap >> frame; // Get a new frame from video or read image

        auto finishReading = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> timeReading = finishReading - startReading;
        readingTimings.push_back(timeReading.count());

        if (frame.empty()) break;

        frameNum = cap.get(CV_CAP_PROP_POS_FRAMES);

        ostringstream ss; // Output stream to operate on strings

        auto startYOLO = std::chrono::high_resolution_clock::now();

        if (frame.channels() == 4)
            cvtColor(frame, frame, COLOR_BGRA2BGR);

        //! [Prepare blob]
        Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false); // Convert Mat to batch of images
        //! [Prepare blob]

        //! [Set input blob]
        net.setInput(inputBlob, "data"); // Set the network input
        //! [Set input blob]

        //! [Make forward pass]
        Mat detectionMat = net.forward("detection_out"); // Compute output
        //! [Make forward pass]

        for (int i = 0; i < detectionMat.rows; i++) {
            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

            size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

            if (confidence > confidenceThreshold) {
                float x = detectionMat.at<float>(i, 0);
                float y = detectionMat.at<float>(i, 1);
                float width = detectionMat.at<float>(i, 2);
                float height = detectionMat.at<float>(i, 3);
                int xLeftBottom = static_cast<int>((x - width / 2) * frame.cols);
                int yLeftBottom = static_cast<int>((y - height / 2) * frame.rows);
                int xRightTop = static_cast<int>((x + width / 2) * frame.cols);
                int yRightTop = static_cast<int>((y + height / 2) * frame.rows);

                PointDB tempPoint = PointDB(objectClass, confidence, frameNum, xLeftBottom, yLeftBottom);
                leftBottomPoints.push_back(tempPoint);
                tempPoint = PointDB(objectClass, confidence, frameNum, xRightTop, yRightTop);
                rightTopPoints.push_back(tempPoint);
            }
        }

        auto finishYOLO = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> timeYOLO = finishYOLO - startYOLO;
        yoloTimings.push_back(timeYOLO.count());

        rectangle(frame, proximityZone, Scalar(0,0,200), 2);
        rectangle(frame, movementZone,  Scalar(0,0,200), 2);

        if(frameNum < 10) { // Before frame #10 print out bBoxes as they are
            for(int i = 0; i < leftBottomPoints.size(); i++) {
                if(leftBottomPoints[i].frameNumber == frameNum) {
                    Rect object(leftBottomPoints[i].x, leftBottomPoints[i].y,
                                rightTopPoints[i].x - leftBottomPoints[i].x,
                                rightTopPoints[i].y - leftBottomPoints[i].y);

                    rectangle(frame, object, Scalar(0, 255, 0));
                    circle(frame, Point(object.x + (object.width / 2), object.y + (object.height / 2)), 3, Scalar(0, 255, 0), -1);

                    PointDB tempLB = leftBottomPoints[i];

                    // Print out the detected object's classification and confidence
                    if (tempLB.objectClass < classNamesVec.size()) {
                        ss.str("");
                        ss << tempLB.confidence;
                        String conf(ss.str());
                        String label = String(classNamesVec[tempLB.objectClass] + ": " + conf);
                        int baseLine = 0;
                        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                        rectangle(frame, Rect(Point(tempLB.x, tempLB.y - (labelSize.height + baseLine)), Size(labelSize.width, labelSize.height + baseLine)), Scalar(255, 255, 255), CV_FILLED);
                        putText(frame, label, Point(tempLB.x, tempLB.y - baseLine), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
                    }

                    objArea = object.area();

                    proximIntersection = object & proximityZone;
                    proximIntersecArea = proximIntersection.area();

                    if((proximIntersecArea/objArea) >= proximityThreshold) {
                        movemIntersection = object & movementZone;
                        movemIntersecArea = movemIntersection.area();
                        if((movemIntersecArea/objArea) >= movementThreshold)
                            cout << "Imminent danger!" << endl;
                    }
                }
            }

            clusteringTimings.push_back(0);
            otherTimings.push_back(0);
        }
        else { // From frame #10 calculate the mean values

            auto startClustering = std::chrono::high_resolution_clock::now();

            DBSCAN leftBottomDBSCAN = DBSCAN(eps, minPoints, leftBottomPoints);
            DBSCAN rightTopDBSCAN = DBSCAN(eps, minPoints, rightTopPoints);
            leftBottomDBSCAN.run();
            rightTopDBSCAN.run();

            auto finishClustering = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> timeClustering = finishClustering - startClustering;
            clusteringTimings.push_back(timeClustering.count());

            leftBottomCluster = leftBottomDBSCAN.getCluster();
            rightTopCluster = rightTopDBSCAN.getCluster();

            auto startOther = std::chrono::high_resolution_clock::now();

            // Use the smaller between 'lefBottomCluster' and 'rightTopCluster'. Use 'leftBottomCluster' in case of same dimensions
            if(leftBottomCluster.size() <= rightTopCluster.size()) {
                for(int i = 0; i < leftBottomCluster.size(); i++) { // Scan the leftBottomCluster's clusters
                    for(int j = 0; j < leftBottomCluster[i].size(); j++) { // Scan the elements of the cluster
                        for(int k = 0; k < rightTopCluster.size(); k++) { // Scan the rightTopCluster's cluster

                            // Search for the leftBottomCluster[i][j] element to find the rightTopCluster's corresponding cluster.
                            // NB. If no element is found, the 'find' function returns the iterator to the last element of the vector
                            it = find(rightTopCluster[k].begin(), rightTopCluster[k].end(), leftBottomCluster[i][j]);
                            if(it == rightTopCluster[k].end())
                                continue;
                            else {
                                index = k; // Assign the index of corresponding cluster
                                break;
                            }
                        }
                        if(index > -1) // Corresponding cluster found. No more search
                            break;
                    }

                    int minSize = std::min(leftBottomCluster[i].size(), rightTopCluster[index].size());

                    if(minSize > 4 && index > -1) {

                        if(minSize == 5)
                            alpha = alpha5;
                        else if(minSize == 6)
                            alpha = alpha6;
                        else if(minSize == 7)
                            alpha = alpha7;
                        else if(minSize == 8)
                            alpha = alpha8;
                        else if(minSize == 9)
                            alpha = alpha9;
                        else if(minSize == 10)
                            alpha = alpha10;

                        for(int m = 1; m <= minSize; m++) {
                            tempLBPoints.push_back(leftBottomPoints[leftBottomCluster[i][leftBottomCluster[i].size() - m]]);
                            tempRTPoints.push_back(rightTopPoints[rightTopCluster[index][rightTopCluster[index].size() - m]]);
                        }

                        // Mean values
                        for(int n = 0; n < alpha.size(); n++) {
                            xLB += alpha[n] * tempLBPoints[n].x;
                            yLB += alpha[n] * tempLBPoints[n].y;
                            xRT += alpha[n] * tempRTPoints[n].x;
                            yRT += alpha[n] * tempRTPoints[n].y;
                        }

                        // Round mean values to the nearest integer value
                        xLB = std::roundf(xLB);
                        yLB = std::roundf(yLB);
                        xRT = std::roundf(xRT);
                        yRT = std::roundf(yRT);

                        // Check if mean values belong to the frame dimensions
                        if((xLB >= 0 && xLB <= frameWidth) && (yLB >= 0 && yLB <= frameHeight) &&
                                (xRT >= 0 && xRT <= frameWidth) && (yRT >= 0 && yRT <= frameHeight)) {
                            Rect object(xLB, yLB,
                                        xRT - xLB,
                                        yRT - yLB);

                            rectangle(frame, object, Scalar(0, 255, 0));
                            circle(frame, Point(object.x + (object.width / 2), object.y + (object.height / 2)), 3, Scalar(0, 255, 0), -1);

                            PointDB temp = leftBottomPoints[leftBottomCluster[i][leftBottomCluster[i].size() - 1]];

                            // Print out the detected object's classification and confidence
                            if (temp.objectClass < classNamesVec.size()) {
                                ss.str("");
                                ss << temp.confidence;
                                String conf(ss.str());
                                String label = String(classNamesVec[temp.objectClass] + ": " + conf);
                                int baseLine = 0;
                                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                                rectangle(frame, Rect(Point(xLB, yLB - (labelSize.height + baseLine)), Size(labelSize.width, labelSize.height + baseLine)), Scalar(255, 255, 255), CV_FILLED);
                                putText(frame, label, Point(xLB, yLB - baseLine), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
                            }

                            objArea = object.area();

                            proximIntersection = object & proximityZone;
                            proximIntersecArea = proximIntersection.area();

                            if((proximIntersecArea/objArea) >= proximityThreshold) {
                                movemIntersection = object & movementZone;
                                movemIntersecArea = movemIntersection.area();
                                if((movemIntersecArea/objArea) >= movementThreshold)
                                    cout << "Imminent danger!" << endl;
                            }
                        }
                    }

                    index = -1;

                    alpha.clear();
                    tempLBPoints.clear();
                    tempRTPoints.clear();

                    xLB = 0, yLB = 0, xRT = 0, yRT = 0;
                }
            }
            else {
                for(int i = 0; i < rightTopCluster.size(); i++) { // Scan the rightTopCluster's clusters
                    for(int j = 0; j < rightTopCluster[i].size(); j++) { // Scan the elements of the cluster
                        for(int k = 0; k < leftBottomCluster.size(); k++) { // Scan the leftBottomCluster's clusters

                            // Search for the rightTopCluster[i][j] element to find the leftBottomCluster's corresponding cluster.
                            // NB. If no element is found, the 'find' function returns the iterator to the last element of the vector
                            it = find(leftBottomCluster[k].begin(), leftBottomCluster[k].end(), rightTopCluster[i][j]);
                            if(it == leftBottomCluster[k].end())
                                continue;
                            else {
                                index = k; // Assign the index of the corresponding cluster
                                break;
                            }
                        }
                        if(index > -1) // Corresponding cluster found. No more search
                            break;
                    }

                    int minSize = std::min(leftBottomCluster[index].size(), rightTopCluster[i].size());

                    if(minSize > 4 && index > -1) {

                        if(minSize == 5)
                            alpha = alpha5;
                        else if(minSize == 6)
                            alpha = alpha6;
                        else if(minSize == 7)
                            alpha = alpha7;
                        else if(minSize == 8)
                            alpha = alpha8;
                        else if(minSize == 9)
                            alpha = alpha9;
                        else if(minSize == 10)
                            alpha = alpha10;

                        for(int m = 1; m <= minSize; m++) {
                            tempLBPoints.push_back(leftBottomPoints[leftBottomCluster[index][leftBottomCluster[index].size() - m]]);
                            tempRTPoints.push_back(rightTopPoints[rightTopCluster[i][rightTopCluster[i].size() - m]]);
                        }

                        // Mean values
                        for(int n = 0; n < alpha.size(); n++) {
                            xLB += alpha[n] * tempLBPoints[n].x;
                            yLB += alpha[n] * tempLBPoints[n].y;
                            xRT += alpha[n] * tempRTPoints[n].x;
                            yRT += alpha[n] * tempRTPoints[n].y;
                        }

                        // Round mean values to the nearest integer value
                        xLB = std::roundf(xLB);
                        yLB = std::roundf(yLB);
                        xRT = std::roundf(xRT);
                        yRT = std::roundf(yRT);

                        // Check if mean values belong to the frame dimensions
                        if((xLB >= 0 && xLB <= frameWidth) && (yLB >= 0 && yLB <= frameHeight) &&
                                (xRT >= 0 && xRT <= frameWidth) && (yRT >= 0 && yRT <= frameHeight)) {
                            Rect object(xLB, yLB,
                                        xRT - xLB,
                                        yRT - yLB);

                            rectangle(frame, object, Scalar(0, 255, 0));
                            circle(frame, Point(object.x + (object.width / 2), object.y + (object.height / 2)), 3, Scalar(0, 255, 0), -1);

                            PointDB temp = rightTopPoints[rightTopCluster[i][rightTopCluster[i].size() - 1]];

                            // Print out the detected object's classification and confidence
                            if (temp.objectClass < classNamesVec.size()) {
                                ss.str("");
                                ss << temp.confidence;
                                String conf(ss.str());
                                String label = String(classNamesVec[temp.objectClass] + ": " + conf);
                                int baseLine = 0;
                                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                                rectangle(frame, Rect(Point(xLB, yLB - (labelSize.height + baseLine)), Size(labelSize.width, labelSize.height + baseLine)), Scalar(255, 255, 255), CV_FILLED);
                                putText(frame, label, Point(xLB, yLB - baseLine), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
                            }

                            objArea = object.area();

                            proximIntersection = object & proximityZone;
                            proximIntersecArea = proximIntersection.area();

                            if((proximIntersecArea/objArea) >= proximityThreshold) {
                                movemIntersection = object & movementZone;
                                movemIntersecArea = movemIntersection.area();
                                if((movemIntersecArea/objArea) >= movementThreshold)
                                    cout << "Imminent danger!" << endl;
                            }
                        }
                    }

                    index = -1;

                    alpha.clear();
                    tempLBPoints.clear();
                    tempRTPoints.clear();

                    xLB = 0, yLB = 0, xRT = 0, yRT = 0;
                }
            }

            auto finishOther = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> timeOther = finishOther - startOther;
            otherTimings.push_back(timeOther.count());

            // Simulate the sliding window by erasing the elements with oldest frame reference
            for(int i = 0; i < leftBottomPoints.size(); i++) {
                if(leftBottomPoints[i].frameNumber == windowStart) {
                    leftBottomPoints.erase(leftBottomPoints.begin() + i);
                    rightTopPoints.erase(rightTopPoints.begin() + i);
                    i--;
                }
            }

            windowStart++;
        }

        // myfile << frameNum << "\t" << readingTimings.back() << "\t" << yoloTimings.back() << "\t" << clusteringTimings.back() << "\t" << otherTimings.back() << endl;

        // outputVideo << frame;

        imshow("YOLO: Detections", frame);
        if (waitKey(1) >= 0) break;
    }

    while(clusteringTimings.front() == 0)
        clusteringTimings.erase(clusteringTimings.begin());

    while(otherTimings.front() == 0)
        otherTimings.erase(otherTimings.begin());

    int clusteringTimingsSize = clusteringTimings.size();
    int otherTimingsSize = otherTimings.size();
    int readingTimingsSize = readingTimings.size();
    int yoloTimingsSize = yoloTimings.size();

    // Mean values
    double meanClusteringTime = std::accumulate(clusteringTimings.begin(), clusteringTimings.end(), 0.0) / clusteringTimingsSize;
    double meanOtherTime = std::accumulate(otherTimings.begin(), otherTimings.end(), 0.0) / otherTimingsSize;
    double meanReadingTime = std::accumulate(readingTimings.begin(), readingTimings.end(), 0.0) / readingTimingsSize;
    double meanYoloTime = std::accumulate(yoloTimings.begin(), yoloTimings.end(), 0.0) / yoloTimingsSize;

    double clusteringTimingsSum = 0;
    double otherTimingsSum = 0;
    double readingTimingsSum = 0;
    double yoloTimingsSum = 0;

    for (std::vector<double>::iterator it = clusteringTimings.begin(); it != clusteringTimings.end(); ++it)
    {
        clusteringTimingsSum += pow((*it - meanClusteringTime), 2.0);
    }

    for (std::vector<double>::iterator it = otherTimings.begin(); it != otherTimings.end(); ++it)
    {
        otherTimingsSum += pow((*it - meanOtherTime), 2.0);
    }

    for (std::vector<double>::iterator it = readingTimings.begin(); it != readingTimings.end(); ++it)
    {
        readingTimingsSum += pow((*it - meanReadingTime), 2.0);
    }

    for (std::vector<double>::iterator it = yoloTimings.begin(); it != yoloTimings.end(); ++it)
    {
        yoloTimingsSum += pow((*it - meanYoloTime), 2.0);
    }

    // Standard deviation values
    double clusteringStanDeviation = sqrt(clusteringTimingsSum / clusteringTimingsSize);
    double otherStanDeviation = sqrt(otherTimingsSum / otherTimingsSize);
    double readingStanDeviation = sqrt(readingTimingsSum / readingTimingsSize);
    double yoloStanDeviation = sqrt(yoloTimingsSum / yoloTimingsSize);

    // myfile << endl << endl;
    // myfile << "Reading." << endl << "Mean: " << meanReadingTime << " s \t Standard Deviation: " << readingStanDeviation << " s" << endl << endl;
    // myfile << "Yolo." << endl << "Mean: " << meanYoloTime << " s \t Standard Deviation: " << yoloStanDeviation << " s" << endl << endl;
    // myfile << "Clustering." << endl << "Mean: " << meanClusteringTime << " s \t Standard Deviation: " << clusteringStanDeviation << " s" << endl << endl;
    // myfile << "Other." << endl << "Mean: " << meanOtherTime << " s \t Standard Deviation: " << otherStanDeviation << " s" << endl << endl;

    // myfile.close();

    cout << std::fixed << std::setprecision(8);
    cout << "Reading." << endl << "Mean: " << meanReadingTime << " s \t Standard Deviation: " << readingStanDeviation << " s" << endl << endl;
    cout << "Yolo." << endl << "Mean: " << meanYoloTime << " s \t Standard Deviation: " << yoloStanDeviation << " s" << endl << endl;
    cout << "Clustering." << endl << "Mean: " << meanClusteringTime << " s \t Standard Deviation: " << clusteringStanDeviation << " s" << endl << endl;
    cout << "Other." << endl << "Mean: " << meanOtherTime << " s \t Standard Deviation: " << otherStanDeviation << " s" << endl << endl;

    return 0;
}
