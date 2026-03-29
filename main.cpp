#include <iostream>
#include <fstream>
#include <sstream>

#include <windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>


using namespace cv;
using namespace cv::dnn;
using namespace std;


Mat captureScreen() {
    // Get screen dimensions
    int width = GetSystemMetrics(SM_CXSCREEN);
    int height = GetSystemMetrics(SM_CYSCREEN);

    // Set up screen capture
    HDC hScreen = GetDC(NULL);
    HDC hCapture = CreateCompatibleDC(hScreen);
    HBITMAP hBitmap = CreateCompatibleBitmap(hScreen, width, height);
    SelectObject(hCapture, hBitmap);

    // Copy screen to bitmap
    BitBlt(hCapture, 0, 0, width, height, hScreen, 0, 0, SRCCOPY);

    // Convert to OpenCV Mat
    BITMAPINFOHEADER bi;
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height;  // negative = top-down
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;

    Mat frame(height, width, CV_8UC4);
    GetDIBits(hCapture, hBitmap, 0, height, frame.data,
        (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    // Cleanup
    DeleteObject(hBitmap);
    DeleteDC(hCapture);
    ReleaseDC(NULL, hScreen);

    // Convert from BGRA to BGR (OpenCV standard)
    Mat bgr;
    cvtColor(frame, bgr, COLOR_BGRA2BGR);
    return bgr;
}


// Full COCO 80 class names (YOLOv8 default)
// If your custom model only has specific classes, replace this list
vector<string> cocoNames = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

// Colors for difference detection types
Scalar FACE_COLOR(0, 255, 0); // green
Scalar VEHICLE_COLOR(255, 0, 0);
Scalar PLATE_COLOR(0, 255, 255); // yellow

int main(int argc, char** argv) {
    Mat frame = captureScreen();
    const float detScale = 0.5f;  // run detection at half resolution
    Mat small;
    resize(frame, small, Size(), detScale, detScale);

    // Face model
    Ptr<FaceDetectorYN> detector = FaceDetectorYN::create(
        "face_detector/face_detection_yunet_2023mar.onnx",
        "",
        small.size(),
        0.5f,   // confidence threshold
        0.3f,   // nms threshold
        100     // top k
    );

    // YOLOv8 Vehicle model
    Net yoloNet = readNetFromONNX("YOLOv8/yolov8n.onnx");
    if (yoloNet.empty()) {
        cerr << "Failed to load YOLO Model!" << endl;
        return - 1;
    }

    namedWindow("Screen Face Detection", WINDOW_NORMAL);
    resizeWindow("Screen Face Detection", 800, 600);

    while (true) {
        frame = captureScreen();

        resize(frame, small, Size(), detScale, detScale);
        detector->setInputSize(small.size());

        Mat faces;
        detector->detect(small, faces);

        for (int i = 0; i < faces.rows; i++) {
            float* data = faces.ptr<float>(i);
            int x1 = static_cast<int>(data[0] / detScale);
            int y1 = static_cast<int>(data[1] / detScale);
            int x2 = x1 + static_cast<int>(data[2] / detScale);
            int y2 = y1 + static_cast<int>(data[3] / detScale);
            float confidence = data[14];

            // Clamp coordinates to frame bounds
            x1 = max(0, x1);
            y1 = max(0, y1);
            x2 = min(x2, frame.cols);
            y2 = min(y2, frame.rows);

            int faceW = x2 - x1;
            int faceH = y2 - y1;
            if (faceW <= 0 || faceH <= 0) continue;

            //// Draw rectangle on main image
            rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 1);
            string label = "Face: " + to_string(int(confidence * 100)) + "%";
            putText(frame, label, Point(x1, y1 - 10),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
            // crop face
            Mat faceCrop = frame(Rect(x1, y1, faceW, faceH)).clone();

            // Resize the crop to a fixed thumbnail size
            int thumbSize = 150;
            Mat thumbnail;
            resize(faceCrop, thumbnail, Size(thumbSize, thumbSize));

            // Place it in the upper-left corner
            // Offset each face so they don't overlap
            int offsetX = 10 + i * (thumbSize + 10);
            int offsetY = 10;

            // Make sure it fits on screen
            if (offsetX + thumbSize < frame.cols && offsetY + thumbSize < frame.rows) {
                Mat roi = frame(Rect(offsetX, offsetY, thumbSize, thumbSize));
                double alpha = 1; // thumbnail opacity
                addWeighted(thumbnail, alpha, roi, 1.0 - alpha, 0, roi);

                // Draw a border around the thumbnail
                rectangle(frame, Rect(offsetX, offsetY, thumbSize, thumbSize),
                    Scalar(0, 255, 0), 1);

                // Label it
                string label = "Face " + to_string(i + 1);
                putText(frame, label, Point(offsetX, offsetY + thumbSize + 20),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
            }
        }

        // YOLOv8 inference
        const int yoloSize = 640;
        float xScale = (float)frame.cols / yoloSize;
        float yScale = (float)frame.rows / yoloSize;

        Mat yoloBlob = blobFromImage(frame, 1.0 / 255.0, Size(yoloSize, yoloSize), Scalar(), true, false);
        yoloNet.setInput(yoloBlob);
        Mat yoloOut = yoloNet.forward();

        // Reshape [1, 4+classes, 8400] -> transpose to [8400, 4+classes]
        Mat output = yoloOut.reshape(1, yoloOut.size[1]);
        Mat outputT;
        transpose(output, outputT);

        int numClasses = outputT.cols - 4;
        const float yoloConfThresh = 0.5f;
        const float yoloNMSThresh  = 0.4f;

        vector<Rect>  yoloBoxes;
        vector<float> yoloConfs;
        vector<int>   yoloClassIds;

        for (int i = 0; i < outputT.rows; i++) {
            float* row = outputT.ptr<float>(i);

            // Find highest scoring class
            float maxScore = 0;
            int classId = 0;
            for (int c = 0; c < numClasses; c++) {
                if (row[4 + c] > maxScore) {
                    maxScore = row[4 + c];
                    classId = c;
                }
            }
            if (maxScore < yoloConfThresh) continue;

            // Convert cx,cy,w,h -> x1,y1,w,h
            float cx = row[0] * xScale;
            float cy = row[1] * yScale;
            float w  = row[2] * xScale;
            float h  = row[3] * yScale;
            int x1 = static_cast<int>(cx - w / 2);
            int y1 = static_cast<int>(cy - h / 2);

            yoloBoxes.push_back(Rect(x1, y1, (int)w, (int)h));
            yoloConfs.push_back(maxScore);
            yoloClassIds.push_back(classId);
        }

        // Non-maximum suppression
        vector<int> yoloIndices;
        NMSBoxes(yoloBoxes, yoloConfs, yoloConfThresh, yoloNMSThresh, yoloIndices);

        for (int idx : yoloIndices) {
            Rect  box     = yoloBoxes[idx];
            int   classId = yoloClassIds[idx];
            float conf    = yoloConfs[idx];

            string className = (classId < (int)cocoNames.size())
                ? cocoNames[classId] : "obj" + to_string(classId);
            string label = className + ": " + to_string(int(conf * 100)) + "%";

            rectangle(frame, box, VEHICLE_COLOR, 2);
            putText(frame, label, Point(box.x, box.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.6, VEHICLE_COLOR, 2);
        }

        imshow("Screen Face Detection", frame);

        if (waitKey(1) == 'q')
            break;
    }

    destroyAllWindows();
    return 0;
}