#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <atomic>

#include <windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>


using namespace cv;
using namespace cv::dnn;
using namespace std;


class ScreenCapturer {
public:
    ScreenCapturer() {
        width = GetSystemMetrics(SM_CXSCREEN);
        height = GetSystemMetrics(SM_CYSCREEN);

        hScreen = GetDC(NULL);
        hCapture = CreateCompatibleDC(hScreen);
        if (!hScreen || !hCapture)
            throw runtime_error("Failed to create screen DC");

        hBitmap = CreateCompatibleBitmap(hScreen, width, height);
        oldObj = SelectObject(hCapture, hBitmap);

        bi.biSize = sizeof(BITMAPINFOHEADER);
        bi.biWidth = width;
        bi.biHeight = -height; // top-down
        bi.biPlanes = 1;
        bi.biBitCount = 32;
        bi.biCompression = BI_RGB;

        bgra.create(height, width, CV_8UC4);
        bgr.create(height, width, CV_8UC3);
    }

    ~ScreenCapturer() {
        SelectObject(hCapture, oldObj);
        DeleteObject(hBitmap);
        DeleteDC(hCapture);
        ReleaseDC(NULL, hScreen);
    }

    ScreenCapturer(const ScreenCapturer&) = delete;
    ScreenCapturer& operator=(const ScreenCapturer&) = delete;

    const Mat& grab() {
        BitBlt(hCapture, 0, 0, width, height, hScreen, 0, 0, SRCCOPY);
        GetDIBits(hCapture, hBitmap, 0, height, bgra.data,
                  (BITMAPINFO*)&bi, DIB_RGB_COLORS);
        cvtColor(bgra, bgr, COLOR_BGRA2BGR);
        return bgr;
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }

private:
    int width = 0, height = 0;
    HDC hScreen = nullptr;
    HDC hCapture = nullptr;
    HBITMAP hBitmap = nullptr;
    HGDIOBJ oldObj = nullptr;
    BITMAPINFOHEADER bi{};
    Mat bgra, bgr;
};


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

// Colors for detection types
Scalar FACE_COLOR(0, 255, 0); // green
Scalar VEHICLE_COLOR(255, 0, 0);
Scalar PLATE_COLOR(0, 255, 255); // yellow

// Shared state
atomic<bool> running(true);

Mat latestFrame;
mutex frameMutex;

struct FaceResult {
    Rect box;
    float confidence;
};
vector<FaceResult> faceResults;
mutex faceMutex;

struct YoloResult {
    Rect box;
    int classId;
    float confidence;
};
vector<YoloResult> yoloResults;
mutex yoloMutex;

void captureThread(ScreenCapturer& screen) {
    while (running) {
        Mat frame = screen.grab().clone();
        {
            lock_guard<mutex> lock(frameMutex);
            latestFrame = frame;
        }
        this_thread::sleep_for(chrono::milliseconds(1));
    }
}

void faceDetectionThread() {
    const float detScale = 0.5f;

    // Get initial frame size to create detector
    Mat frame;
    while (running) {
        lock_guard<mutex> lock(frameMutex);
        if (!latestFrame.empty()) {
            frame = latestFrame.clone();
            break;
        }
    }
    if (!running) return;

    Mat small;
    resize(frame, small, Size(), detScale, detScale);

    Ptr<FaceDetectorYN> detector = FaceDetectorYN::create(
        "face_detector/face_detection_yunet_2023mar.onnx",
        "",
        small.size(),
        0.5f, 0.3f, 100
    );

    while (running) {
        {
            lock_guard<mutex> lock(frameMutex);
            if (latestFrame.empty()) continue;
            frame = latestFrame.clone();
        }

        resize(frame, small, Size(), detScale, detScale);
        detector->setInputSize(small.size());

        Mat faces;
        detector->detect(small, faces);

        vector<FaceResult> results;
        for (int i = 0; i < faces.rows; i++) {
            float* data = faces.ptr<float>(i);
            int x1 = static_cast<int>(data[0] / detScale);
            int y1 = static_cast<int>(data[1] / detScale);
            int w = static_cast<int>(data[2] / detScale);
            int h = static_cast<int>(data[3] / detScale);

            x1 = max(0, x1);
            y1 = max(0, y1);
            w = min(w, frame.cols - x1);
            h = min(h, frame.rows - y1);
            if (w <= 0 || h <= 0) continue;

            results.push_back({Rect(x1, y1, w, h), data[14]});
        }

        {
            lock_guard<mutex> lock(faceMutex);
            faceResults = move(results);
        }
    }
}

void yoloDetectionThread() {
    Net yoloNet = readNetFromONNX("YOLOv8/yolo11m.onnx");
    if (yoloNet.empty()) {
        cerr << "Failed to load YOLO Model!" << endl;
        running = false;
        return;
    }

    const int yoloSize = 640;
    const float yoloConfThresh = 0.5f;
    const float yoloNMSThresh = 0.4f;

    while (running) {
        Mat frame;
        {
            lock_guard<mutex> lock(frameMutex);
            if (latestFrame.empty()) continue;
            frame = latestFrame.clone();
        }

        float xScale = (float)frame.cols / yoloSize;
        float yScale = (float)frame.rows / yoloSize;

        Mat blob = blobFromImage(frame, 1.0 / 255.0, Size(yoloSize, yoloSize), Scalar(), true, false);
        yoloNet.setInput(blob);
        Mat yoloOut = yoloNet.forward();

        Mat output = yoloOut.reshape(1, yoloOut.size[1]);
        Mat outputT;
        transpose(output, outputT);

        int numClasses = outputT.cols - 4;

        vector<Rect> boxes;
        vector<float> confs;
        vector<int> classIds;

        for (int i = 0; i < outputT.rows; i++) {
            float* row = outputT.ptr<float>(i);

            float maxScore = 0;
            int classId = 0;
            for (int c = 0; c < numClasses; c++) {
                if (row[4 + c] > maxScore) {
                    maxScore = row[4 + c];
                    classId = c;
                }
            }
            if (maxScore < yoloConfThresh) continue;

            float cx = row[0] * xScale;
            float cy = row[1] * yScale;
            float w = row[2] * xScale;
            float h = row[3] * yScale;
            int x1 = static_cast<int>(cx - w / 2);
            int y1 = static_cast<int>(cy - h / 2);

            boxes.push_back(Rect(x1, y1, (int)w, (int)h));
            confs.push_back(maxScore);
            classIds.push_back(classId);
        }

        vector<int> indices;
        NMSBoxes(boxes, confs, yoloConfThresh, yoloNMSThresh, indices);

        vector<YoloResult> results;
        for (int idx : indices) {
            results.push_back({boxes[idx], classIds[idx], confs[idx]});
        }

        {
            lock_guard<mutex> lock(yoloMutex);
            yoloResults = move(results);
        }
    }
}

int main(int argc, char** argv) {
    ScreenCapturer screen;

    thread capThread(captureThread, ref(screen));
    thread faceThread(faceDetectionThread);
    thread yoloThread(yoloDetectionThread);

    namedWindow("ScreenSight", WINDOW_NORMAL);
    resizeWindow("ScreenSight", 800, 600);

    while (running) {
        Mat frame;
        {
            lock_guard<mutex> lock(frameMutex);
            if (latestFrame.empty()) continue;
            frame = latestFrame.clone();
        }

        // Draw face detections
        {
            lock_guard<mutex> lock(faceMutex);
            for (int i = 0; i < (int)faceResults.size(); i++) {
                const auto& f = faceResults[i];
                rectangle(frame, f.box, FACE_COLOR, 1);
                string label = "Face: " + to_string(int(f.confidence * 100)) + "%";
                putText(frame, label, Point(f.box.x, f.box.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.6, FACE_COLOR, 2);

                // Clamp crop to frame bounds
                Rect safeBox = f.box & Rect(0, 0, frame.cols, frame.rows);
                if (safeBox.width <= 0 || safeBox.height <= 0) continue;

                Mat faceCrop = frame(safeBox).clone();
                int thumbSize = 150;
                Mat thumbnail;
                resize(faceCrop, thumbnail, Size(thumbSize, thumbSize));

                int offsetX = 10 + i * (thumbSize + 10);
                int offsetY = 10;

                if (offsetX + thumbSize < frame.cols && offsetY + thumbSize < frame.rows) {
                    Mat roi = frame(Rect(offsetX, offsetY, thumbSize, thumbSize));
                    thumbnail.copyTo(roi);
                    rectangle(frame, Rect(offsetX, offsetY, thumbSize, thumbSize), FACE_COLOR, 1);
                    string thumbLabel = "Face " + to_string(i + 1);
                    putText(frame, thumbLabel, Point(offsetX, offsetY + thumbSize + 20),
                        FONT_HERSHEY_SIMPLEX, 0.6, FACE_COLOR, 2);
                }
            }
        }

        // Draw YOLO detections
        {
            lock_guard<mutex> lock(yoloMutex);
            for (const auto& r : yoloResults) {
                string className = (r.classId < (int)cocoNames.size())
                    ? cocoNames[r.classId] : "obj" + to_string(r.classId);
                string label = className + ": " + to_string(int(r.confidence * 100)) + "%";

                rectangle(frame, r.box, VEHICLE_COLOR, 2);
                putText(frame, label, Point(r.box.x, r.box.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.6, VEHICLE_COLOR, 2);
            }
        }

        imshow("ScreenSight", frame);

        if (waitKey(1) == 'q')
            running = false;
    }

    capThread.join();
    faceThread.join();
    yoloThread.join();

    destroyAllWindows();
    return 0;
}