# ScreenSight — Real-Time Screen Object & Face Detection

A C++ application that continuously captures your screen and runs dual deep learning inference in real time — detecting faces with YuNet and objects/vehicles with YOLOv8.

## Features

- **Real-time screen capture** using Windows GDI
- **Face detection** via YuNet (ONNX) with confidence overlay and 150×150 thumbnail previews
- **Object detection** via YOLOv8 nano (COCO 80-class) with labeled bounding boxes
- Simultaneous dual-model inference at ~1000 FPS polling
- Annotated live window showing both detection layers

## Demo

| Detection Type | Color | Info Shown |
|---|---|---|
| Faces | Green | Confidence % + thumbnail in corner |
| Objects / Vehicles | Blue | Class name + confidence % |

## Requirements

- Windows 10/11 (x64)
- Visual Studio 2022 (MSVC)
- CMake 3.15+
- OpenCV 4.12.0 (via vcpkg)

## Build

```bash
# Configure
cmake -B build -G "Visual Studio 17 2022" -A x64

# Build
cmake --build build --config Release
```

## Model Files

Place the following model files before running:

```
face_detector/
  face_detection_yunet_2023mar.onnx    # YuNet face detector
YOLOv8/
  yolov8n.onnx                         # YOLOv8 nano (COCO)
```

The application will exit on startup if either model file is missing.

## Run

```bash
build\Release\FlightSimulator.exe
```

Press **`q`** to quit.

## How It Works

### Screen Capture
Uses `BitBlt` + `GetDIBits` via Windows GDI to capture the full desktop into an OpenCV `Mat` on each frame.

### Face Detection (YuNet)
- Input scaled to 0.5× for speed
- Confidence threshold: `0.5` | NMS threshold: `0.3` | Top-K: `100`
- Detected faces are cropped to 150×150 thumbnails and displayed in the top-left corner

### Object Detection (YOLOv8)
- Input resized to `640×640`
- Output tensor `[1, 84, 8400]` reshaped and post-processed per class
- Confidence threshold: `0.5` | NMS threshold: `0.4`
- Draws class name + confidence on each bounding box

## Project Structure

```
FlightSimulator/
├── main.cpp                                     # All application logic
├── CMakeLists.txt                               # CMake build config
├── face_detector/
│   └── face_detection_yunet_2023mar.onnx        # Face model
├── YOLOv8/
│   ├── yolov8n.onnx                             # Object detection model
│   └── best.onnx                                # Custom-trained variant
└── build/                                       # CMake build output
```

## Dependencies

| Library | Purpose |
|---|---|
| OpenCV 4.12.0 | Image processing, DNN inference, display |
| Windows GDI | Screen capture (`windows.h`) |
| vcpkg | Package management |

## License

This project does not currently include a license file.
