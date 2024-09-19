**Sensor Data Fusion for Autonomous Vehicles**

This project focuses on integrating RGB camera and 3D automotive radar data to enhance road user detection and motion prediction. By leveraging advanced object detection models and data fusion techniques, this system aims to improve the accuracy of object detection and tracking in various driving scenarios.

**Project Overview**
**Objectives**
Object Detection: Utilize YOLOv8 and Faster R-CNN models to detect road users from RGB camera data.
Radar Data Processing: Apply advanced radar clustering and data association techniques to enhance object detection and tracking.
Motion Prediction: Integrate spatially correlated data using Kalman filters and temporal association techniques for accurate motion forecasting.

**Key Components**
**Object Detection Models:**

**YOLOv8:** Trained on the Fraunhofer INFRA-3DRC-Dataset for detecting objects in RGB images.

**Faster R-CNN:** Another object detection model utilized for comparative analysis and improved accuracy.
Radar Data Clustering:

**DBSCAN:** Developed a radar clustering algorithm to optimize object detection and tracking from radar data.

**Data Fusion:**

**Spatial Data Association:** Techniques to accurately fuse data from the RGB camera and radar sensors.
**Kalman Filter:** Used to integrate spatially correlated data and motion predictions for precise forecasting.
Temporal Data Association:

**Hungarian Algorithm:** Applied for accurate tracking of objects over time by associating detected objects in consecutive frames.
**Tools and Libraries**
**Programming Language**: Python
**Development Environment:** VS Code, Anaconda
**Libraries:** OpenCV, PyTorch, TensorFlow, Ultralytics
**Algorithms:** YOLO, Faster R-CNN, DBSCAN, Kalman Filter, Hungarian Algorithm
**Platforms**: Google Colab
