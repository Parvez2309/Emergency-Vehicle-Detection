# Emergency Vehicle Detection System

A real-time AI system that detects ambulances in traffic and automatically controls traffic lights to give emergency vehicles priority passage.

## Features

- **Real-time Detection**: Uses YOLOv8 to identify ambulances, buses, cars, motorcycles, and trucks
- **Smart Traffic Control**: Automatically switches traffic lights to green when ambulances are detected
- **Multi-format Support**: Processes both images (JPG, PNG) and videos (MP4, AVI, MOV)
- **Web Interface**: Easy-to-use Streamlit application with drag-and-drop file upload

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install streamlit opencv-python ultralytics pillow numpy
   ```

2. **Run the Application**
   ```bash
   streamlit run App.py
   ```

3. **Upload Media**
   - Drag and drop images or videos into the web interface
   - Watch real-time ambulance detection with traffic light simulation

## How It Works

- **Green Light**: Activated automatically when ambulance is detected
- **Red Light**: Default state for normal traffic
- **Visual Feedback**: Bounding boxes around detected vehicles with labels
- **Status Updates**: Real-time detection notifications

## Model Performance

- **Classes**: 5 vehicles (Ambulance, Bus, Car, Motorcycle, Truck)
- **Training**: 100 epochs with YOLOv8
- **Accuracy**: 60.77% precision, 48.10% recall, 49.29% mAP50

## Use Case

Designed to improve emergency response times by automatically managing traffic signals, potentially saving lives through faster ambulance passage through intersections.
