<<<<<<< HEAD
# Visual Quality Inspection System

An AI-powered visual quality inspection system that uses computer vision and machine learning to detect defects in manufacturing and production processes. This system provides real-time inspection capabilities with both live camera feed and batch image processing modes.

## Features

### Agentic AI Capabilities
- **Autonomous Defect Detection**: AI-powered system that automatically identifies and classifies defects without human intervention
- **Automated Alert Triggering**: Real-time alerts automatically triggered when defects are detected, with configurable notification systems
- **Background Task Processing**: Asynchronous processing for logging, alerting, and data persistence
- **Intelligent Decision Making**: AI-driven pass/fail determination based on confidence thresholds and defect patterns

### Core Functionality
- **Real-time Defect Detection**: Live camera feed inspection with configurable detection parameters
- **Batch Processing**: Upload and inspect multiple images simultaneously
- **AI-powered Analysis**: Uses YOLOv8L models with 97.86% accuracy for precise defect detection
- **Langflow Integration**: Advanced AI workflow processing for complex decision logic
- **Comprehensive Logging**: Detailed inspection history with automated data persistence
- **Professional UI**: Modern Streamlit-based interface for monitoring and control
- **REST API**: FastAPI backend for seamless integration with external systems

### Deployment & Automation
- **Production-Ready Architecture**: Designed for 24/7 deployment in industrial environments
- **Automated Workflows**: Background tasks handle logging, alerting, and data storage automatically
- **Event-Driven Actions**: Automatic image saving, alert generation, and database updates on defect detection
- **Scalable Infrastructure**: API-first design enables integration with manufacturing execution systems

## System Architecture

The system consists of three main components:

1. **API Server** (`api.py`): FastAPI-based REST API for image processing
2. **Camera Agent** (`camera_agent.py`): Streamlit web interface for real-time inspection
3. **Detection Engine**: YOLO-based computer vision model for defect detection

## Installation

### Prerequisites

- Python 3.8 or higher
- Camera device (for live inspection mode)
- GPU support (optional, for faster inference)

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Visual-Quality-Inspection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy `.env` file and configure your API keys
   - Update Google API key if using Langflow features

4. **Download the model**:
   - Ensure `model/best.pt` is present in the model directory
   - This is the pre-trained YOLO model for defect detection

## Usage

### Starting the System

1. **Start the API Server**:
   ```bash
   uvicorn api:app --port 8000 --reload
   ```

2. **Start the Camera Agent** (in a new terminal):
   ```bash
   streamlit run camera_agent.py
   ```

3. **Optional - Start Langflow** (for advanced AI workflows):
   ```bash
   # Start Langflow server on port 7860
   # Configure your flow with ID: a4468afd-7e5f-43e8-a509-59bbb024d2eb
   ```

### Using the Interface

#### Live Camera Mode
1. Select "Live Camera" mode in the sidebar
2. Configure detection parameters (confidence, IoU thresholds)
3. Choose whether to route through Langflow
4. Click "Start Inspection" to begin real-time analysis
5. View results and statistics in real-time

#### Batch Upload Mode
1. Select "Browse & Upload" mode
2. Upload multiple images (JPG, PNG, BMP, WebP supported)
3. Adjust detection parameters as needed
4. Click "Run Inspection" to process all images
5. Review individual results and summary statistics

## API Endpoints

The FastAPI server provides the following endpoints:

### Detection Endpoints

- `POST /api/detect` - Detect defects in uploaded image (multipart form)
- `POST /api/detect-b64` - Detect defects in base64 encoded image (JSON)

### Management Endpoints

- `POST /api/save-defect` - Save defect images
- `POST /api/save-defect-b64` - Save defect images from base64
- `POST /api/alert` - Trigger defect alert
- `POST /api/log` - Log inspection results

### System Endpoints

- `GET /api/` - Root endpoint with API info
- `GET /api/health` - Health check
- `GET /api/routes` - List all available routes

## Configuration

### Detection Parameters

- **Confidence Threshold**: Minimum confidence score for detections (0.10-0.95)
- **IoU Threshold**: Intersection over Union threshold for NMS (0.10-0.95)
- **Capture Interval**: Time between frames in live mode (1-10 seconds)

### Camera Settings

- **Camera Index**: Default is 0 (first available camera)
- **Image Quality**: JPEG encoding quality for processing

### Alert Configuration

Configure email alerts by modifying these settings in `camera_agent.py`:
```python
ALERT_EMAIL_ENABLED = True
ALERT_EMAIL_FROM = "your-email@example.com"
ALERT_EMAIL_TO = "team@example.com"
ALERT_SMTP_HOST = "smtp.example.com"
ALERT_SMTP_PORT = 587
```

## Database

The system uses SQLite for data persistence:

- **Database file**: `visionqa.db`
- **Counters table**: Tracks total inspections, passes, failures, alerts
- **History table**: Detailed inspection logs with timestamps and results

### Reset Database

Use the "Reset counters" button in the sidebar or delete the database file to start fresh.

## Output Directories

- `inspection_outputs/`: Saved defect images
- `outputs/`: Log files and alert records
- `temp_*.jpg`: Temporary files (automatically cleaned up)

## Model Information

The system uses a YOLOv8-Large (YOLOv8L) model for object detection, trained on a comprehensive dataset:

### Model Performance Metrics
- **Accuracy**: 97.86%
- **False Negative Rate**: 0.17%
- **Training Dataset**: ~4,800 images
- **Optimal Confidence Threshold**: 0.35 (recommended for best performance)

### Technical Specifications
- **Model file**: `model/best.pt`
- **Model Architecture**: YOLOv8-Large
- **Input size**: 640x640 pixels
- **Classes**: Configured for specific defect types
- **Inference speed**: Optimized for real-time processing

### Performance Notes
The model achieves excellent balance between precision and recall at the recommended confidence threshold of 0.35. This setting provides:
- High detection accuracy while minimizing false positives
- Optimal performance for industrial quality inspection applications
- Reliable detection across various lighting and surface conditions

## Integration Examples

### Python Client Example

```python
import requests
import base64

# Load image
with open("test_image.jpg", "rb") as f:
    image_data = f.read()

# Send to API
response = requests.post(
    "http://localhost:8000/api/detect",
    files={"image": ("test.jpg", image_data, "image/jpeg")},
    params={"conf": 0.35, "iou": 0.50}
)

result = response.json()
print(f"Defects detected: {result['total_detections']}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/detect" \
  -F "image=@test_image.jpg" \
  -F "conf=0.35" \
  -F "iou=0.50"
```

## Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Check camera connections
   - Verify camera index in settings
   - Ensure no other applications are using the camera

2. **API connection errors**:
   - Ensure API server is running on port 8000
   - Check firewall settings
   - Verify CORS configuration

3. **Model loading errors**:
   - Ensure `model/best.pt` exists
   - Check file permissions
   - Verify model compatibility

4. **Langflow connection issues**:
   - Ensure Langflow server is running on port 7860
   - Verify flow ID configuration
   - Check network connectivity

### Performance Optimization

- Use GPU acceleration for faster inference
- Adjust image resolution for quality/speed trade-off
- Optimize detection thresholds for your specific use case
- Consider model quantization for edge deployment

## Development

### Project Structure

```
Visual-Quality-Inspection/
|
|-- api.py              # FastAPI REST API server
|-- camera_agent.py     # Streamlit web interface
|-- requirements.txt    # Python dependencies
|-- .env               # Environment variables
|-- visionqa.db        # SQLite database
|
|-- model/             # Machine learning models
|   |-- best.pt       # YOLO model file
|
|-- utils/             # Utility functions
|   |-- detect.py     # Detection helper functions
|   |-- formatter.py  # Result formatting utilities
|
|-- outputs/           # Generated outputs
|   |-- log.txt      # Inspection logs
|   |-- alert_log.txt # Alert history
|
`-- inspection_outputs/ # Saved defect images
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add comprehensive docstrings
- Include error handling and logging

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting section
- Review the API documentation at `http://localhost:8000/docs`
- Examine the system logs for error details

## Version History

- **v5.0**: Current version with enhanced UI and Langflow integration
- **v4.0**: API improvements and better error handling
- **v3.0**: Database integration and history tracking
- **v2.0**: Streamlit interface addition
- **v1.0**: Initial release with basic detection capabilities

---

**Note**: This system is designed for industrial quality inspection applications. Ensure proper calibration and validation for your specific use case before deployment in production environments.
=======
# Visual-Quality-Inspection-agent
AI-powered Visual Quality Inspection system for detecting automotive surface and assembly defects using YOLOv8l, FastAPI, Streamlit, and LangFlow.
>>>>>>> cebd03d408b242b1b5b70b2bdcf7b2ec514b361f
