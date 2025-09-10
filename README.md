# TianWan2

Microservice-based object detection system using YOLOX, providing fire detection, helmet safety detection, and safety belt detection through REST APIs.

## Supported Models

- 🔥 **Fire Detection** (`/fire`) - Real-time fire detection in images
- ⛑️ **Helmet Detection** (`/helmet`) - Safety helmet compliance detection
- 🔗 **Safety Belt Detection** (`/safetybelt`) - Safety belt compliance detection

## Architecture

Microservice architecture with:
- API Gateway (port 8080) - Routes requests to appropriate services
- Fire Detection Service (port 8901)
- Helmet Detection Service (port 8902) 
- Safety Belt Detection Service (port 8903)

## Docker Deployment

```bash
# Build image with all dependencies
docker build -t tianwan2:latest .

# GPU deployment
docker run -d -p 8902:8080 --gpus '"device=0"' --cpus=16 tianwan2:latest

# CPU testing
docker run -d --rm -p 8902:8080 tianwan2:latest
```

## Usage

### Command Line Usage

```bash
# Helmet detection
cd YOLO-main-helmet
python tools/demo.py --demo video --path "video_path" --conf 0.5 --save_result --device cpu

# Fire detection
cd YOLO-main-fire  
python tools/demo.py --demo video --path "video_path" --conf 0.5 --save_result --device cpu

# Safety belt detection
cd YOLO-main-safetybelt
python tools/demo.py --demo video --path "video_path" --conf 0.5 --save_result --device cpu
```

## Project Structure

```
TianWanAI2/
├── YOLO-main-fire/          # Fire detection model
├── YOLO-main-helmet/        # Helmet detection model
├── YOLO-main-safetybelt/    # Safety belt detection model
├── venv/                    # Python virtual environment
├── requirements.txt         # Dependencies
└── README.md               # Project documentation
```

## Contributing

Issues and Pull Requests are welcome.

## License

MIT License
