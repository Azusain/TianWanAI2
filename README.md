# TianWanAI2

Multi-model object detection system based on YOLOX, including fire detection, helmet detection, and safety belt detection.

## Features

- 🔥 **Fire Detection**: Real-time fire detection in video streams
- ⛑️ **Helmet Detection**: Detect if personnel are wearing safety helmets
- 🔗 **Safety Belt Detection**: Detect if personnel are wearing safety belts

## Requirements

- Python 3.9+
- PyTorch 2.8.0+
- OpenCV 4.12.0+
- See requirements.txt for other dependencies

## Installation

1. Clone the repository
```bash
git clone https://github.com/Azusain/TianWanAI2.git
cd TianWanAI2
```

2. Create virtual environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate   # Linux/Mac
```

3. Install dependencies
```bash
pip install -r requirements.txt
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
