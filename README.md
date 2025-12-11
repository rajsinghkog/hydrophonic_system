# Disease Classification App

A Streamlit-based web application for disease classification using deep learning.

## Setup

1. **Create a virtual environment** (if not already created):
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

### Option 1: Using the startup script (macOS/Linux)
```bash
./run_app.sh
```

### Option 2: Manual run
```bash
source venv/bin/activate
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

- Upload images for disease classification
- Use camera to capture images
- Real-time predictions with confidence scores
- Visual probability distribution charts

## Requirements

- Python 3.7+
- `model.pth` file (must be in the project root)
- `classes.json` file (must be in the project root)


# hydrophonic_system
# hydrophonic_system
