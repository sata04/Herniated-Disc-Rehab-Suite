# Herniated Disc Rehab Suite

## Installation

1. Clone this repository.
2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

Use Streamlit to launch the web application:
```bash
streamlit run app.py
```

Navigate to the URL shown in the terminal (usually http://localhost:8501) to access the Herniated Disc Rehab Suite web interface.

## Usage

- Select a stretch from the sidebar and click **Start Exercise** to begin.
- You can stop the session anytime with the **Stop Exercise** button.
- The webcam feed will display live pose estimation and exercise feedback along with timer information.

## Configuring Stretches

Stretch definitions are stored in `config/stretch_config.yaml`. You can add or modify stretches by editing this file and restarting the app.
