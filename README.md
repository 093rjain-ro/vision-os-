# Vision OS: Edge-Deployed IoT Security & Analytics System

Vision OS is a comprehensive, edge-deployable computer vision and IoT security appliance designed for corporate offices, industrial zones, and gated facilities. By combining lightweight AI models (YOLOv8, FaceNet) with localized edge processing, it delivers real-time access control, smart attendance, and automated hardware actuation without relying on cloud computation.

## 🚀 Key Features

### 1. Smart Attendance & Person Tracking
*   **Face Matching:** Local facial embedding matching for real-time employee check-ins.
*   **Soft-Attribute Filtering:** Debounced gender and clothing color classification for tracking individuals in restricted zones without strict biometric facial ID, adhering to privacy constraints.
*   **Centroid Tracking:** Assigns unique IDs to prevent database spam and ensure accurate event logging.

### 2. Vehicle Access Control
*   **ALPR Pipeline:** YOLO-based license plate detection paired with an automated Image Enhancer (Fast NL Means Denoising) and EasyOCR for high-accuracy reads on degraded CCTV feeds.
*   **Watchlist Routing:** Automatically grants or denies access based on a local `config.yaml` watchlist.

### 3. Edge-First Architecture & Hardware Integration
*   **GPIO Actuation:** Native support for Raspberry Pi / Jetson GPIO (via `gpiozero`) to physically trigger servo barriers and LED alarm sirens. 
*   **Mock Mode Support:** Seamlessly run on Windows/Mac without crashing by simulating hardware interfaces.
*   **Hardware Sensors:** Background threaded loops handling ADXL345 vibration/tamper events and monitoring voltage drops for backup-battery failovers.

### 4. Advanced Alerting & Storage
*   **Cellular Store-and-Forward:** Uses MQTT to push JSON alerts (with approximate location). If the connection drops, alerts are queued locally in SQLite and retried.
*   **Dual-Stream Capture:** Maintains a continuous low-res rolling buffer (Stream A) to save disk space, while saving high-res screenshot evidence (Stream B) precisely when AI events trigger.
*   **Dial-In IVR Check:** Simulated text-to-speech status endpoint.

### 5. Multi-Page Dashboard
*   A sleek, hot-reloading Streamlit interface featuring Live Camera feeds, Attendance Logs, Tamper Alarms, and a Remote Disarm override.

## 🛠️ Tech Stack
*   **Core:** Python 3.10+, OpenCV, SQLite
*   **AI Models:** Ultralytics YOLOv8, EasyOCR
*   **IoT & Hardware:** Paho MQTT, gpiozero
*   **UI:** Streamlit, Pandas

## ⚙️ Quick Start

1. **Clone & Install Dependencies:**
   ```bash
   git clone https://github.com/093rjain-ro/vision-os-.git
   cd vision-os-
   pip install -r requirements.txt
   ```

2. **Configure Watchlists:**
   Edit `config/config.yaml` to add authorized license plates or restricted attributes (e.g., flagging 'black' clothing to enforce hi-vis vest compliance in hazard zones).
   
   **ALPR Configuration:**
   Configure `ocr_every_n_frames`, `min_plate_confidence`, and `plate_format_regex` under the `alpr:` section to optimize performance and filter out invalid reads at high-throughput gates.
   
   **Visitor Auto-ID & Privacy:**
   You can enable auto-tracking for unenrolled faces via `visitor_management.enable_visitor_auto_id`. 
   *Note: Auto-collecting biometric embeddings of unenrolled visitors may require legal/consent review depending on your jurisdiction. A `visitor_retention_days` guardrail (default 30 days) is included to automatically purge stale records.*

3. **Run the Edge Appliance:**
   Open two terminals:
   ```bash
   # Terminal 1: Start the AI pipeline and hardware loops
   python main.py
   
   # Terminal 2: Launch the local administrative dashboard
   python -m streamlit run dashboard/app.py
   ```

4. **Access the Dashboard:**
   Open a browser and navigate to `http://localhost:8501`.

## 📁 Repository Structure
*   `/src/core/` - Video ingestion, denoising, and dual-stream storage managers.
*   `/src/detection/` - AI extractors for plates, faces, fire/smoke, and attributes.
*   `/src/hardware/` - GPIO actuators, battery/tamper simulators, and MQTT comms.
*   `/dashboard/` - The Streamlit UI interface.
*   `/data/` - Local SQLite database and saved event frames.
*   `/config/` - YAML rule configurations.

## 📄 License
This project was built as a formal semester project addressing real-world edge IoT constraints. License pending.
