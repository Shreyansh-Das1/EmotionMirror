# Emotion Mirror — Real-Time Face Tracking & Emotion Analysis Engine

This repository contains a high-fidelity, containerized real-time face tracking and emotion analysis web application. The project is designed with an asynchronous processing pipeline that balances smooth client-side visualization with optimized, resource-conscious machine learning compute on the backend.

---

## Architectural & Engineering Highlights

Instead of relying on a resource-heavy monolithic loop, the system implements core streaming and runtime optimizations to ensure high performance inside constrained deployment environments:

* 
**Decoupled Dual-Loop Architecture:** Separates the application into a hardware-accelerated client rendering loop running at 60 FPS via `requestAnimationFrame` and an isolated backend ingestion pipeline throttled purposefully to 5–10 FPS over WebSockets to mitigate server network and CPU congestion.


* 
**Algorithmic Tracking Stability (MTCNN Integration):** Leverages a Multi-task Cascaded Convolutional Network (`mtcnn=True`) for facial extraction. This configuration handles minor head rotations, tilts, and varying lighting profiles cleanly, completely eliminating the bounding box jitter and tracking dropouts common in legacy Haar Cascade models.


* 
**Payload Optimization:** Converts individual video canvas frames into a 50%-compressed JPEG bitstream on the client side before transmission. This exponentially reduces the network payload size, eliminating thread choking and keeping latency near zero.


* 
**Zero-Dependency Static Asset Ingestion:** Bypasses external file-streaming package requirements (such as `aiofiles`). The application natively reads and serves the frontend shell into memory as an immutable text block via FastAPI's `HTMLResponse`.


* 
**Production Microservice Packaging:** Containerized using a clean `python:3.10-slim` baseline image. It explicitly targets `opencv-python-headless` and `tensorflow-cpu` to eliminate heavy graphical driver dependencies (X11/Qt linkages) and bloated GPU binaries, yielding a highly predictable and lightweight deployment package.



---

## Edge Case Handling & Testing Methodology

The system addresses real-world, out-of-distribution user scenarios defensively to guarantee uptime and logical execution:

1. 
**Zero-Occupancy State Optimization:** When no face is detected in the frame, the application automatically applies a dark canvas overlay mask to dim the display. This provides clear UI feedback to the user while keeping the backend pipeline in a low-overhead standby sequence until a face is re-acquired.


2. 
**Multi-Face Concurrency:** The backend pipeline seamlessly processes multiple bounding box matrices across consecutive frames, dynamically tracking independent region-of-interest (ROI) targets concurrently without dropping the connection thread.


3. 
**Hardware Display Validation Framework (Tablet Testing):** To rigorously evaluate model classification across a wide distribution of target expressions (such as intense surprise or anger) without subjective user bias, a physical tablet interface was introduced during validation. This setup fed high-contrast, pre-validated testing samples directly into the ingestion matrix to confirm classification boundaries.



---

## Deployment Blueprint

### Prerequisites

* Docker Desktop or Docker CLI installed on the host machine.



### 1. Build the Unified Container Image

Run the following command from the project root directory to compile the optimized microservice:

```bash
docker build -t emotion-mirror-app .

```

### 2. Launch the Application Container

Instantiate the service container, mapping the execution boundaries to host port 8000:

```bash
docker run -d -p 8000:8000 --name emotion-mirror emotion-mirror-app

```

### 3. Access the Live Interface

Open your web browser and navigate to:

```text
http://localhost:8000

```

> 
> **Note on Initial Startup:** Upon processing the very first frame, the environment will take roughly 5–10 seconds to download the pre-trained model weight files over the network into the running container space. Subsequent inferences execute instantaneously at full operational velocity.
> 
> 

---

## Technical Challenges & Key Learnings

The primary engineering challenge focused on mitigating **Edge-to-Server Network Saturation**. Initial stress tests that passed raw, unthrottled video frames across standard WebSocket channels choked local CPU performance. This drove latency upwards and fractured synchronization between the actual video state and the inferred bounding box overlays.

This bottleneck was structurally resolved by decoupling the application's processing speeds. Throttling frame transmission down to 10 FPS and pairing it with a 50% JPEG compression profile cut the individual frame footprint significantly, with negligible loss to emotion accuracy scores.

Furthermore, migrating the application layout inside an unprivileged Docker environment highlighted the sensitivity of binary wheel compilation for computer vision libraries. Transitioning explicitly to the headless release of OpenCV successfully prevented the engine from looking for missing graphical drivers, resulting in a stable and completely portable container architecture.

---

## File Structure Reference

```text
├── Dockerfile              # Minimal Debian-slim multi-step build configuration
├── requirements.txt        # Headless and CPU-optimized dependency pinning
├── main.py                 # FastAPI backend, native HTML response, and WebSocket engine
└── index.html              # Dynamic WebSocket connection and decoupled canvas loops

```
