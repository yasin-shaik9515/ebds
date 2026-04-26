from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import os
import sys
import logging
from detector import EBDSDetector
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")
detector = None
camera = None
current_mode = 'drowsiness'

def init_detector_and_camera():
    global detector, camera
    
    # Try to initialize detector (MediaPipe)
    try:
        logger.info("Initializing detector...")
        detector = EBDSDetector()
        logger.info("Detector initialized successfully")
    except (ImportError, OSError, EnvironmentError) as e:
        logger.warning(f"MediaPipe graphics libraries not available: {e}")
        logger.info("Running without face detection features")
        detector = None
    except Exception as e:
        logger.error(f"Error initializing detector: {e}")
        detector = None
    
    # Try to initialize camera (independent of detector)
    try:
        logger.info("Initializing camera...")
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logger.warning("Camera not available - running in demo mode")
            camera = None
        else:
            logger.info("Camera initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        camera = None

def status_worker():
    """Background worker to send status updates via SocketIO"""
    global current_mode
    while True:
        # We don't want to process again here, but we need the latest status.
        # For simplicity, we'll let the gen_frames update a global status.
        time.sleep(0.1)

latest_status = "Normal"

def gen_frames():
    global current_mode, latest_status, detector, camera
    if not camera:
        # Create a demo frame with status message
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Demo Mode - No Camera Available", (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Mode: {current_mode}", (50, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Status: {latest_status}", (50, 280), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Deployed on Render", (50, 350), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    
    # Camera is available
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if detector:
                # Full processing with face detection
                frame, status, ear = detector.process_frame(frame, mode=current_mode)
                latest_status = status
            else:
                # Basic video streaming without processing
                status = "Camera Active (No Face Detection)"
                latest_status = status
            
            # Emit status via socketio
            socketio.emit('status_update', {'status': status, 'mode': current_mode})
            
            # Overlay status on frame
            color = (0, 0, 255) if "Drowsy" in status else (0, 255, 0)
            cv2.putText(frame, f"Mode: {current_mode}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Status: {status}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint for Render monitoring"""
    mode = 'full' if detector and camera else 'camera-only' if camera else 'demo'
    return jsonify({
        'status': 'ok',
        'mode': mode,
        'detector': 'ready' if detector else 'not available (no face detection)',
        'camera': 'ready' if camera else 'not available',
        'message': f'App is running in {mode} mode on Render'
    }), 200

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_mode, detector
    mode = request.json.get('mode')
    if mode in ['drowsiness', 'hci', 'attendance']:
        current_mode = mode
        # Reset attendance if switching to it
        if mode == 'attendance' and detector:
            detector.attendance_logged = False
            detector.attendance_counter = 0
        return jsonify(success=True, mode=current_mode)
    return jsonify(success=False), 400

if __name__ == '__main__':
    init_detector_and_camera()
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting app on port {port}, debug={debug}")
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f"Failed to start app: {e}")
        sys.exit(1)
