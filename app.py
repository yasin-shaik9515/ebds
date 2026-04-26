from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import os
from detector import EBDSDetector
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')
detector = None
camera = None
current_mode = 'drowsiness'

def init_detector_and_camera():
    global detector, camera
    try:
        detector = EBDSDetector()
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Warning: Camera not available")
            camera = None
    except Exception as e:
        print(f"Error initializing detector or camera: {e}")
        detector = None
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
    if not detector or not camera:
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + b'Camera not available\r\n')
        return
        
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame, status, ear = detector.process_frame(frame, mode=current_mode)
            latest_status = status
            
            # Emit status via socketio
            socketio.emit('status_update', {'status': status, 'mode': current_mode})
            
            # Overlay status on frame
            color = (0, 0, 255) if status == "Drowsy" else (0, 255, 0)
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

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_mode
    mode = request.json.get('mode')
    if mode in ['drowsiness', 'hci', 'attendance']:
        current_mode = mode
        # Reset attendance if switching to it
        if mode == 'attendance':
            detector.attendance_logged = False
            detector.attendance_counter = 0
        return jsonify(success=True, mode=current_mode)
    return jsonify(success=False), 400

if __name__ == '__main__':
    init_detector_and_camera()
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    socketio.run(app, host='0.0.0.0', port=port, debug=debug)
