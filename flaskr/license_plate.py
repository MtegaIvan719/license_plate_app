from . import create_app  # Import the create_app function
from flask import Blueprint, render_template, request, url_for, flash, redirect, session, g, jsonify  # Add jsonify to the imports
from flask import Blueprint, render_template, request, url_for, flash, redirect, session, g
import cv2
import os
import math
import json
import threading
from flask import Blueprint, render_template, Response, current_app
from datetime import datetime, timedelta
from .db import get_db
from ultralytics import YOLO
from paddleocr import PaddleOCR
from .auth import login_required  # Import the login_required decorator

bp = Blueprint('license_plate', __name__)

@bp.route('/about')
@login_required
def about():
    return render_template('about.html')

@bp.route('/wash_process')
@login_required
def wash_process():
    """Render the Wash Process page with live video feed."""
    return render_template('wash_process.html')

@bp.route('/')
@login_required
def index():
    """Fetch today's license plate data and render the Home page."""
    today = datetime.now().date()
    db = get_db()
    rows = db.execute(
        '''
        SELECT id, start_time, end_time, license_plate, payment
        FROM LicensePlates
        WHERE DATE(start_time) = ?
        ''',
        (today.isoformat(),)
    ).fetchall()

    # Calculate the estimated amount earned
    estimated_amount = calculate_estimated_amount(rows)

    return render_template('home.html', rows=rows, estimated_amount=estimated_amount)

def calculate_estimated_amount(rows):
    """Calculate the estimated amount earned."""
    estimated_amount = sum(row['payment'] for row in rows if row['payment'] is not None)
    return estimated_amount
@bp.route('/update_data')
@login_required
def update_data():
    """Fetch updated license plate data and estimated amount earned."""
    today = datetime.now().date()
    db = get_db()
    rows = db.execute(
        '''
        SELECT id, start_time, end_time, license_plate, payment
        FROM LicensePlates
        WHERE DATE(start_time) = ?
        ''',
        (today.isoformat(),)
    ).fetchall()

    # Calculate the estimated amount earned
    estimated_amount = sum(row['payment'] for row in rows if row['payment'] is not None)

    # Convert data to a JSON-serializable format
    rows_json = [
        {
            'index': i + 1,
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'license_plate': row['license_plate'],
            'payment': row['payment']
        }
        for i, row in enumerate(rows)
    ]

    return jsonify({'rows': rows_json, 'estimated_amount': estimated_amount})


def generate_video_feed():
    """Stream video frames with YOLO detection and display license plate number."""
    cap = cv2.VideoCapture("videos/carLicence4.mp4")  # Update path to the video file
    model = YOLO("models/model.pt")  # Update path to the YOLO model file
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang='en')  # Ensure correct language and configuration
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO detection
        results = model.predict(frame, conf=0.45)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                label = paddle_ocr(frame, x1, y1, x2, y2, ocr)

                if label:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Display license plate number

        # Encode the frame and yield it as a video stream
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@bp.route('/video_feed')
@login_required
def video_feed():
    """Provide the video feed for the Wash Process page."""
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

def paddle_ocr(frame, x1, y1, x2, y2, ocr):
    """Extract text from a detected license plate using PaddleOCR."""
    frame = frame[y1:y2, x1:x2]
    result = ocr.ocr(frame, det=True, rec=True, cls=False)
    if result and result[0] is not None:
        text = " ".join(
            [line[1][0].replace("???", "").replace("O", "0") for r in result for line in r if line[1][1] > 0.6]
        )
        return text
    return ""

def video_processing(app):
    """Continuously process video frames for license plate detection and logging."""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    cap = cv2.VideoCapture("videos/carLicence4.mp4")  # Update path to the video file
    if not cap.isOpened():
        print("Error: Video file could not be opened.")
        return
    model = YOLO("models/model.pt")  # Update path to the YOLO model file
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang='en')  # Ensure correct language and configuration
    payment_amount = 5000
    license_plates = set()
    buffer = {}
    detection_timeout = timedelta(minutes=1)

    with app.app_context():  # Create an application context
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame could not be read.")
                break

            # Perform YOLO detection
            results = model.predict(frame, conf=0.45)
            current_time = datetime.now()

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = paddle_ocr(frame, x1, y1, x2, y2, ocr)

                    if label:
                        print(f"Detected license plate: {label}")  # Debugging output
                        if label in buffer:
                            buffer[label] = current_time
                        else:
                            log_entry(label, current_time)
                            buffer[label] = current_time
                            license_plates.add(label)

            # Log plates that have timed out
            for plate, last_seen in list(buffer.items()):
                if current_time - last_seen > detection_timeout:
                    print(f"Logging plate to database: {plate}")
                    save_to_database({plate}, last_seen - detection_timeout, last_seen, payment_amount)
                    log_exit(plate, last_seen)
                    del buffer[plate]

    cap.release()

def save_to_database(license_plates, start_time, end_time, payment_amount):
    """Save detected license plates into the database."""
    db = get_db()
    for plate in license_plates:
        db.execute(
            'INSERT INTO LicensePlates (start_time, end_time, license_plate, payment) VALUES (?, ?, ?, ?)',
            (start_time.isoformat(), end_time.isoformat(), plate, payment_amount)
        )
    db.commit()

def log_entry(license_plate, entry_time):
    """Log entry of a license plate."""
    db = get_db()
    db.execute(
        'INSERT INTO LicensePlates (start_time, end_time, license_plate, payment) VALUES (?, ?, ?, ?)',
        (entry_time.isoformat(), None, license_plate, 0)
    )
    db.commit()

def log_exit(license_plate, exit_time):
    """Log exit of a license plate and update payment."""
    db = get_db()
    db.execute(
        'UPDATE LicensePlates SET end_time = ?, payment = ? WHERE license_plate = ? AND end_time IS NULL',
        (exit_time.isoformat(), 5000, license_plate)
    )
    db.commit()

# Create the Flask app instance
app = create_app()

# Start video processing in a separate thread
video_thread = threading.Thread(target=video_processing, args=(app,), daemon=True)
video_thread.start()
