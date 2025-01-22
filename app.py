import cv2
import numpy as np
import os
from flask import Flask, render_template, request, redirect, url_for, session, Response , jsonify
from werkzeug.utils import secure_filename

import csv
import time
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'
LOG_FILE = "attendance_log.csv"

frame = None

face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error: Haar Cascade xml file not loaded successfully")
else:
    print("Haar Cascade loaded successfully")


def load_images():
    dataset = []
    labels = []
    employees = {}  
    
    for idx, file in enumerate(os.listdir('photo')):
        img_path = os.path.join('photo', file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        dataset.append(image)
        labels.append(idx)
        employees[idx] = file.split('_')[0]  
    
    return dataset, np.array(labels), employees

faces, labels, employees = load_images()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)


if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Employee ID", "Name", "Action (Entry/Exit)"])

employee_status = {}  
last_logged_time = {}  
LOG_INTERVAL = 60  

def log_to_csv(employee_id, name, action):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, employee_id, name, action])



# Function to get the last 5 records from CSV
def get_latest_logs():
    try:
        with open(LOG_FILE, "r") as file:
            reader = list(csv.reader(file))
            logs = reader[1:]  # Skip header
            logs = logs[-7:]  # Get last 5 entries

            log_data = []
            for row in logs:
                log_data.append({
                    "timestamp": row[0],
                    "emp_id": row[1],
                    "name": row[2],
                    "action": row[3]
                })
            return log_data
    except Exception as e:
        print(f"Error reading log file: {e}")
        return []

# Route to serve face recognition page
@app.route('/recognize_face')
def facerec():
    records = get_latest_logs()  # Load logs on page load
    return render_template('recognize_face.html', records=records)

# API to get live updated logs (AJAX calls this)
@app.route('/get_latest_logs')
def get_latest_logs_api():
    return jsonify(get_latest_logs())



def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces_detected:
            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)

            name = employees.get(label, "Unknown")
            text = f"{name} ({round(confidence, 2)})"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            current_time = time.time()

            if label not in employee_status or employee_status[label] == 0:
                if label not in last_logged_time or current_time - last_logged_time[label] >= LOG_INTERVAL:
                    log_to_csv(label, name, "Entry")
                    employee_status[label] = 1  
                    last_logged_time[label] = current_time  

            else:
                if label not in last_logged_time or current_time - last_logged_time[label] >= LOG_INTERVAL:
                    log_to_csv(label, name, "Exit")
                    employee_status[label] = 0  
                    last_logged_time[label] = current_time  

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/facerec_video')
def facerec_video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')





@app.route('/')
def index():
    return render_template('index.html')  # Main page with Add Student button

@app.route('/add_student')
def add_student():
    return render_template('add_student.html')  # Add Student form page

@app.route('/details', methods=['POST'])
def details():
    # Get employee details from the form
    name = request.form.get("name")
    emp_id = request.form.get("emp_id")
    
    # Save the details to session
    session["name"] = name
    session["emp_id"] = emp_id

    return redirect(url_for('capture_page'))

@app.route('/capture_page')
def capture_page():
    # Get employee details from session
    name = session.get("name", "unknown")
    emp_id = session.get("emp_id", "000")
    
    # Pass name and emp_id to template
    return render_template('capture.html', name=name, emp_id=emp_id)


@app.route('/capture', methods=['POST'])
def capture():
    global frame
    if frame is None:
        print("Error: No frame captured")
        return "Error: No frame captured", 400

    # Get employee details from session
    name = session.get("name", "unknown")
    emp_id = session.get("emp_id", "000")

    # Ensure 'photo/' directory exists
    save_path = 'photo'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Decode and check frame
    np_frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    if np_frame is None:
        print("Error: Failed to decode frame")
        return "Error: Failed to process the frame", 400

    # Debugging: Frame shape before face detection
    print(f"Captured frame shape: {np_frame.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(np_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if faces are detected
    if len(faces) == 0:
        print("No faces detected")
        return "No face detected, try again!", 400

    print(f"Detected faces: {faces}")

    # Save the first detected face
    x, y, w, h = faces[0]
    face = np_frame[y:y+h, x:x+w]

    # Save the face to file
    img_path = f"{save_path}/{name}_{emp_id}.jpg"
    cv2.imwrite(img_path, face)

    # Redirect to home page with a success message
    return render_template('index.html', message="Image saved successfully!")

@app.route('/video')
def video():
    def generate():
        global frame
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/attendance_data')
def attendance_data():
    try:
        # Read the CSV file and extract attendance data
        with open(LOG_FILE, "r") as file:
            reader = list(csv.reader(file))
            logs = reader[1:]  # Skip header

        # Initialize counters
        entries = 0
        exits = 0
        total_employees = len(set([log[1] for log in logs]))  # Unique employee IDs

        # Count entries and exits
        for row in logs:
            action = row[3]  # Action (Entry/Exit)
            if action == "Entry":
                entries += 1
            elif action == "Exit":
                exits += 1

        # Return data in JSON format
        return jsonify({
            'entries': entries,
            'exits': exits,
            'total_employees': total_employees
        })
    except Exception as e:
        print(f"Error reading log file: {e}")
        return jsonify({'error': 'Failed to process the attendance data'}), 500



@app.route('/visualize_data')
def visualize_data():
    return render_template('visualize_data.html')


if __name__ == '__main__':
    app.run(debug=True)
