from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import cv2
import os
import time
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config.update(
    UPLOAD_FOLDER=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads'),
    DETECTION_FOLDER=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detections'),
    MAX_CONTENT_LENGTH=2 * 1024 * 1024 * 1024,  # 2GB max-size
    STREAM_CHUNK_SIZE=1024 * 1024  # 1MB chunks for streaming
)

# Ensure upload and detection directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTION_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<filename>')
def video_feed(filename):
    """
    แสดงวิดีโอสตรีมพร้อม bounding box
    """
    try:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(video_path):
            return jsonify({'error': 'ไม่พบไฟล์วิดีโอ'}), 404

        # เรียกใช้ detector service ผ่าน API แบบ streaming
        try:
            response = requests.get(
                'http://detector:5001/process',
                params={
                    'video_path': video_path,
                    'filename': filename
                },
                stream=True
            )
            
            if response.status_code == 200:
                return Response(
                    response.iter_content(chunk_size=None),
                    mimetype='multipart/x-mixed-replace; boundary=frame'
                )
            else:
                return jsonify({'error': f'Detector service error: {response.text}'}), 500
                
        except requests.exceptions.RequestException as e:
            print(f"ไม่สามารถเชื่อมต่อกับ detector service: {e}")
            return jsonify({'error': 'ไม่สามารถเชื่อมต่อกับระบบประมวลผลวิดีโอได้'}), 503

    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการแสดงวิดีโอ: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_video():
    """
    รับไฟล์วิดีโอและเริ่มประมวลผล
    """
    if 'video' not in request.files:
        return jsonify({'error': 'กรุณาเลือกไฟล์วิดีโอ'}), 400
        
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'ไม่ได้เลือกไฟล์'}), 400
    
    try:    
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)
        
        print(f"บันทึกวิดีโอที่: {video_path}")
        
        # แจ้ง detector service เพื่อเริ่มประมวลผล
        try:
            response = requests.post(
                'http://detector:5001/process',
                json={
                    'video_path': video_path,
                    'filename': filename
                },
                timeout=2
            )
            
            if response.status_code != 200:
                print(f"Warning: ไม่สามารถเริ่มการประมวลผลได้: {response.text}")
                
        except Exception as e:
            print(f"Warning: ไม่สามารถติดต่อ detector service ได้: {e}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'video_path': video_path
        })
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการอัปโหลด: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detections/<filename>', methods=['GET'])
def get_detection(filename):
    detection_folder = app.config['DETECTION_FOLDER']
    file_path = os.path.join(detection_folder, filename)
    if os.path.exists(file_path):
        return send_from_directory(detection_folder, filename)
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/violations', methods=['GET'])
def get_violations():
    detection_folder = app.config['DETECTION_FOLDER']
    violations = []

    # อ่านไฟล์ในโฟลเดอร์ detections
    for filename in os.listdir(detection_folder):
        if filename.endswith('_motorcycle.jpg'):  # ตรวจสอบภาพมอเตอร์ไซค์
            plate_image = filename.replace('_motorcycle.jpg', '_plate.jpg')
            if os.path.exists(os.path.join(detection_folder, plate_image)):
                # เรียก processor.py เพื่อดึงข้อมูลป้ายทะเบียน
                license_plate_text = 'รอการตรวจสอบ'
                try:
                    response = requests.get(f'http://processor:5002/ocr/{plate_image}', timeout=5)
                    if response.status_code == 200:
                        license_plate_text = response.json().get('license_plate', 'รอการตรวจสอบ')
                except Exception as e:
                    print(f"Error fetching license plate text: {e}")

                # เรียก detector.py เพื่อดึงค่า confidence
                confidence = 0.0
                try:
                    detection_response = requests.get(f'http://detector:5001/confidence/{filename}', timeout=5)
                    if detection_response.status_code == 200:
                        confidence = detection_response.json().get('confidence', 0.0)
                except Exception as e:
                    print(f"Error fetching confidence: {e}")

                violations.append({
                    'motorcycle_image': filename,
                    'plate_image': plate_image,
                    'license_plate_text': license_plate_text,  # ข้อมูลจาก processor.py
                    'confidence': confidence,  # ข้อมูลจาก detector.py
                    'timestamp': time.ctime(os.path.getmtime(os.path.join(detection_folder, filename)))
                })

    return jsonify(violations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)