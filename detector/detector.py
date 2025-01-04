from flask import Flask, request, jsonify, Response
from ultralytics import YOLO
import cv2
import torch
import os
import redis
import json
import numpy as np
import requests
import threading
import time
from threading import Event, Lock
from queue import Queue

app = Flask(__name__)
model = None
redis_client = redis.Redis(host='redis', port=6379)

# ตัวแปรสำหรับจัดการการประมวลผลวิดีโอ
video_processes = {}
process_locks = {}
processing_status = {}
video_queues = {}
video_threads = {}
MAX_QUEUE_SIZE = 10

# สร้างโฟลเดอร์สำหรับเก็บภาพที่ตรวจจับได้
DETECTION_FOLDER = os.getenv('DETECTION_FOLDER', 'detections')
os.makedirs(DETECTION_FOLDER, exist_ok=True)

def load_model():
    """โหลดโมเดล YOLO สำหรับตรวจจับวัตถุ"""
    global model
    model_path = os.getenv('MODEL_PATH')
    print(f"กำลังโหลดโมเดลจาก: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {model_path}")
    model = YOLO(model_path)

def save_detection_image(image, filename):
    """บันทึกภาพที่ตรวจจับได้"""
    path = os.path.join(DETECTION_FOLDER, filename)
    cv2.imwrite(path, image)
    return filename

def send_to_processor(filename, frame_number, detections):
    """
    ส่งข้อมูลการตรวจจับไปยัง processor service พร้อมค่าความแม่นยำ
    
    Args:
        filename (str): ชื่อไฟล์วิดีโอ
        frame_number (int): เฟรมที่ตรวจพบการละเมิด
        detections (list): ผลการตรวจจับจาก YOLO model
    """
    try:
        # สร้าง dictionary สำหรับเก็บค่า confidence แต่ละประเภท
        confidences = {
            'motorcycle': 0.0,
            'no_helmet': 0.0,
            'plate': 0.0
        }
        
        # ดึงค่า confidence จากแต่ละการตรวจจับ
        for detection in detections:
            class_id = int(detection[5])
            confidence = float(detection[4])
            
            if class_id == 0:  # Motorcycle
                confidences['motorcycle'] = confidence
            elif class_id == 3:  # NoHelmet
                confidences['no_helmet'] = confidence
            elif class_id == 2:  # LicensePlate
                confidences['plate'] = confidence
        
        # คำนวณค่าความแม่นยำเฉลี่ย
        avg_confidence = sum(confidences.values()) / len(confidences)
        detection_id = f"{filename}_frame{frame_number}"
        
        # บันทึกสถานะการประมวลผล
        status_data = {
            'is_processing': True,
            'confidence': avg_confidence,
            'motorcycle_conf': confidences['motorcycle'],
            'no_helmet_conf': confidences['no_helmet'],
            'plate_conf': confidences['plate']
        }
        
        if filename in processing_status:
            if isinstance(processing_status[filename], dict):
                processing_status[filename].update(status_data)
            else:
                processing_status[filename] = status_data

        # ส่งข้อมูลไปยัง processor
        payload = {
            'id': detection_id,
            'filename': filename,
            'frame_number': frame_number,
            'motorcycle_image': f"{detection_id}_motorcycle.jpg",
            'plate_image': f"{detection_id}_plate.jpg",
            'confidence': avg_confidence,
            'motorcycle_conf': confidences['motorcycle'],
            'no_helmet_conf': confidences['no_helmet'],
            'plate_conf': confidences['plate']
        }
        
        response = requests.post(
            'http://processor:5002/process_frame',
            json=payload,
            timeout=5
        )
        
        if response.status_code != 200:
            print(f"ไม่สามารถส่งข้อมูลไปยัง processor ได้: {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"หมดเวลาในการส่งข้อมูลสำหรับเฟรม {frame_number}")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการส่งข้อมูล: {e}")
        print(f"ข้อมูล: filename={filename}, frame={frame_number}")

@app.route('/confidence/<filename>', methods=['GET'])
def get_confidence(filename):
    """ดึงค่าความแม่นยำจากการตรวจจับ"""
    try:
        print(f"กำลังดึงค่าความแม่นยำสำหรับ: {filename}")
        
        video_name = filename
        if '_motorcycle.jpg' in filename:
            video_name = filename.replace('_motorcycle.jpg', '')
            if '_frame' in video_name:
                video_name = video_name.split('_frame')[0]
        
        print(f"วิดีโอ: {video_name}")
        
        if not video_name in processing_status:
            print(f"ไม่พบข้อมูลความแม่นยำสำหรับ: {video_name}")
            processing_status[video_name] = {
                'confidence': 0.0,
                'motorcycle_conf': 0.0,
                'no_helmet_conf': 0.0,
                'plate_conf': 0.0
            }
        
        result = {
            'confidence': float(processing_status[video_name].get('confidence', 0.0)),
            'motorcycle_conf': float(processing_status[video_name].get('motorcycle_conf', 0.0)),
            'no_helmet_conf': float(processing_status[video_name].get('no_helmet_conf', 0.0)),
            'plate_conf': float(processing_status[video_name].get('plate_conf', 0.0))
        }
        
        print(f"ค่าความแม่นยำ: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการดึงค่าความแม่นยำ: {e}")
        print(f"filename: {filename}")
        print(f"video_name: {video_name if 'video_name' in locals() else 'ไม่สามารถแยกชื่อวิดีโอได้'}")
        
        return jsonify({
            'error': str(e),
            'confidence': 0.0,
            'motorcycle_conf': 0.0,
            'no_helmet_conf': 0.0,
            'plate_conf': 0.0
        }), 200

@app.route('/stop', methods=['POST'])
def stop_processing():
    """หยุดการประมวลผลวิดีโอ"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'ต้องระบุชื่อไฟล์'}), 400

        with process_locks.get(filename, Lock()):
            if filename in processing_status:
                processing_status[filename]['is_processing'] = False
                if filename in video_processes:
                    video_processes[filename].set()
                return jsonify({'success': True})
            return jsonify({'error': 'ไม่พบการประมวลผลที่กำลังทำงาน'}), 404

    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการหยุดการประมวลผล: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['GET', 'POST'])
def process_video():
    """ประมวลผลวิดีโอและตรวจจับวัตถุ"""
    try:
        # รับพารามิเตอร์
        if request.method == 'GET':
            video_path = request.args.get('video_path')
            filename = request.args.get('filename')
        else:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'ไม่พบข้อมูล'}), 400
            video_path = data.get('video_path')
            filename = data.get('filename')

        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': f'ไม่พบไฟล์วิดีโอ: {video_path}'}), 400

        # สร้าง queue สำหรับเก็บเฟรม
        if filename not in video_queues:
            video_queues[filename] = Queue(maxsize=MAX_QUEUE_SIZE)
            processing_status[filename] = {
                'is_processing': True,
                'confidence': 0.0,
                'motorcycle_conf': 0.0,
                'no_helmet_conf': 0.0,
                'plate_conf': 0.0
            }

            def process_video_frames():
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                frame_count = 0
                last_frame_time = time.time()
                
                try:
                    while cap.isOpened() and processing_status[filename]['is_processing']:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame = cv2.resize(frame, (854, 480))
                        
                        # ตรวจจับวัตถุด้วย YOLO
                        with torch.inference_mode():
                            results = model(frame, conf=0.6, iou=0.5, max_det=10, agnostic_nms=True)[0]
                        
                        if len(results.boxes) > 0:
                            # เก็บภาพต้นฉบับ
                            original_frame = frame.copy()
                            
                            boxes = results.boxes
                            xyxy = boxes.xyxy.cpu().numpy()
                            cls = boxes.cls.cpu().numpy()
                            conf = boxes.conf.cpu().numpy()
                            
                            # เก็บพิกัดของวัตถุที่ตรวจพบ
                            detections = {
                                'motorcycle': None,
                                'no_helmet': False,
                                'plate': None
                            }
                            
                            # วนลูปตรวจสอบวัตถุที่พบ
                            for i, box in enumerate(xyxy):
                                x1, y1, x2, y2 = map(int, box)
                                class_id = int(cls[i])
                                confidence = conf[i]
                                
                                # จัดเก็บพิกัดตามประเภท
                                if class_id == 0:  # Motorcycle
                                    detections['motorcycle'] = (x1, y1, x2, y2)
                                elif class_id == 3:  # NoHelmet
                                    detections['no_helmet'] = True
                                elif class_id == 2:  # LicensePlate
                                    detections['plate'] = (x1, y1, x2, y2)
                                
                                # วาด bounding box
                                color, label = colors.get(class_id, ((128, 128, 128), 'ไม่ทราบ'))
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                # เพิ่มป้ายกำกับ
                                label_text = f'{label} ({confidence:.2f})'
                                (text_width, text_height), _ = cv2.getTextSize(
                                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(
                                    frame, 
                                    (x1, y1 - text_height - 10),
                                    (x1 + text_width + 10, y1),
                                    color, 
                                    -1
                                )
                                cv2.putText(
                                    frame,
                                    label_text,
                                    (x1 + 5, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 255),
                                    2
                                )

                            # บันทึกภาพเมื่อตรวจพบครบทุกเงื่อนไข
                            if all([detections['motorcycle'], detections['no_helmet'], detections['plate']]):
                                # ตัดภาพจากภาพต้นฉบับ
                                x1, y1, x2, y2 = detections['motorcycle']
                                motorcycle_img = original_frame[y1:y2, x1:x2].copy()
                                
                                x1, y1, x2, y2 = detections['plate']
                                plate_img = original_frame[y1:y2, x1:x2].copy()
                                
                                motorcycle_filename = save_detection_image(
                                    motorcycle_img, f"{filename}_frame{frame_count}_motorcycle.jpg")
                                plate_filename = save_detection_image(
                                    plate_img, f"{filename}_frame{frame_count}_plate.jpg")
                                
                                threading.Thread(
                                    target=send_to_processor,
                                    args=(filename, frame_count, results.boxes.data.tolist()),
                                    daemon=True
                                ).start()

                        # แสดงสถานะ
                        cv2.putText(
                            frame,
                            'Processing...',
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2
                        )

                        # แสดง FPS
                        fps = 1.0 / (time.time() - last_frame_time)
                        cv2.putText(
                            frame,
                            f'FPS: {fps:.1f}',
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,(255, 255, 255),
                            2
                        )

                        # แปลงภาพและเพิ่มลงใน queue
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        frame_data = buffer.tobytes()
                        
                        try:
                            if not video_queues[filename].full():
                                video_queues[filename].put_nowait(frame_data)
                            else:
                                # ถ้า queue เต็ม ให้นำ frame เก่าออกก่อน
                                video_queues[filename].get_nowait()
                                video_queues[filename].put_nowait(frame_data)
                        except:
                            continue

                        # ปรับ frame rate
                        frame_count += 1
                        elapsed_time = time.time() - last_frame_time
                        if elapsed_time < 1/30:  # รักษา FPS ที่ 30
                            time.sleep(1/30 - elapsed_time)
                        last_frame_time = time.time()

                        # ล้าง GPU memory ทุก 30 เฟรม
                        if frame_count % 30 == 0:
                            torch.cuda.empty_cache()
                            
                except Exception as e:
                    print(f"เกิดข้อผิดพลาดในการประมวลผลเฟรม: {e}")
                finally:
                    cap.release()
                    processing_status[filename]['is_processing'] = False

            # สร้าง thread สำหรับประมวลผลวิดีโอ
            process_thread = threading.Thread(target=process_video_frames, daemon=True)
            video_threads[filename] = process_thread
            process_thread.start()

        def generate_frames():
            """ฟังก์ชันสร้าง video stream"""
            try:
                while processing_status[filename]['is_processing'] or not video_queues[filename].empty():
                    try:
                        frame_data = video_queues[filename].get(timeout=0.1)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                    except:
                        continue
            finally:
                # ทำความสะอาดทรัพยากรเมื่อ generator ถูกปิด
                if filename in video_queues:
                    while not video_queues[filename].empty():
                        try:
                            video_queues[filename].get_nowait()
                        except:
                            pass
                if filename in processing_status:
                    processing_status[filename]['is_processing'] = False
                print(f"ปิด generator สำหรับ {filename}")

        # ส่ง Response แบบ streaming
        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            direct_passthrough=True
        )

    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการประมวลผลวิดีโอ: {e}")
        return jsonify({'error': str(e)}), 500

# กำหนดสีและชื่อคลาส
colors = {
    0: ((255, 140, 0), 'Motorcycle'),
    1: ((0, 255, 50), 'Helmet'),
    2: ((255, 255, 0), 'LicensePlate'),
    3: ((0, 0, 255), 'NoHelmet')
}

if __name__ == '__main__':
    try:
        load_model()
        app.run(host='0.0.0.0', port=5001)
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการเริ่มต้นบริการ: {e}")