from flask import Flask, request, jsonify
import cv2
import numpy as np
import easyocr
import requests
import os
import uuid
from datetime import datetime
import threading

app = Flask(__name__)
reader = easyocr.Reader(['th', 'en'])

# Initialize directories
DETECTION_FOLDER = os.getenv('DETECTION_FOLDER', 'detections')
os.makedirs(DETECTION_FOLDER, exist_ok=True)

def preprocess_plate_image(img):
   """Preprocess license plate image for better OCR"""
   try:
       # Convert to grayscale
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
       # Apply adaptive thresholding
       blur = cv2.GaussianBlur(gray, (5, 5), 0)
       thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
       
       # Noise removal
       kernel = np.ones((3,3), np.uint8)
       opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
       
       # Resize for better OCR
       height, width = opening.shape
       if width > 300:
           scale = 300 / width
           new_width = 300
           new_height = int(height * scale)
           opening = cv2.resize(opening, (new_width, new_height))
           
       return opening
   except Exception as e:
       print(f"Error preprocessing plate image: {e}")
       return img

def read_license_plate(img):
   """Read license plate text using OCR"""
   try:
       # Preprocess image
       processed_img = preprocess_plate_image(img)
       
       # Run OCR
       results = reader.readtext(processed_img)
       
       if results:
           # Combine all detected text
           text = ' '.join([r[1] for r in results])
           confidence = sum([r[2] for r in results]) / len(results)
           
           # Clean the text (remove non-alphanumeric)
           text = ''.join(c for c in text if c.isalnum() or c.isspace())
           
           return text, confidence
       return 'Unknown', 0.0
       
   except Exception as e:
       print(f"Error reading license plate: {e}")
       return 'Unknown', 0.0

def save_violation_images(detection_id, motorcycle_img, plate_img):
   """Save violation images to disk"""
   try:
       motorcycle_path = os.path.join(DETECTION_FOLDER, f"{detection_id}_motorcycle.jpg")
       plate_path = os.path.join(DETECTION_FOLDER, f"{detection_id}_plate.jpg")
       
       cv2.imwrite(motorcycle_path, motorcycle_img)
       cv2.imwrite(plate_path, plate_img)
       
       return {
           'motorcycle_image': f"{detection_id}_motorcycle.jpg",
           'plate_image': f"{detection_id}_plate.jpg"
       }
   except Exception as e:
       print(f"Error saving violation images: {e}")
       return None

def process_violation(data):
    """ประมวลผลข้อมูลการละเมิดและบันทึกลงฐานข้อมูล"""
    try:
        # อ่านและวิเคราะห์ภาพป้ายทะเบียน
        plate_path = os.path.join(DETECTION_FOLDER, data['plate_image'])
        plate_img = cv2.imread(plate_path)
        
        if plate_img is not None:
            # อ่านตัวอักษรป้ายทะเบียน
            plate_text, plate_confidence = read_license_plate(plate_img)
            
            # เตรียมข้อมูลละเมิด
            violation_data = {
                'id': data['id'],
                'video_name': data['filename'],
                'frame_number': data['frame_number'],
                'license_plate_text': plate_text,
                'license_plate_confidence': plate_confidence,
                'motorcycle_image': data['motorcycle_image'],
                'plate_image': data['plate_image'],
                'confidence': data['confidence'],
                'motorcycle_conf': data.get('motorcycle_conf', 0.0),
                'no_helmet_conf': data.get('no_helmet_conf', 0.0),
                'plate_conf': data.get('plate_conf', 0.0)
            }
            
            # บันทึกลงฐานข้อมูล
            try:
                response = requests.post(
                    'http://database:5003/violations',
                    json=violation_data,
                    timeout=5
                )
                
                if response.status_code != 200:
                    print(f"ไม่สามารถบันทึกข้อมูลได้: {response.text}")
                
                return violation_data
                
            except requests.exceptions.Timeout:
                print("หมดเวลาในการเชื่อมต่อฐานข้อมูล")
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการบันทึกข้อมูล: {e}")
        
        return None
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
        return None

@app.route('/ocr/<filename>', methods=['GET'])
def get_license_plate(filename):
    """อ่านป้ายทะเบียนจากภาพ"""
    try:
        # อ่านไฟล์ภาพ
        image_path = os.path.join(DETECTION_FOLDER, filename)
        if not os.path.exists(image_path):
            return jsonify({
                'error': 'ไม่พบไฟล์ภาพ'
            }), 404
            
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({
                'error': 'ไม่สามารถอ่านไฟล์ภาพได้'
            }), 400
            
        # อ่านป้ายทะเบียน
        text, confidence = read_license_plate(img)
        
        return jsonify({
            'license_plate': text,
            'confidence': confidence
        })
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการอ่านป้ายทะเบียน: {e}")
        return jsonify({'error': str(e)}), 500

# @app.route('/process_frame', methods=['POST'])
# def process_frame():
#    try:
#        data = request.get_json()
#        if not data:
#            return jsonify({'error': 'No data received'}), 400
           
#        # Process violation asynchronously
#        threading.Thread(target=process_violation, args=(data,)).start()
       
#        return jsonify({'success': True})
       
#    except Exception as e:
#        print(f"Error in process_frame: {e}")
#        return jsonify({'error': str(e)}), 500

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """เอนด์พอยต์สำหรับรับและประมวลผลเฟรมที่ตรวจพบการละเมิด"""
    try:
        data = request.get_json()
        if not data:
            print("ไม่พบข้อมูลในคำขอ")
            return jsonify({'error': 'ไม่พบข้อมูล'}), 400

        # ตรวจสอบข้อมูลที่จำเป็น
        required_fields = ['id', 'filename', 'frame_number', 'plate_image']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            print(f"ข้อมูลไม่ครบถ้วน: ขาด {', '.join(missing_fields)}")
            return jsonify({'error': f'ข้อมูลไม่ครบถ้วน: {missing_fields}'}), 400

        # ตรวจสอบไฟล์ภาพป้ายทะเบียน
        plate_path = os.path.join(DETECTION_FOLDER, data['plate_image'])
        if not os.path.exists(plate_path):
            print(f"ไม่พบไฟล์ภาพป้ายทะเบียน: {plate_path}")
            return jsonify({'error': 'ไม่พบไฟล์ภาพป้ายทะเบียน'}), 404

        # อ่านไฟล์ภาพ
        plate_img = cv2.imread(plate_path)
        if plate_img is None:
            print(f"ไม่สามารถอ่านไฟล์ภาพป้ายทะเบียนได้: {plate_path}")
            return jsonify({'error': 'ไม่สามารถอ่านไฟล์ภาพ'}), 400

        print(f"เริ่มกระบวนการ OCR สำหรับไฟล์: {data['plate_image']}")
        # อ่านตัวอักษรป้ายทะเบียน
        plate_text, ocr_confidence = read_license_plate(plate_img)
        print(f"ผลการอ่าน OCR: ข้อความ='{plate_text}', ความแม่นยำ={ocr_confidence:.2f}")

        # เตรียมข้อมูลสำหรับบันทึก
        violation_data = {
            'id': data['id'],
            'video_name': data['filename'],
            'frame_number': data['frame_number'],
            'license_plate_text': plate_text,
            'license_plate_confidence': ocr_confidence,
            'motorcycle_image': data.get('motorcycle_image', ''),
            'plate_image': data['plate_image'],
            'confidence': data.get('confidence', 0.0),
            'motorcycle_conf': data.get('motorcycle_conf', 0.0),
            'no_helmet_conf': data.get('no_helmet_conf', 0.0),
            'plate_conf': data.get('plate_conf', 0.0)
        }

        # บันทึกลงฐานข้อมูล
        try:
            response = requests.post(
                'http://database:5003/violations',
                json=violation_data,
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"บันทึกข้อมูลสำเร็จ: ID={data['id']}")
                return jsonify({
                    'success': True,
                    'message': 'บันทึกข้อมูลสำเร็จ',
                    'data': violation_data
                })
            else:
                print(f"ไม่สามารถบันทึกข้อมูลได้: {response.text}")
                return jsonify({
                    'error': f'ไม่สามารถบันทึกข้อมูลได้: {response.text}'
                }), 500
                
        except requests.exceptions.Timeout:
            print("หมดเวลาในการเชื่อมต่อฐานข้อมูล")
            return jsonify({'error': 'หมดเวลาในการเชื่อมต่อฐานข้อมูล'}), 500
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการบันทึกข้อมูล: {e}")
            return jsonify({'error': f'เกิดข้อผิดพลาดในการบันทึกข้อมูล: {str(e)}'}), 500

    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
   """Health check endpoint"""
   return jsonify({'status': 'healthy'})

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5002)