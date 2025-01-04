from flask import Flask, request, jsonify
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
DB_PATH = os.getenv('DB_PATH', 'violations.db')

def init_db():
    """สร้างฐานข้อมูลและตารางเก็บข้อมูลการละเมิด พร้อมค่าความแม่นยำแยกตามประเภท"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    id TEXT PRIMARY KEY,
                    video_name TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    license_plate_text TEXT,
                    license_plate_confidence FLOAT,
                    motorcycle_image TEXT,
                    plate_image TEXT,
                    confidence FLOAT,
                    motorcycle_conf FLOAT,
                    no_helmet_conf FLOAT,
                    plate_conf FLOAT
                )
            ''')
            print("สร้างตารางฐานข้อมูลสำเร็จ")
    except sqlite3.Error as e:
        print(f"เกิดข้อผิดพลาดในการสร้างฐานข้อมูล: {e}")
        raise

@app.route('/violations', methods=['POST'])
def add_violation():
    """บันทึกข้อมูลการละเมิดพร้อมค่าความแม่นยำแยกตามประเภท"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'ไม่พบข้อมูลที่ส่งมา'}), 400

        # ตรวจสอบข้อมูลที่จำเป็น
        required_fields = ['id', 'video_name', 'frame_number']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'ข้อมูลไม่ครบถ้วน: {", ".join(missing_fields)}'
            }), 400

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                '''INSERT INTO violations 
                   (id, video_name, frame_number, timestamp,
                    license_plate_text, license_plate_confidence,
                    motorcycle_image, plate_image, confidence,
                    motorcycle_conf, no_helmet_conf, plate_conf)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    data['id'],
                    data['video_name'],
                    data['frame_number'],
                    datetime.now(),
                    data.get('license_plate_text', 'รอการตรวจสอบ'),
                    data.get('license_plate_confidence', 0.0),
                    data.get('motorcycle_image', ''),
                    data.get('plate_image', ''),
                    data.get('confidence', 0.0),  # ค่าความแม่นยำรวม
                    data.get('motorcycle_conf', 0.0),  # ความแม่นยำการตรวจจับรถ
                    data.get('no_helmet_conf', 0.0),   # ความแม่นยำการตรวจจับคนไม่ใส่หมวก
                    data.get('plate_conf', 0.0)        # ความแม่นยำการตรวจจับป้าย
                )
            )
            return jsonify({
                'success': True,
                'message': 'บันทึกข้อมูลสำเร็จ'
            })

    except sqlite3.IntegrityError:
        return jsonify({
            'error': 'ข้อมูลซ้ำ: พบ ID นี้ในระบบแล้ว'
        }), 409
    except sqlite3.Error as e:
        print(f"เกิดข้อผิดพลาดในการบันทึกข้อมูล: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")
        return jsonify({'error': 'เกิดข้อผิดพลาดในการประมวลผล'}), 500

@app.route('/violations', methods=['GET'])
def fetch_violations():
    """ดึงข้อมูลการละเมิดพร้อมค่าความแม่นยำทั้งหมด"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM violations 
                ORDER BY timestamp DESC
            ''')
            violations = [dict(row) for row in cursor.fetchall()]
            return jsonify(violations)

    except sqlite3.Error as e:
        print(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    init_db()
    app.run(host='0.0.0.0', port=5003)