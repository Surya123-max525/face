from flask import Flask, render_template, Response, send_file
import cv2
import mediapipe as mp
import numpy as np
import io

app = Flask(__name__)

# ---------- Mediapipe init ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# ---------- Video capture ----------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1920)
cap.set(4, 1080)

canvas = None
prev_x, prev_y = None, None
smooth_x, smooth_y = 0, 0
alpha = 0.2
lost_frames = 0
max_lost_frames = 3
draw_color = (0, 0, 255) # BGR format
draw_thickness = 6
bg_mode = 'camera'
last_frame = None

def gen_frames():
    global canvas, prev_x, prev_y, smooth_x, smooth_y, lost_frames, draw_color, draw_thickness, bg_mode, last_frame


    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            h, w, _ = frame.shape
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        draw_mode = False
        clear_mode = False

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            h, w, _ = frame.shape

            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            smooth_x = int(alpha * ix + (1 - alpha) * smooth_x)
            smooth_y = int(alpha * iy + (1 - alpha) * smooth_y)

            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y
            ring_up = lm[16].y < lm[14].y
            pinky_up = lm[20].y < lm[18].y
            thumb_up = lm[4].y < lm[2].y
            total_fingers = sum([index_up, middle_up, ring_up, pinky_up, thumb_up])

            if index_up and not middle_up:
                draw_mode = True
            if total_fingers == 5:
                clear_mode = True

            if draw_mode:
                if prev_x is None:
                    prev_x, prev_y = smooth_x, smooth_y
                cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), draw_color, draw_thickness)
                prev_x, prev_y = smooth_x, smooth_y
                lost_frames = 0
            else:
                lost_frames += 1
                if lost_frames > max_lost_frames:
                    prev_x, prev_y = None, None

            if clear_mode:
                canvas[:] = 0
                prev_x, prev_y = None, None

        else:
            lost_frames += 1
            if lost_frames > max_lost_frames:
                prev_x, prev_y = None, None

        # Opaque overlay technique using mask
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(canvas_gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        if bg_mode == 'black':
            frame = np.zeros_like(frame)
        elif bg_mode == 'white':
            frame = np.ones_like(frame) * 255
            
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        frame = cv2.add(frame_bg, canvas)
        
        last_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_color/<color>')
def set_color(color):
    global draw_color, draw_thickness
    draw_thickness = 6
    if color == 'red': draw_color = (0, 0, 255)
    elif color == 'blue': draw_color = (255, 0, 0)
    elif color == 'green': draw_color = (0, 255, 0)
    elif color == 'yellow': draw_color = (0, 255, 255)
    elif color == 'purple': draw_color = (255, 0, 255)
    elif color == 'pink': draw_color = (203, 192, 255)
    elif color == 'eraser':
        draw_color = (0, 0, 0)
        draw_thickness = 40
    return '{"status": "success"}'

@app.route('/set_bg/<mode>')
def set_bg(mode):
    global bg_mode
    bg_mode = mode
    return '{"status": "success"}'

@app.route('/set_thickness/<int:size>')
def set_thickness(size):
    global draw_thickness
    draw_thickness = size
    return '{"status": "success"}'

@app.route('/save_image')
def save_image():
    global last_frame
    if last_frame is not None:
        ret, buffer = cv2.imencode('.png', last_frame)
        response = Response(buffer.tobytes(), mimetype='image/png')
        response.headers['Content-Disposition'] = 'attachment; filename=auradraw_masterpiece.png'
        return response
    return '{"status": "error"}'

@app.route('/clear')
def clear_board():
    global canvas
    if canvas is not None:
        canvas[:] = 0
    return '{"status": "success"}'

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)