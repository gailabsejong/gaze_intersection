import cv2
import mediapipe as mp
import numpy as np
import math

# YOLO 모델 설정
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# COCO 클래스 파일 로드
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 비디오 파일 열기
cap = cv2.VideoCapture('group1_front.mp4')

angles = []
gaze_intersection_count = 0

def calculate_cosine_similarity(angle1, angle2):
    vector1 = [math.cos(angle1), math.sin(angle1)]
    vector2 = [math.cos(angle2), math.sin(angle2)]
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    return dot_product / (magnitude1 * magnitude2)

# 선분과 직사각형의 교차 확인 함수
def line_rectangle_intersection(line, rect):
    x1, y1, x2, y2 = line
    rx, ry, rw, rh = rect

    left = line_intersection((x1, y1, x2, y2), (rx, ry, rx, ry+rh))
    right = line_intersection((x1, y1, x2, y2), (rx+rw, ry, rx+rw, ry+rh))
    top = line_intersection((x1, y1, x2, y2), (rx, ry, rx+rw, ry))
    bottom = line_intersection((x1, y1, x2, y2), (rx, ry+rh, rx+rw, ry+rh))

    return left or right or top or bottom

# 선분 교차 확인 함수
def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return False

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

    if 0 <= t <= 1 and 0 <= u <= 1:
        return True
    return False

def calculate_shoulder_distance(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    return math.sqrt((left_shoulder.x - right_shoulder.x)**2 + (left_shoulder.y - right_shoulder.y)**2)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape

        # YOLO를 사용하여 사람 감지
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 0 and confidence > 0.5:  # '0'은 'person' 클래스
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        people_data = []  # 각 사람의 데이터를 저장할 리스트
        max_shoulder_distance = 0
        front_facing_person = None

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                crop_img = frame[y:y+h, x:x+w]

                # BGR 이미지를 RGB로 변환
                crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                crop_img_rgb.flags.writeable = False

                # Pose 검출
                results = pose.process(crop_img_rgb)

                crop_img_rgb.flags.writeable = True
                crop_img = cv2.cvtColor(crop_img_rgb, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        crop_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )

                    landmarks = results.pose_landmarks.landmark
                    shoulder_distance = calculate_shoulder_distance(landmarks)

                    if shoulder_distance > max_shoulder_distance:
                        max_shoulder_distance = shoulder_distance
                        front_facing_person = i

                    # 좌표 추출
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
                    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]
                    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
                    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]

                    # 목 좌표 계산
                    neck_x = (left_shoulder.x + right_shoulder.x) / 2
                    neck_y = (left_shoulder.y + right_shoulder.y) / 2

                    # 귀의 중간 좌표 계산
                    ear_mid_x = (left_ear.x + right_ear.x) / 2
                    ear_mid_y = (left_ear.y + right_ear.y) / 2

                    # 눈의 중간 좌표 계산
                    eye_mid_x = (left_eye.x + right_eye.x) / 2
                    eye_mid_y = (left_eye.y + right_eye.y) / 2

                    # 좌표값을 이미지 크기에 맞게 변환
                    ch, cw, _ = crop_img.shape
                    neck_x, neck_y = int(neck_x * cw), int(neck_y * ch)
                    ear_mid_x, ear_mid_y = int(ear_mid_x * cw), int(ear_mid_y * ch)
                    eye_mid_x, eye_mid_y = int(eye_mid_x * cw), int(eye_mid_y * ch)
                    nose_x, nose_y = int(nose.x * cw), int(nose.y * ch)

                    # 시선 방향 계산 (정면을 보는 사람과 그 외의 사람 구분)
                    if i == front_facing_person:
                        gaze_x1, gaze_y1 = eye_mid_x, eye_mid_y
                        gaze_x2, gaze_y2 = nose_x, nose_y
                    else:
                        gaze_x1, gaze_y1 = neck_x, neck_y
                        gaze_x2, gaze_y2 = ear_mid_x, ear_mid_y

                    # 시선 방향선 그리기
                    cv2.line(crop_img, (gaze_x1, gaze_y1), (gaze_x2, gaze_y2), (255, 0, 0), 2)
                    cv2.circle(crop_img, (gaze_x1, gaze_y1), 3, (0, 255, 0), -1)
                    cv2.circle(crop_img, (gaze_x2, gaze_y2), 3, (0, 0, 255), -1)

                    # 각도 계산
                    angle = math.degrees(math.atan2(gaze_y2 - gaze_y1, gaze_x2 - gaze_x1))
                    angles.append(angle)
                    cv2.putText(crop_img, f"Angle: {angle:.2f}", (gaze_x1, gaze_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # 시선 방향선 연장
                    length = math.sqrt((gaze_x2 - gaze_x1) ** 2 + (gaze_y2 - gaze_y1) ** 2)
                    extension_factor = max(width, height) / length
                    if i == front_facing_person:
                        perp_x1 = int(gaze_x2 - length * extension_factor)
                        perp_y1 = int(gaze_y2 + length * extension_factor)
                        perp_x2 = int(gaze_x2 + length * extension_factor)
                        perp_y2 = int(gaze_y2 - length * extension_factor)
                    else:
                        perp_x1 = int(gaze_x2 - length * extension_factor * math.sin(math.radians(angle)))
                        perp_y1 = int(gaze_y2 + length * extension_factor * math.cos(math.radians(angle)))
                        perp_x2 = int(gaze_x2 + length * extension_factor * math.sin(math.radians(angle)))
                        perp_y2 = int(gaze_y2 - length * extension_factor * math.cos(math.radians(angle)))

                    # 전체 프레임에서의 좌표로 변환
                    global_perp_x1 = x + perp_x1
                    global_perp_y1 = y + perp_y1
                    global_perp_x2 = x + perp_x2
                    global_perp_y2 = y + perp_y2

                    # 노란색 시선 방향 선 그리기 (전체 프레임에)
                    cv2.line(frame, (global_perp_x1, global_perp_y1), (global_perp_x2, global_perp_y2), (0, 255, 255), 2)

                    # 머리 바운딩 박스 계산 (코와 귀를 포함하는 영역)
                    head_margin = 0.2  # 머리 크기의 20%를 여백으로 추가
                    head_width = max(right_ear.x, nose.x, left_ear.x) - min(right_ear.x, nose.x, left_ear.x)
                    head_height = max(right_ear.y, nose.y, left_ear.y) - min(right_ear.y, nose.y, left_ear.y)

                    head_x_min = min(nose.x, left_ear.x, right_ear.x) - head_width * head_margin
                    head_x_max = max(nose.x, left_ear.x, right_ear.x) + head_width * head_margin
                    head_y_min = min(nose.y, left_ear.y, right_ear.y) - head_height * head_margin
                    head_y_max = max(nose.y, left_ear.y, right_ear.y) + head_height * head_margin

                    head_x = max(0, int(head_x_min * cw))
                    head_y = max(0, int(head_y_min * ch))
                    head_w = min(cw - head_x, int((head_x_max - head_x_min) * cw))
                    head_h = min(ch - head_y, int((head_y_max - head_y_min) * ch))

                    # 머리 바운딩 박스 그리기
                    cv2.rectangle(crop_img, (head_x, head_y), (head_x + head_w, head_y + head_h), (255, 0, 0), 2)

                    # 전체 프레임에서의 좌표로 변환
                    global_head_x = x + head_x
                    global_head_y = y + head_y

                    # 사람 데이터 저장
                    people_data.append({
                        'head_bbox': [global_head_x, global_head_y, head_w, head_h],
                        'gaze_line': [global_perp_x1, global_perp_y1, global_perp_x2, global_perp_y2]
                    })

                frame[y:y+h, x:x+w] = crop_img

        # 시선 교차 확인
        for i, person in enumerate(people_data):
            for j, other_person in enumerate(people_data):
                if i != j:
                    if line_rectangle_intersection(person['gaze_line'], other_person['head_bbox']):
                        gaze_intersection_count += 1
                        cv2.line(frame, (person['gaze_line'][0], person['gaze_line'][1]),
                                 (person['gaze_line'][2], person['gaze_line'][3]), (0, 0, 255), 2)

        # 각도들의 코사인 유사도 평균 계산
        if len(angles) > 1:
            cosine_similarities = []
            for i in range(len(angles)):
                for j in range(i + 1, len(angles)):
                    cos_sim = calculate_cosine_similarity(math.radians(angles[i]), math.radians(angles[j]))
                    cosine_similarities.append(cos_sim)
            avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
            avg_cosine_similarity_degrees = math.degrees(math.acos(avg_cosine_similarity))
            cv2.putText(frame, f"Avg Cosine Angle: {avg_cosine_similarity_degrees:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 시선 교차 카운트 표시
        cv2.putText(frame, f"Gaze Intersections: {gaze_intersection_count}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 결과 프레임 표시
        cv2.imshow('Skeleton Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Total Gaze Intersections: {gaze_intersection_count}")