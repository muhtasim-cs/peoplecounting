import cv2
import numpy as np
import cvzone
import math
from ultralytics import YOLO
from sort import Sort


VIDEO_PATH = "......"
YOLO_MODEL_PATH = "....."
MASK_PATH = "...."
GRAPHICS_PATH = "....."


classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


LIMITS_UP = [103, 161, 296, 161]
LIMITS_DOWN = [527, 489, 735, 489]


cap = cv2.VideoCapture(VIDEO_PATH)


model = YOLO(YOLO_MODEL_PATH)


mask = cv2.imread(MASK_PATH)
imgGraphics = cv2.imread(GRAPHICS_PATH, cv2.IMREAD_UNCHANGED)


tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


totalCountUp = []
totalCountDown = []


def process_frame(img):
    imgRegion = cv2.bitwise_and(img, mask)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))

    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

    resultsTracker = tracker.update(detections)

    return resultsTracker


def draw_tracking_lines(img):
    cv2.line(img, (LIMITS_UP[0], LIMITS_UP[1]), (LIMITS_UP[2], LIMITS_UP[3]), (0, 0, 255), 5)
    cv2.line(img, (LIMITS_DOWN[0], LIMITS_DOWN[1]), (LIMITS_DOWN[2], LIMITS_DOWN[3]), (0, 0, 255), 5)


def update_counts_and_draw_results(img, resultsTracker):
    global totalCountUp, totalCountDown

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if LIMITS_UP[0] < cx < LIMITS_UP[2] and LIMITS_UP[1] - 15 < cy < LIMITS_UP[1] + 15:
            if id not in totalCountUp:
                totalCountUp.append(id)
                cv2.line(img, (LIMITS_UP[0], LIMITS_UP[1]), (LIMITS_UP[2], LIMITS_UP[3]), (0, 255, 0), 5)

        if LIMITS_DOWN[0] < cx < LIMITS_DOWN[2] and LIMITS_DOWN[1] - 15 < cy < LIMITS_DOWN[1] + 15:
            if id not in totalCountDown:
                totalCountDown.append(id)
                cv2.line(img, (LIMITS_DOWN[0], LIMITS_DOWN[1]), (LIMITS_DOWN[2], LIMITS_DOWN[3]), (0, 255, 0), 5)

    # Display Counts
    cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)


def main_loop():
    while True:
        success, img = cap.read()
        if not success:
            break

        resultsTracker = process_frame(img)
        draw_tracking_lines(img)
        update_counts_and_draw_results(img, resultsTracker)

        cv2.imshow("Image", img)
        # cv2.imshow("ImageRegion", imgRegion)
        cv2.waitKey(1)


if __name__ == "__main__":
    main_loop()
    cap.release()
    cv2.destroyAllWindows()

