#!/usr/bin/env python
import cv2
import numpy as np
from ultralytics import YOLO

# 클래스 ID와 이름 매핑 (데이터셋 어노테이션에 따라 아래 순서를 맞추세요)
CLASS_NAMES = {
    0: "baby_sleeping",  # 아기가 누워 자고 있는 경우
    1: "baby_crying",    # 아기가 울고 있는 경우
    2: "bedding",        # 침구류
    3: "toy"             # 장난감
}

def iou(boxA, boxB):
    """두 박스의 IoU(Intersection over Union) 계산"""
    ix1 = max(boxA[0], boxB[0])
    iy1 = max(boxA[1], boxB[1])
    ix2 = min(boxA[2], boxB[2])
    iy2 = min(boxA[3], boxB[3])
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)

def check_breathing_area(baby_box, obstruction_boxes, threshold=0.2):
    """
    아기가 누워 있을 때, 호흡기관이 (대략적으로) 아기 박스의 상단 1/3 영역에 해당한다고 가정.
    해당 영역과 침구류 또는 장난감이 일정 IoU 이상 겹치면 가려졌다고 판단.
    """
    x1, y1, x2, y2 = baby_box
    baby_height = y2 - y1
    # 호흡 영역: 아기 박스의 상단 1/3
    breathing_area = [x1, y1, x2, y1 + baby_height / 3]
    
    for ob in obstruction_boxes:
        if iou(breathing_area, ob) > threshold:
            return False  # 호흡 부위가 가려짐
    return True

def process_image(image_path, model):
    """
    주어진 이미지에 대해 YOLOv8 모델로 추론을 수행하고,
    아기가 누워 자고 있는지, 호흡기관이 가려졌는지, 아기가 울고 있는지 판단.
    """
    results = model(image_path, conf=0.25)
    result = results[0]  # 첫 번째 결과 사용

    # 탐지된 객체들을 분류별로 분리
    baby_sleeping_boxes = []
    baby_crying_boxes = []
    obstruction_boxes = []  # bedding 및 toy

    for box in result.boxes:
        cls_id = int(box.cls.cpu().numpy())
        box_coords = box.xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
        class_name = CLASS_NAMES.get(cls_id, "unknown")
        if class_name == "baby_sleeping":
            baby_sleeping_boxes.append(box_coords)
        elif class_name == "baby_crying":
            baby_crying_boxes.append(box_coords)
        elif class_name in ["bedding", "toy"]:
            obstruction_boxes.append(box_coords)

    # 조건 판단 결과 저장
    status = {}
    if baby_sleeping_boxes:
        status['sleeping'] = True
        # 여러 탐지 결과가 있다면, 하나라도 호흡 부위가 가려졌는지 확인
        unobstructed = True
        for baby_box in baby_sleeping_boxes:
            if not check_breathing_area(baby_box, obstruction_boxes):
                unobstructed = False
                break
        status['breathing_unobstructed'] = unobstructed
    else:
        status['sleeping'] = False
        status['breathing_unobstructed'] = None

    if baby_crying_boxes:
        status['crying'] = True
    else:
        status['crying'] = False

    # 모델이 그린 주석이 달린 이미지를 반환 (시각화 확인용)
    annotated_img = result.plot()  # numpy array, BGR 형식
    return status, annotated_img

def main():
    # 1. YOLOv8 모델 로드 (학습된 모델의 경로로 수정하세요)
    model_path = "best.pt"  # 학습된 모델 파일 경로
    model = YOLO(model_path)

    # 2. 테스트 이미지 경로 (적절한 경로로 수정)
    image_path = "test_image.jpg"
    
    # 3. 추론 및 조건 판단 수행
    status, annotated_img = process_image(image_path, model)

    # 4. 결과 출력
    print("=== 탐지 결과 ===")
    if status.get('sleeping'):
        print("아기가 누워 자고 있습니다.")
        if status.get('breathing_unobstructed'):
            print("아기의 호흡기관이 침구류 및 장난감으로부터 가려있지 않습니다.")
        else:
            print("경고: 아기의 호흡기관이 침구류 또는 장난감에 의해 가려졌습니다!")
    else:
        print("아기가 누워 자고 있지 않습니다.")

    if status.get('crying'):
        print("아이가 울고 있습니다.")
    else:
        print("아이가 울고 있지 않습니다.")

    # 5. 주석 달린 이미지 저장 (BGR 형식이므로 cv2.imwrite 사용)
    output_path = "annotated_output.jpg"
    cv2.imwrite(output_path, annotated_img)
    print(f"주석이 달린 이미지가 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    main()
