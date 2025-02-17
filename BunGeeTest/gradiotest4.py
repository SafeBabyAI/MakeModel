"""
커스텀 비전으로 0.5초씩 이미지 저장해서 예측하기

실습 때 사용한 번지 찾기 예측 프로그램을 변형하였습니다.
1. 그라지오 창에 동영상을 올려두면 
2. img 폴더를 생성하고 
3. 지정한 시간에 이미지를 저장합니다
4. 저장한 이미지를 애저 API로 보내어 
5. 예측 결과를 받아옵니다.
6. 그리고 예측 결과의 좌표를 오른쪽에 이미지로 보여줍니다.

중간에 
output_dir = r"\BunGeeTest\img"
은 환경에 맞춰 변경해야할 수도 있습니다.
"""

# Azure의 Custom Vision 라이브러리를 추가. 예측을 위하여 prediction을 포함
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
# OpenAPI 스펙에 맞춰서 Authentication을 처리할 수 있도록 해주는 코드
from msrest.authentication import ApiKeyCredentials
# Python Image 라이브러리로 이미지 그리기
from PIL import Image, ImageDraw, ImageFont
# Python Numpy (수학 및 과학 연산 패키지) 포함
import numpy as np
# 파일 처리 작업을 위해 os 라이브러리 포함
import os
import cv2
import gradio as gr
import time

# 사용자가 만든 AI 모델의 예측 기능을 사용하기 위한 endpoint 지정
prediction_endpoint = "https://6b001cv20250210-prediction.cognitiveservices.azure.com"
# 애저 api key 값
prediction_key = "9TPDMpGrwBQHI695jWXGc6hVX6dgG0tzMXT2vtiHXgM3DP3Jun0AJQQJ99BBACYeBjFXJ3w3AAAIACOGRgVe"
# 애저 project id
project_id = "5b5abbee-837b-4cb3-926c-5c147ce98e20"
# 모델 이름
model_name = "Iteration1_bunny"

# 앞에서 지정한 API KEY를 써서 커스텀 비전 모델을 사용할 클라이언트를 인증
credentials = ApiKeyCredentials(in_headers={"Prediction-key":prediction_key})
# endpoint를 써서 클라이언트 등록
predictor = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)

def azuer_test(image_file, cnt):
    image = Image.open(image_file)
    # Numpy에서 이미지의 shape을 높이, 폭, 채널 읽기
    h, w, ch = np.array(image).shape

    # 테스트 이미지를 열고 모델에 적용해서 결과를 저장
    with open(image_file, mode='rb') as image_data:
        results = predictor.detect_image(project_id, model_name, image_data)
    
    return results, h, w

def save_frame(frame):
    # 현재 시간을 이용해 고유한 파일명 생성
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # 저장할 디렉토리 
    # 다운로드 환경에 맞춰서 디렉토리 변경 필요합니다 #
    output_dir = r"\BunGeeTest\img"  
    ###############################################
    
    # 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 저장 경로+파일명 만들기
    output_path = os.path.join(output_dir, f"frame_{timestamp}.jpg")
    # 위에서 만든 경로에 frame 저장하기
    cv2.imwrite(output_path, frame)

    return output_path

def detect_faces_from_video(video_path):
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)  
    # 출력 확인용 cnt 
    cnt = 0

    while cap.isOpened():
        cnt += 1
        # 비디오 캡쳐 읽기
        ret, frame = cap.read()
        # 초당 프레임 정보 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not ret:
            break  # 더 이상 프레임이 없으면 종료

        # if(cnt % int(fps) == 0): # 1초에 한 프레임
        if(cnt % int(fps/2) == 0): # 0.5초에 한 프레임
            image_path = save_frame(frame)
            results, h, w  = azuer_test(image_path, cnt)
            # 개체 인식된 모든 결과에 대해서
            for prediction in results.predictions:
                # 확률이 50%이 이상인 경우 bounding box 값을 읽음
                if(prediction.probability * 100) > 50:
                    x1 = int(prediction.bounding_box.left * w)
                    y1 = int(prediction.bounding_box.top * h)
                    x2 = int((prediction.bounding_box.left + prediction.bounding_box.width) * w)
                    y2 = int((prediction.bounding_box.top + prediction.bounding_box.height) * h)

                    # 사각형 그리기 (cv2.rectangle()은 정수 좌표를 필요로 함)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # # RGB 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame  # 프레임을 하나씩 반환

    cap.release()

# Gradio Blocks 인터페이스 구성
with gr.Blocks() as demo:
    with gr.Row():
        video_input = gr.Video(label="Upload a Video")  # 비디오 파일 입력
        output_video = gr.Image(label="Processed Video Frames")  # 결과 프레임 출력

    video_input.change(fn=detect_faces_from_video, inputs=video_input, outputs=output_video)

demo.launch(share=True)