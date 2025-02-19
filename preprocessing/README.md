## OpenCV 전처리 코드

아기 이미지 데이터 전처리 코드입니다.

## 전처리 내용

- 이미지 선별
- 리사이징
- 보간법 사용 이미지 확대 및 화질개선
  
# opencv 라이브러리 이용

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 이미지 폴더 경로 설정

folder_path = 'C:/Users/USER/Desktop/babyai/frontraw_03'  

# Super Resolution (2배 확대용 ESPCN 모델)
sr = cv2.dnn_superres.DnnSuperResImpl_create()
model_path = 'C:/Users/USER/Downloads/ESPCN_x2.pb'  # ESPCN_x2.pb 모델 파일 경로
sr.readModel(model_path)
sr.setModel("espcn", 2)  # 2배 확대

    # 폴더에서 모든 이미지 파일 읽기
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 결과 저장 폴더 설정
save_dir = 'C:/Users/USER/Desktop/babyai/processed_front_03'
os.makedirs(save_dir, exist_ok=True)

    # 이미지 처리 및 저장
file_counter = 1  # 파일 번호 초기화

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    
    # 이미지 로드
    img = cv2.imread(image_path)

    # 이미지 리사이즈 (224x224로 크기 변경)
    img_resized = cv2.resize(img, (224, 224))
    
    # 이미지 밝게/어둡게 처리
    img2 = cv2.convertScaleAbs(img_resized, alpha=1, beta=30)  # 밝기 30만큼 밝게 처리
    img3 = cv2.convertScaleAbs(img_resized, alpha=1, beta=-30)  # 밝기 30만큼 어둡게 처리

    # 포화 연산: 이미지 값이 0~255 범위를 넘지 않도록 처리
    img2 = np.clip(img2, 0, 255)  # 밝게 처리된 이미지를 포화 연산
    img3 = np.clip(img3, 0, 255)  # 어둡게 처리된 이미지를 포화 연산

    # Super Resolution 적용 (ESPCN 모델)
    sr_img = sr.upsample(img_resized)  # Super Resolution 모델로 이미지 확대
    sr_img2 = sr.upsample(img2)  # 밝게 처리된 이미지에도 Super Resolution 적용
    sr_img3 = sr.upsample(img3)  # 어둡게 처리된 이미지에도 Super Resolution 적용

    # 좌우 반전 및 180도 회전 처리 (원본 이미지에만 적용)
    img4 = cv2.flip(sr_img, 1)  # 좌우 반전
    img5 = cv2.rotate(sr_img, cv2.ROTATE_180)  # 180도 회전

    # Super Resolution 후 강제 리사이즈 (224x224로)
    sr_img_resized = cv2.resize(sr_img, (224, 224))  # Super Resolution 후 224x224로 리사이즈
    img4_resized = cv2.resize(img4, (224, 224))  # 좌우 반전 후 224x224로 리사이즈
    img5_resized = cv2.resize(img5, (224, 224))  # 180도 회전 후 224x224로 리사이즈

    # 밝게 처리된 이미지와 어두운 처리된 이미지도 224x224로 리사이즈
    sr_img2_resized = cv2.resize(sr_img2, (224, 224))  # 밝게 처리된 이미지 리사이즈
    sr_img3_resized = cv2.resize(sr_img3, (224, 224))  # 어둡게 처리된 이미지 리사이즈

    # 결과 화면에 표시
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(sr_img_resized, cv2.COLOR_BGR2RGB))  # Super Resolution 결과 이미지
    plt.title(f'SR Image: {image_file}')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(img4_resized, cv2.COLOR_BGR2RGB))  # Super Resolution 좌우 반전 이미지
    plt.title('SR Flipped Image')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(img5_resized, cv2.COLOR_BGR2RGB))  # Super Resolution 180도 회전 이미지
    plt.title('SR Rotated Image')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))  # 리사이즈된 원본 이미지
    plt.title(f'Original Image: {image_file}')
    plt.axis('off')

    plt.show()

    # 파일 이름 형식
    file_name_base = f"f{file_counter:04d}"  # b0001, b0002와 같은 형식으로 저장

    # 저장: Super Resolution 이미지 저장 (원본, 밝기 조정, 어두운 처리된 이미지 각각)
    sr_save_path = os.path.join(save_dir, f"{file_name_base}_sr.jpeg")
    cv2.imwrite(sr_save_path, sr_img_resized)

    # 좌우 반전 이미지 저장
    
    # 180도 회전된 이미지 저장
    sr_fli_save_path = os.path.join(save_dir, f"{file_name_base}_sr_fli.jpeg")
    cv2.imwrite(sr_fli_save_path, img4_resized)  # 좌우 반전 이미지 저장

    sr_ro_save_path = os.path.join(save_dir, f"{file_name_base}_sr_ro.jpeg")
    cv2.imwrite(sr_ro_save_path, img5_resized)  # 180도 회전 이미지 저장

    # 밝게 처리된 이미지 저장
    sr_bright_save_path = os.path.join(save_dir, f"{file_name_base}_bright_sr.jpeg")
    cv2.imwrite(sr_bright_save_path, sr_img2_resized)

    # 어둡게 처리된 이미지 저장
    sr_dark_save_path = os.path.join(save_dir, f"{file_name_base}_dark_sr.jpeg")
    cv2.imwrite(sr_dark_save_path, sr_img3_resized)


    # 파일 번호 증가
    file_counter += 1

print("모든 이미지 처리 완료!")
