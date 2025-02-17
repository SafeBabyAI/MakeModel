# MakeModel
훈련시킨 모델과 사용한 데이터셋
### ResNet50 분류 모델 결합 예상 흐름도
```plaintext
 이미지 입력(촬영) → ResNet50 분류 -> [front / side / back] 
                          |
                          |--- "back" -> **경고 알림** -> **처리 종료 (저장 X)** 
                          |
                          |--- "front" or "side" -> **데이터베이스 저장** 
                                     |
                                      |--- YOLOv8 객체 검출 (nose & mouth)
                                      |
                                      |--- "nose or mouth 감지됨" -> **정상 로그 기록** 
                                      |
                                      |--- "nose and mouth 감지 안됨" -> **경고 알림**
``` 
#### YOLOv8 단독사용을 하지 않는 이유

- **리소스 절약**
    -> 분류 성능이 뛰어나고 리소스 효율이 좋은 ResNet50으로 아기의 방향을 먼저 분류

- **YOLOv8의 탐지 성능 향상**
    -> 후면 사진 자체를 배제함으로써 후면 이미지에서 객체를 탐지하는 오류를 방지</br>
    예) 옷이나 침구의 프린팅, 인형 및 장난감 등이 인식되는 등 

- **추가적인 성능 개선 가능성**
    - 현 모델에는 YOLOv8에도 다양한 방향의 이미지를 학습시켰으나 모델이 안정화된 후 YOLOv8에는 앞모습만 학습시켜 ResNet50과 역할을 완전히 분리함으로써 효율 개선을 제고할 수 있음

- **연구 자료의 존재**
    - ResNet50 분류 델과 YOLOv8 객체 인식 결합 모델의 성능이 높게 측정된 연구자료를 근거로 높은 성능을 기대
    > 고추작물의정밀질병진단을위한딥러닝모델통합연구YOLOv8ResNet50FasterRCNN의성능분석, 2024
