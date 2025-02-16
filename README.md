# MakeModel
훈련시킨 모델과 사용한 데이터셋
### ResNet50 분류 모델 결합 예상 흐름도
```plaintext
 이미지 입력(촬영) → ResNet50 분류 → [front / side / back]
                          |
                          |--- "back" → **처리 종료 (저장 X)**
                          |
                          |--- "front" or "side" → **데이터베이스 저장**
                                      |
                                      |--- YOLOv8 객체 검출 (nose & mouth)
                                      |
                                      |--- "nose or mouth 감지됨" → **정상 로그 기록**
                                      |
                                      |--- "nose and mouth 감지 안됨" → **경고 알림**
```
