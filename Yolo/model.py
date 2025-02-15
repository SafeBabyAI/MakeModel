import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# 1. Custom Dataset 클래스 정의
# -------------------------------
class CustomFaceDataset(Dataset):
    """
    이미지와 YOLO 형식의 어노테이션을 읽어오는 데이터셋 클래스.
    어노테이션은 각 이미지와 같은 이름의 .txt 파일로 저장되며,
    각 라인은: class_id center_x center_y width height (모두 정규화된 값) 형식입니다.
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        # 이미지 읽기 및 RGB 변환
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 0~1 범위의 실수형 텐서 변환
        image = image.astype(np.float32) / 255.0

        # 어노테이션 파일 경로 (이미지와 같은 이름의 .txt 파일)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # 각 라인은 "class_id center_x center_y width height"
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, bw, bh = map(float, parts)
                        boxes.append([int(cls), cx, cy, bw, bh])
        else:
            # 어노테이션이 없는 경우 빈 배열 반환
            boxes = []
        
        # 이미지 shape: [H, W, C] -> tensor: [C, H, W]
        image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        sample = {'image': image_tensor, 'boxes': torch.tensor(boxes, dtype=torch.float32)}
        return sample

# -------------------------------
# 2. 모델 구성: Backbone, Head, 전체 YOLO
# -------------------------------
class SimpleBackbone(nn.Module):
    """
    간단한 CNN Backbone 예제.
    실제 YOLO에서는 더 깊고 복잡한 구조(예: Darknet)를 사용합니다.
    """
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        return self.features(x)

class YOLOHead(nn.Module):
    """
    YOLO detection head.
    num_anchors: 한 grid cell 당 예측할 앵커 개수 (보통 3)
    각 앵커마다 (x, y, w, h, confidence) + num_classes 값을 예측합니다.
    """
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super(YOLOHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # 최종 출력 채널: num_anchors * (5 + num_classes)
        self.conv = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), kernel_size=1)
    
    def forward(self, x):
        out = self.conv(x)
        # out의 shape: [B, num_anchors*(5+num_classes), H, W]
        return out

class SimpleYOLO(nn.Module):
    """
    간단한 YOLO 모델 예제.
    Backbone과 Head로 구성되어 있으며, 실제 모델에서는 여러 스케일의 예측과 후처리(NMS 등)가 추가됩니다.
    """
    def __init__(self, num_classes):
        super(SimpleYOLO, self).__init__()
        self.backbone = SimpleBackbone()
        self.head = YOLOHead(in_channels=32, num_classes=num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

# -------------------------------
# 3. Non-Maximum Suppression (NMS) 자리 표시자 함수
# -------------------------------
def non_max_suppression(predictions, conf_thresh=0.5, iou_thresh=0.5):
    """
    NMS는 여러 겹치는 박스들 중 신뢰도가 높은 박스를 선택하는 후처리 기법입니다.
    여기서는 자리 표시자 함수로, 실제 사용 시 IoU 계산 등을 포함한 구현이 필요합니다.
    """
    # 실제 구현 시, 박스 좌표, confidence score, 클래스 정보 등을 추출하여 NMS 적용
    return predictions

# -------------------------------
# 4. 학습 루프 함수
# -------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        images = batch['image'].to(device)
        # 실제 YOLO loss 계산은 예측과 ground truth 간의 복합 손실(MSE, CrossEntropy 등)로 구성됩니다.
        # 여기서는 단순화를 위해 출력 자체를 대상으로 dummy loss를 계산합니다.
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, outputs)  # 더 적절한 손실 함수를 구현 필요
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# -------------------------------
# 5. 메인 함수: 데이터 로딩, 학습 및 모델 저장
# -------------------------------
def main():
    # 데이터셋 경로 설정 (실제 데이터 경로에 맞게 수정)
    train_image_dir = './data/images/train'
    train_label_dir = './data/labels/train'
    
    # 하이퍼파라미터 설정
    batch_size = 4
    epochs = 10
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset 및 DataLoader 생성
    train_dataset = CustomFaceDataset(train_image_dir, train_label_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 모델 초기화 (클래스: front, side, back → num_classes = 3)
    num_classes = 3
    model = SimpleYOLO(num_classes=num_classes).to(device)
    
    # 손실 함수와 옵티마이저 정의 (여기서는 예시로 MSELoss 사용; 실제로는 YOLO 손실 함수를 구현해야 함)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 루프 실행
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")
    
    # 학습 완료 후 모델 저장
    torch.save(model.state_dict(), 'simple_yolo_face.pth')
    print("모델이 저장되었습니다: simple_yolo_face.pth")

if __name__ == "__main__":
    main()