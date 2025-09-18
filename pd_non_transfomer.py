import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from transformers import ASTForAudioClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

# 0) device 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 데이터셋 클래스
class ParkinsonDataset(Dataset):
    def __init__(self, data_root: str, fixed_time_len: int = 1024):
        self.files = []
        self.labels = []
        self.fixed_time_len = fixed_time_len

        for folder, label in zip(['HC_M', 'HC_F', 'PD_M', 'PD_F'], [0, 0, 1, 1]):
            folder_path = os.path.join(data_root, folder)
            for fn in sorted(os.listdir(folder_path)):
                if fn.endswith('.npy'):
                    self.files.append(os.path.join(folder_path, fn))
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        spec = np.squeeze(np.load(self.files[idx]))   # spec: (128, T)
        T = self.fixed_time_len
        if spec.shape[1] < T:
            pad = np.zeros((spec.shape[0], T - spec.shape[1]), dtype=spec.dtype)
            spec = np.concatenate((spec, pad), axis=1)
        else:
            spec = spec[:, :T]

        spec_tensor = torch.from_numpy(spec.astype(np.float32))  # ⚠️ unsqueeze 제거!
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spec_tensor, label

# 2) 데이터 준비
data_root = 'data/spectrogram_augmented'
dataset = ParkinsonDataset(data_root)
train_idx, val_idx = train_test_split(
    np.arange(len(dataset)),
    test_size=0.2,
    random_state=42,
    stratify=dataset.labels
)
train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx),
                          batch_size=16, shuffle=True)
val_loader   = DataLoader(torch.utils.data.Subset(dataset, val_idx),
                          batch_size=16, shuffle=False)

# 3) AST 모델 로드 + GPU
ast_model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    ignore_mismatched_sizes=True
).to(device)
ast_model.eval()

# 4) TransformerClassifier 정의
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=527, num_classes=2, nhead=17, num_layers=1, dim_feedforward=256):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True  # 🚩 성능 향상 옵션 추가 (선택적)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# 5) Classifier + 손실, 최적화 + TensorBoard
classifier = TransformerClassifier(input_dim=527, nhead=17).to(device)
criterion  = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(classifier.parameters(), lr=1e-4)
writer     = SummaryWriter(log_dir='runs/transformer_transfer')

# 6) EarlyStopping 파라미터
patience = 3
best_acc = 0.0
counter  = 0
epoch    = 0

# 7) 학습 루프
while True:
    # --- Train ---
    classifier.train()
    total_loss = 0.0
    for spec, label in train_loader:
        spec = spec.to(device)     # [B, 128, T] 형태로 정상 동작
        label = label.to(device)

        with torch.no_grad():
            logits = ast_model(spec).logits  # [B, 527]
        
        logits = logits.unsqueeze(1)  # Transformer 입력 형태 맞춤
        outputs = classifier(logits)
        loss    = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)

    # --- Validation ---
    classifier.eval()
    preds, targets = [], []
    with torch.no_grad():
        for spec, label in val_loader:
            spec  = spec.to(device)
            label = label.to(device)
            logits = ast_model(spec).logits.unsqueeze(1)
            outputs = classifier(logits)
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            targets.extend(label.cpu().numpy())

    accuracy = accuracy_score(targets, preds)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {accuracy:.4f}')

    # --- EarlyStopping 체크 ---
    if accuracy > best_acc:
        best_acc = accuracy
        counter = 0
        torch.save(classifier.state_dict(), 'transformer_classifier.pth')
        print("🎉 모델 저장 완료 (최고 성능)")
    else:
        counter += 1
        print(f"EarlyStopping Counter: {counter}/{patience}")
        if counter >= patience:
            print("🛑 EarlyStopping triggered")
            break

    epoch += 1

writer.close()
