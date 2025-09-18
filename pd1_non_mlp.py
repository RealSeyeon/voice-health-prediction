import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import ASTForAudioClassification
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (1) Dataset 정의
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, fixed_time_len=1024):
        self.file_paths = file_paths
        self.labels = labels
        self.fixed_time_len = fixed_time_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        spec = np.squeeze(np.load(self.file_paths[idx]))
        T = self.fixed_time_len
        if spec.shape[1] < T:
            pad = np.zeros((spec.shape[0], T - spec.shape[1]))
            spec = np.concatenate([spec, pad], axis=1)
        else:
            spec = spec[:, :T]

        spec_tensor = torch.tensor(spec, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spec_tensor, label

# (2) 데이터 로딩
data_root = 'data/spectrogram_augmented'
folders = ['HC_M', 'HC_F', 'PD_M', 'PD_F']

file_paths, labels = [], []
for folder in folders:
    path = os.path.join(data_root, folder)
    for fn in sorted(os.listdir(path)):
        if fn.endswith('.npy'):
            file_paths.append(os.path.join(path, fn))
            labels.append(0 if folder.startswith('HC') else 1)

train_f, val_f, train_l, val_l = train_test_split(
    file_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

train_ds = AudioDataset(train_f, train_l)
val_ds = AudioDataset(val_f, val_l)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# (3) AST 모델 로드 (중간 encoder 출력 사용, 일부 층 학습)
ast_model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    ignore_mismatched_sizes=True,
    output_hidden_states=True
).to(device)

# 일부 층만 Fine-tuning
for name, param in ast_model.named_parameters():
    param.requires_grad = any(layer in name for layer in ['encoder.layer.11', 'encoder.layer.10', 'encoder.layer.9'])

# (4) MLP 모델 정의 (hidden layer 3층)
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, num_classes)
        )

    def forward(self, x):
        return self.fc_layers(x)

mlp_model = MLPClassifier().to(device)

# (5) Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(mlp_model.parameters()) + [p for p in ast_model.parameters() if p.requires_grad], lr=1e-5
)

# TensorBoard 설정 (100 step마다 기록)
writer = SummaryWriter(log_dir='runs/mlp_transfer')

# EarlyStopping 설정 (accuracy & val_loss)
patience = 5
best_acc, best_loss = 0.0, float('inf')
counter, epoch, step = 0, 0, 0

# (6) 학습 루프
for epoch in range(100):
    ast_model.train()
    mlp_model.train()
    total_loss = 0
    for spec, label in train_loader:
        spec, label = spec.to(device), label.to(device)

        hidden_states = ast_model(spec).hidden_states[-1].mean(dim=1)
        outputs = mlp_model(hidden_states)
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % 100 == 0:
            writer.add_scalar('Loss/train_step', loss.item(), step)
        step += 1

    avg_loss = total_loss / len(train_loader)

    # 검증
    ast_model.eval()
    mlp_model.eval()
    preds, targets, val_loss_total = [], [], 0
    with torch.no_grad():
        for spec, label in val_loader:
            spec, label = spec.to(device), label.to(device)
            hidden_states = ast_model(spec).hidden_states[-1].mean(dim=1)
            outputs = mlp_model(hidden_states)
            val_loss_total += criterion(outputs, label).item()
            preds += outputs.argmax(dim=1).tolist()
            targets += label.tolist()

    val_loss = val_loss_total / len(val_loader)
    accuracy = accuracy_score(targets, preds)

    writer.add_scalar('Accuracy/val_epoch', accuracy, epoch)
    writer.add_scalar('Loss/val_epoch', val_loss, epoch)

    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

    # EarlyStopping (accuracy & loss)
    if accuracy > best_acc or (accuracy == best_acc and val_loss < best_loss):
        best_acc, best_loss, counter = accuracy, val_loss, 0
        torch.save({'ast': ast_model.state_dict(), 'mlp': mlp_model.state_dict()}, 'mlp_classifier.pth')
    else:
        counter += 1
        if counter >= patience:
            print("🛑 EarlyStopping triggered")
            break

writer.close()