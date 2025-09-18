import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from transformers import ASTForAudioClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset 클래스 정의
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, fixed_time_len=1024):
        self.file_paths = file_paths
        self.labels = labels
        self.fixed_time_len = fixed_time_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        spec = np.squeeze(np.load(self.file_paths[idx]))  # (128,T)
        T = self.fixed_time_len
        if spec.shape[1] < T:
            pad = np.zeros((spec.shape[0], T - spec.shape[1]), dtype=spec.dtype)
            spec = np.concatenate([spec, pad], axis=1)
        else:
            spec = spec[:, :T]
        spec_tensor = torch.tensor(spec, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spec_tensor, label

# 데이터 경로 설정
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
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

train_ds = AudioDataset(train_f, train_l)
val_ds = AudioDataset(val_f, val_l)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# TransformerClassifier 정의
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=2, nhead=16, num_layers=2):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# AST 모델 로딩 (encoder 층 사용, 학습X)
ast_model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    ignore_mismatched_sizes=True,
    output_hidden_states=True
).to(device).eval()

for param in ast_model.parameters():
    param.requires_grad = False

classifier = TransformerClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

best_acc, best_loss = 0, float('inf')
patience, counter, step = 5, 0, 0

writer = SummaryWriter(log_dir='runs/transformer_transfer_all')

for epoch in range(50):
    classifier.train()
    total_loss = 0
    for spec, label in train_loader:
        spec, label = spec.to(device), label.to(device)

        with torch.no_grad():
            hidden_states = ast_model(spec).hidden_states[-1]

        outputs = classifier(hidden_states)
        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if step % 100 == 0:
            writer.add_scalar('Loss/train_step', loss.item(), step)
        step += 1

    avg_loss = total_loss / len(train_loader)

    classifier.eval()
    preds, targets, val_loss_total = [], [], 0
    with torch.no_grad():
        for spec, label in val_loader:
            spec, label = spec.to(device), label.to(device)
            hidden_states = ast_model(spec).hidden_states[-1]
            outputs = classifier(hidden_states)
            val_loss_total += criterion(outputs, label).item()
            preds.extend(outputs.argmax(1).cpu().numpy())
            targets.extend(label.cpu().numpy())

    val_loss = val_loss_total / len(val_loader)
    accuracy = accuracy_score(targets, preds)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)

    print(f'Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f}')

    if accuracy > best_acc or (accuracy == best_acc and val_loss < best_loss):
        best_acc, best_loss = accuracy, val_loss
        counter = 0
        torch.save(classifier.state_dict(), 'transformer_transfer_all.pth')
    else:
        counter += 1
        if counter >= patience:
            print("EarlyStopping triggered")
            break

writer.close()
print(f'Completed transformer transfer learning. Best Accuracy: {best_acc:.4f}')
