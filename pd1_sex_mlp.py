import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from transformers import ASTForAudioClassification
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import shutil

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 클래스 정의
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

# 데이터 준비
data_root = 'data/spectrogram_augmented'
folders = ['HC_M', 'HC_F', 'PD_M', 'PD_F']
file_paths, diag_labels, genders = [], [], []
for folder in folders:
    path = os.path.join(data_root, folder)
    for fn in sorted(os.listdir(path)):
        if fn.endswith('.npy'):
            file_paths.append(os.path.join(path, fn))
            diag_labels.append(0 if folder.startswith('HC') else 1)
            genders.append(0 if folder.endswith('_M') else 1)

# 성별 분할 함수
def split_by_gender(gender_value):
    indices = [i for i, g in enumerate(genders) if g == gender_value]
    fps_gender = [file_paths[i] for i in indices]
    labs_gender = [diag_labels[i] for i in indices]
    return fps_gender, labs_gender

# AST 모델 로딩 (중간 hidden_states 사용)
ast_model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    ignore_mismatched_sizes=True,
    output_hidden_states=True
).to(device)

# 마지막 3개 encoder 레이어만 Fine-tuning
for name, param in ast_model.named_parameters():
    param.requires_grad = any(layer in name for layer in ['encoder.layer.11', 'encoder.layer.10', 'encoder.layer.9'])

# MLPClassifier (3층)
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_labels=2):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, num_labels)
        )

    def forward(self, x):
        return self.fc_layers(x)

# 성별로 모델 학습
for gender_value, gender_name in [(0, 'male'), (1, 'female')]:
    fps_gender, labs_gender = split_by_gender(gender_value)
    train_f, val_f, train_l, val_l = train_test_split(
        fps_gender, labs_gender, test_size=0.2, stratify=labs_gender, random_state=42
    )

    train_loader = DataLoader(AudioDataset(train_f, train_l), batch_size=8, shuffle=True)
    val_loader = DataLoader(AudioDataset(val_f, val_l), batch_size=8)

    mlp_model = MLPClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(mlp_model.parameters()) + [p for p in ast_model.parameters() if p.requires_grad], lr=1e-5
    )

    writer = SummaryWriter(log_dir=f'runs/mlp_{gender_name}')
    patience, best_acc, best_loss, counter, step = 5, 0.0, float('inf'), 0, 0

    for epoch in range(100):
        ast_model.train(), mlp_model.train()
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
        ast_model.eval(), mlp_model.eval()
        preds, targets, probs, val_loss_total = [], [], [], 0
        with torch.no_grad():
            for spec, label in val_loader:
                spec, label = spec.to(device), label.to(device)
                hidden_states = ast_model(spec).hidden_states[-1].mean(dim=1)
                outputs = mlp_model(hidden_states)
                val_loss_total += criterion(outputs, label).item()
                probs += torch.softmax(outputs, dim=1)[:, 1].tolist()
                preds += outputs.argmax(dim=1).tolist()
                targets += label.tolist()

        val_loss = val_loss_total / len(val_loader)
        accuracy = accuracy_score(targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary')
        roc_auc = roc_auc_score(targets, probs)

        writer.add_scalar('Accuracy/val_epoch', accuracy, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        writer.add_scalar('F1_Score/val_epoch', f1, epoch)
        writer.add_scalar('Recall/val_epoch', recall, epoch)
        writer.add_scalar('ROC_AUC/val_epoch', roc_auc, epoch)

        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, ROC AUC: {roc_auc:.4f}')

        if accuracy > best_acc or (accuracy == best_acc and val_loss < best_loss):
            best_acc, best_loss, counter = accuracy, val_loss, 0
            torch.save({'ast': ast_model.state_dict(), 'mlp': mlp_model.state_dict()}, f'model_{gender_name}.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"EarlyStopping triggered for {gender_name}")
                break

    writer.close()

print("✅ 모든 과정 완료!")
