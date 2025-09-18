import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from transformers import ASTForAudioClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    roc_curve
)
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# device 정의
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 클래스 정의 (이전 코드와 동일)
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
            pad = np.zeros((spec.shape[0], T - spec.shape[1]), dtype=spec.dtype)
            spec = np.concatenate([spec, pad], axis=1)
        else:
            spec = spec[:, :T]
        spec_tensor = torch.tensor(spec, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spec_tensor, label

# 데이터 경로 설정 및 train/val split (이전 코드와 동일)
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

# 데이터 로더 정의 (반드시 포함되어야 하는 부분)
train_ds = AudioDataset(train_f, train_l)
val_ds = AudioDataset(val_f, val_l)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)


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

# AST 모델 로드 (encoder 층 사용, 이전 코드와 동일하게 정의)
ast_model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    ignore_mismatched_sizes=True,
    output_hidden_states=True
).to(device)

# classifier 정의 및 이전 모델 가중치 로드
classifier = TransformerClassifier().to(device)
classifier.load_state_dict(torch.load('transformer_transfer_all.pth'))

# ========= 이전에 저장된 classifier 로드 =========
classifier.load_state_dict(torch.load('transformer_transfer_all.pth'))

# ========= AST 모델 전체를 풀어서 학습 가능 상태로 변경 =========
for param in ast_model.parameters():
    param.requires_grad = True

# ========= Optimizer 재정의 (AST + classifier 함께 학습) =========
optimizer = torch.optim.Adam(
    list(ast_model.parameters()) + list(classifier.parameters()),
    lr=1e-5
)

criterion = nn.CrossEntropyLoss()

# Tensorboard Writer 새로 정의 (구분을 위해)
writer = SummaryWriter(log_dir='runs/full_finetune_after_transfer')

best_f1, patience, counter, step = 0, 5, 0, 0


for epoch in range(50):
    ast_model.train()
    classifier.train()
    total_loss = 0

    for spec, label in train_loader:
        spec, label = spec.to(device), label.to(device)

        hidden_states = ast_model(spec).hidden_states[-1]
        outputs = classifier(hidden_states)
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if step % 100 == 0:
            writer.add_scalar('Loss/train_step_full_ft', loss.item(), step)
        step += 1

    avg_loss = total_loss / len(train_loader)

    # 검증 단계
    ast_model.eval()
    classifier.eval()
    preds, targets, probs, val_loss_total = [], [], [], 0

    with torch.no_grad():
        for spec, label in val_loader:
            spec, label = spec.to(device), label.to(device)
            hidden_states = ast_model(spec).hidden_states[-1]
            outputs = classifier(hidden_states)

            val_loss_total += criterion(outputs, label).item()

            prob = torch.softmax(outputs, dim=1)[:, 1]
            preds.extend(outputs.argmax(1).cpu().numpy())
            probs.extend(prob.cpu().numpy())
            targets.extend(label.cpu().numpy())

    val_loss = val_loss_total / len(val_loader)
    accuracy = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary')
    roc_auc = roc_auc_score(targets, probs)

    # TensorBoard에 기록
    writer.add_scalar('Loss/train_full_ft', avg_loss, epoch)
    writer.add_scalar('Loss/val_full_ft', val_loss, epoch)
    writer.add_scalar('Accuracy/val_full_ft', accuracy, epoch)
    writer.add_scalar('Precision/val_full_ft', precision, epoch)
    writer.add_scalar('Recall/val_full_ft', recall, epoch)
    writer.add_scalar('F1_Score/val_full_ft', f1, epoch)
    writer.add_scalar('ROC_AUC/val_full_ft', roc_auc, epoch)

    # ROC Curve 기록
    fpr, tpr, _ = roc_curve(targets, probs)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')

    writer.add_figure('ROC_Curve_full_ft', fig, epoch)
    plt.close(fig)

    print(f'[Full Finetune] Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROC AUC: {roc_auc:.4f}')

    # 성능 기준으로 모델 저장 (EarlyStopping)
    if f1 > best_f1:
        best_f1 = f1
        counter = 0
        torch.save({
            'ast_model_state_dict': ast_model.state_dict(),
            'classifier_state_dict': classifier.state_dict()
        }, 'full_finetuned_ast_transformer.pth')
    else:
        counter += 1
        if counter >= patience:
            print("[Full Finetune] EarlyStopping triggered")
            break

writer.close()
print(f'Completed Full AST finetuning. Best F1 Score: {best_f1:.4f}')
