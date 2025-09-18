import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_auc_score
from transformers import ASTForAudioClassification

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

# 데이터 로딩
data_root = 'data/spectrogram_augmented'
folders = ['HC_M', 'HC_F', 'PD_M', 'PD_F']
file_paths, labels = [], []

for folder in folders:
    path = os.path.join(data_root, folder)
    for fn in sorted(os.listdir(path)):
        if fn.endswith('.npy'):
            file_paths.append(os.path.join(path, fn))
            labels.append(0 if folder.startswith('HC') else 1)

_, val_f, _, val_l = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

val_ds = AudioDataset(val_f, val_l)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

# 모델 정의
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=2, nhead=16, num_layers=2, max_len=2000):
        super().__init__()
        self.pos_encoder = self.positional_encoding(max_len, input_dim).to(device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, num_classes)

    def positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# AST 모델 로드
ast_model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    ignore_mismatched_sizes=True,
    output_hidden_states=True
).to(device).eval()

classifier = TransformerClassifier().to(device)
classifier.load_state_dict(torch.load('transformer_transfer_PE_all.pth', map_location=device))
classifier.eval()

# 평가 수행
preds, targets, probs = [], [], []

with torch.no_grad():
    for spec, label in val_loader:
        spec, label = spec.to(device), label.to(device)
        hidden_states = ast_model(spec).hidden_states[-1]
        outputs = classifier(hidden_states)
        preds.extend(outputs.argmax(dim=1).cpu().numpy())
        targets.extend(label.cpu().numpy())
        probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

accuracy = accuracy_score(targets, preds)
precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary')
roc_auc = roc_auc_score(targets, probs)

print("Accuracy:", f"{accuracy:.4f}")
print("Precision:", f"{precision:.4f}")
print("Recall:", f"{recall:.4f}")
print("F1-score:", f"{f1:.4f}")
print("ROC AUC:", f"{roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(targets, preds, target_names=['HC', 'PD'], digits=4))

print("Confusion Matrix:")
print(confusion_matrix(targets, preds))
