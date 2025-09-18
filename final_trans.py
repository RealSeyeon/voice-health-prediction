import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from transformers import ASTForAudioClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# 데이터 로딩
folders = ['HC_M', 'HC_F', 'PD_M', 'PD_F']
data_root = 'data/spectrogram_augmented'

file_paths, labels, genders = [], [], []
for folder in folders:
    path = os.path.join(data_root, folder)
    for fn in sorted(os.listdir(path)):
        if fn.endswith('.npy'):
            file_paths.append(os.path.join(path, fn))
            labels.append(0 if folder.startswith('HC') else 1)
            genders.append('M' if folder.endswith('_M') else 'F')

# 모델 정의
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

ast_model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    ignore_mismatched_sizes=True,
    output_hidden_states=True
).to(device).eval()

for param in ast_model.parameters():
    param.requires_grad = False

classifier = TransformerClassifier().to(device)
classifier.load_state_dict(torch.load('transformer_transfer_all.pth', map_location=device))
classifier.eval()
criterion = nn.CrossEntropyLoss()

# 성별로 평가
for gender in ['M', 'F']:
    gender_file_paths = [fp for fp, g in zip(file_paths, genders) if g == gender]
    gender_labels = [lb for lb, g in zip(labels, genders) if g == gender]

    _, val_f, _, val_l = train_test_split(
        gender_file_paths, gender_labels, test_size=0.2, random_state=42, stratify=gender_labels
    )

    val_ds = AudioDataset(val_f, val_l)
    val_loader = DataLoader(val_ds, batch_size=8)

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

    print(f'[Gender: {"Male" if gender == "M" else "Female"}] Val Loss: {val_loss:.4f} | Val Accuracy: {accuracy:.4f}')
