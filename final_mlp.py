import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import ASTForAudioClassification

# Device 설정
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

# (2) 데이터 로딩 (성별 포함)
data_root = 'data/spectrogram_augmented'
folders = ['HC_M', 'HC_F', 'PD_M', 'PD_F']

file_paths, labels, genders = [], [], []
for folder in folders:
    path = os.path.join(data_root, folder)
    for fn in sorted(os.listdir(path)):
        if fn.endswith('.npy'):
            file_paths.append(os.path.join(path, fn))
            labels.append(0 if folder.startswith('HC') else 1)
            genders.append('M' if folder.endswith('_M') else 'F')

# 모델 로드 및 정의
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

ast_model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    ignore_mismatched_sizes=True,
    output_hidden_states=True
).to(device)

mlp_model = MLPClassifier().to(device)
checkpoint = torch.load('mlp_classifier.pth', map_location=device)
ast_model.load_state_dict(checkpoint['ast'])
mlp_model.load_state_dict(checkpoint['mlp'])
ast_model.eval()
mlp_model.eval()
criterion = nn.CrossEntropyLoss()

# 성별 평가
for gender in ['M', 'F']:
    gender_file_paths = [fp for fp, g in zip(file_paths, genders) if g == gender]
    gender_labels = [lb for lb, g in zip(labels, genders) if g == gender]

    _, val_f, _, val_l = train_test_split(
        gender_file_paths, gender_labels, test_size=0.2, random_state=42, stratify=gender_labels
    )

    val_ds = AudioDataset(val_f, val_l)
    val_loader = DataLoader(val_ds, batch_size=16)

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

    print(f'[Gender: {"Male" if gender == "M" else "Female"}] Val Loss: {val_loss:.4f} | Val Accuracy: {accuracy:.4f}')