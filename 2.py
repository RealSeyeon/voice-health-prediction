import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from transformers import ASTForAudioClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, genders, fixed_time_len=1024):
        self.file_paths = file_paths
        self.labels = labels
        self.genders = genders
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
        gender = self.genders[idx]
        return spec_tensor, label, gender

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=2, nhead=16, num_layers=2, max_len=2000):
        super().__init__()
        self.pos_encoder = self.positional_encoding(max_len, input_dim).to(device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, batch_first=True)
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

# 데이터 로딩
data_root = 'data/spectrogram_augmented'
folders = ['HC_M', 'HC_F', 'PD_M', 'PD_F']

file_paths, labels, genders = [], [], []
for folder in folders:
    path = os.path.join(data_root, folder)
    for fn in sorted(os.listdir(path)):
        if fn.endswith('.npy'):
            file_paths.append(os.path.join(path, fn))
            labels.append(0 if folder.startswith('HC') else 1)
            genders.append(0 if folder.endswith('_M') else 1)

# 평가 데이터 분할
_, val_f, _, val_l, _, val_g = train_test_split(
    file_paths, labels, genders, test_size=0.2, random_state=42, stratify=labels
)

val_ds = AudioDataset(val_f, val_l, val_g)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

# AST 모델 로딩
ast_model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    ignore_mismatched_sizes=True,
    output_hidden_states=True
).to(device).eval()

# 학습된 모델 로딩
classifier = TransformerClassifier().to(device)
classifier.load_state_dict(torch.load('transformer_transfer_PE_all.pth', map_location=device))
classifier.eval()

# 성별로 평가
def evaluate_gender(loader, gender_value, gender_name):
    preds, targets = [], []
    with torch.no_grad():
        for spec, label, gender in loader:
            mask = (gender == gender_value)
            if mask.sum() == 0:
                continue
            spec = spec[mask].to(device)
            label = label[mask].to(device)
            hidden_states = ast_model(spec).hidden_states[-1]
            outputs = classifier(hidden_states)
            preds.extend(outputs.argmax(1).cpu().numpy())
            targets.extend(label.cpu().numpy())

    accuracy = accuracy_score(targets, preds)
    report = classification_report(targets, preds, target_names=['HC', 'PD'], digits=4)
    print(f"\n===== {gender_name.upper()} 데이터 평가 결과 =====")
    print(f"Accuracy: {accuracy:.4f}")
