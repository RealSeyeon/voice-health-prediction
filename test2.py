import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from transformers import ASTForAudioClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

# Sinusoidal Positional Encoding 추가 Transformer 모델
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=2, nhead=16, num_layers=2, max_len=1500):
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

# AST 모델 로딩 (encoder 층 사용, 학습X)
ast_model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    ignore_mismatched_sizes=True,
    output_hidden_states=True
).to(device).eval()

for param in ast_model.parameters():
    param.requires_grad = False

# 성별로 나누어 모델 학습
for gender_value, gender_name in [(0, 'male'), (1, 'female')]:
    print(f"\n===== Training for {gender_name.upper()} =====")

    idx_gender = [i for i, g in enumerate(genders) if g == gender_value]
    fps_gender = [file_paths[i] for i in idx_gender]
    labels_gender = [labels[i] for i in idx_gender]

    train_f, val_f, train_l, val_l = train_test_split(
        fps_gender, labels_gender, test_size=0.2, random_state=42, stratify=labels_gender
    )

    train_ds = AudioDataset(train_f, train_l)
    val_ds = AudioDataset(val_f, val_l)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    classifier = TransformerClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

    best_f1, patience, counter = 0, 5, 0

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

        avg_loss = total_loss / len(train_loader)

        # 평가
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
        precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary')

        print(f"[{gender_name.upper()}] Epoch {epoch+1}: Val Loss={val_loss:.4f}, Accuracy={accuracy:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}")

        # Best F1 모델 저장
        if f1 > best_f1:
            best_f1 = f1
            counter = 0
            torch.save(classifier.state_dict(), f'transformer_transfer_PE_{gender_name}.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"[{gender_name.upper()}] Early stopping triggered at epoch {epoch+1}")
                break

    print(f"[{gender_name.upper()}] Best F1-Score: {best_f1:.4f}")
