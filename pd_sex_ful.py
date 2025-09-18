import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from transformers import ASTForAudioClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
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

# TransformerClassifier 정의
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=527, num_classes=2, nhead=17, num_layers=1):
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

# 데이터 경로 설정
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

# AST 모델 로딩 (전체 파인튜닝)
ast_model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    ignore_mismatched_sizes=True
).to(device)

for gender_value, gender_name in [(0, 'male'), (1, 'female')]:
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
    classifier.load_state_dict(torch.load(f'transformer_classifier_{gender_name}.pth'))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(ast_model.parameters()) + list(classifier.parameters()), lr=1e-5)

    best_f1, patience, counter, step = 0, 5, 0, 0

    writer = SummaryWriter(log_dir=f'runs/full_finetune_{gender_name}')

    for epoch in range(50):
        ast_model.train()
        classifier.train()
        total_loss = 0

        for spec, label in train_loader:
            spec, label = spec.to(device), label.to(device)
            logits = ast_model(spec).logits.unsqueeze(1)
            outputs = classifier(logits)
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        ast_model.eval()
        classifier.eval()
        preds, targets, probs = [], [], []
        with torch.no_grad():
            for spec, label in val_loader:
                spec, label = spec.to(device), label.to(device)
                logits = ast_model(spec).logits.unsqueeze(1)
                outputs = classifier(logits)
                prob = torch.softmax(outputs, dim=1)[:, 1]
                preds.extend(outputs.argmax(1).cpu().numpy())
                probs.extend(prob.cpu().numpy())
                targets.extend(label.cpu().numpy())

        accuracy = accuracy_score(targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary')
        roc_auc = roc_auc_score(targets, probs)

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/val', accuracy, epoch)
        writer.add_scalar('Precision/val', precision, epoch)
        writer.add_scalar('Recall/val', recall, epoch)
        writer.add_scalar('F1_Score/val', f1, epoch)
        writer.add_scalar('ROC_AUC/val', roc_auc, epoch)

        print(f'Epoch {epoch+1}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, ROC AUC: {roc_auc:.4f}')

        if f1 > best_f1:
            best_f1 = f1
            counter = 0
            torch.save({'ast_model': ast_model.state_dict(), 'classifier': classifier.state_dict()}, f'final_model_{gender_name}.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"EarlyStopping triggered for {gender_name}")
                break

    writer.close()