import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    ASTForAudioClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from sklearn.model_selection import train_test_split
from collections import Counter
import os
from transformers.integrations import TensorBoardCallback
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import shutil

# 1) 평가 지표 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"eval_accuracy": accuracy_score(labels, preds)}


# AST 모델 로드
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=2,
    ignore_mismatched_sizes=True
)

# backbone 동결
for p in model.audio_spectrogram_transformer.parameters():
    p.requires_grad = False

def collator(features):
    specs = torch.stack([f["input_values"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
    return {"input_values": specs, "labels": labels}
    


# 데이터 준비
data_root = 'data/spectrogram_augmented'
folders = ['HC_M', 'HC_F', 'PD_M', 'PD_F']

file_paths, diag_labels = [], []
for f in folders:
    folder_path = os.path.join(data_root, f)
    for fn in os.listdir(folder_path):
        if not fn.endswith('.npy'):
            continue
        file_paths.append(os.path.join(folder_path, fn))
        diag_labels.append(0 if f.startswith('HC') else 1)

print("클래스 분포:", Counter(diag_labels))

# AudioDataset 정의
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, fixed_time_len=1024):
        self.file_paths = file_paths
        self.labels = labels
        self.fixed_time_len = fixed_time_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        spec = np.squeeze(np.load(self.file_paths[idx]))  # → (128, T)
        T = self.fixed_time_len
        if spec.shape[1] < T:
            pad = np.zeros((spec.shape[0], T - spec.shape[1]), dtype=spec.dtype)
            spec = np.concatenate([spec, pad], axis=1)
        else:
            spec = spec[:, :T]

        spec_tensor = torch.tensor(spec, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {"input_values": spec_tensor, "labels": label}

# 데이터 분할
train_f, val_f, train_l, val_l = train_test_split(
    file_paths, diag_labels,
    test_size=0.2,
    random_state=42,
    stratify=diag_labels
)

train_ds = AudioDataset(train_f, train_l)
val_ds   = AudioDataset(val_f, val_l)



# 학습 설정 수정 (step 단위)
training_args = TrainingArguments(
    output_dir=".ast/results_all",
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=1,
    logging_strategy="steps",
    logging_steps=100,
    eval_steps=50,
    save_steps=50,
    logging_dir="./logs_all",
    report_to=["tensorboard"],       # ← 이 줄을 추가하세요
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=50,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Trainer 다시 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=5)
    ]
)


trainer.train()

print(f"\n===== Training all-data model ({len(train_ds)} samples) =====")

# 모델 저장
trainer.save_model("./ast_all_model")

# 체크포인트 삭제
shutil.rmtree(".ast/results_all", ignore_errors=True)

print("✅ 모든 과정이 정상적으로 완료되었습니다!")