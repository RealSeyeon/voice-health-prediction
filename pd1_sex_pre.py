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
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import shutil
from sklearn.metrics import classification_report, confusion_matrix

# 평가 지표 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"eval_accuracy": accuracy_score(labels, preds)}


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
        return {"input_values": spec_tensor, "labels": label}

def collator(features):
    specs = torch.stack([f["input_values"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
    return {"input_values": specs, "labels": labels}

data_root = 'data/spectrogram_augmented'
folders = ['HC_M', 'HC_F', 'PD_M', 'PD_F']

file_paths, diag_labels, genders = [], [], []
for f in folders:
    folder_path = os.path.join(data_root, f)
    for fn in os.listdir(folder_path):
        if not fn.endswith('.npy'):
            continue
        file_paths.append(os.path.join(folder_path, fn))
        diag_labels.append(0 if f.startswith('HC') else 1)
        genders.append(0 if f.endswith('_M') else 1)


for gender_value, gender_name in [(0, 'male'), (1, 'female')]:
    print(f"\n🚀 Training AST model for {gender_name}")

    gender_indices = [i for i, g in enumerate(genders) if g == gender_value]
    gender_files = [file_paths[i] for i in gender_indices]
    gender_labels = [diag_labels[i] for i in gender_indices]

    train_f, val_f, train_l, val_l = train_test_split(
        gender_files, gender_labels, test_size=0.2,
        random_state=42, stratify=gender_labels
    )

    train_ds = AudioDataset(train_f, train_l)
    val_ds = AudioDataset(val_f, val_l)

    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    for p in model.audio_spectrogram_transformer.parameters():
        p.requires_grad = False

    training_args = TrainingArguments(
        output_dir=f"./results_{gender_name}",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        logging_strategy="steps",
        report_to=["tensorboard"],
        logging_steps=100,
        eval_steps=50,
        save_steps=50,
        logging_dir=f"./logs_{gender_name}",
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
    trainer.save_model(f"./ast_{gender_name}_model")
    shutil.rmtree(f"./results_{gender_name}", ignore_errors=True)

print("✅ 모든 성별 모델 학습이 완료되었습니다!")