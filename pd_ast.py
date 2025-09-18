import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from transformers import ASTForAudioClassification

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 데이터셋 클래스 (🚩 파일명 정렬 추가!)
class ParkinsonDataset(Dataset):
    def __init__(self, data_root: str, fixed_time_len: int = 1024):
        self.files = []
        self.fixed_time_len = fixed_time_len

        for folder in ['HC_M', 'HC_F', 'PD_M', 'PD_F']:
            folder_path = os.path.join(data_root, folder)
            file_names = sorted([fn for fn in os.listdir(folder_path) if fn.endswith('.npy')])
            for fn in file_names:
                self.files.append(os.path.join(folder_path, fn))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        spec = np.load(self.files[idx])
        spec = np.squeeze(spec)

        T = self.fixed_time_len
        if spec.shape[1] < T:
            pad = np.zeros((spec.shape[0], T - spec.shape[1]), dtype=spec.dtype)
            spec = np.concatenate((spec, pad), axis=1)
        else:
            spec = spec[:, :T]

        spec_tensor = torch.from_numpy(spec.astype(np.float32)).unsqueeze(0)
        return spec_tensor

# 데이터 로더 생성
data_root = 'data/spectrogram_augmented'
dataset = ParkinsonDataset(data_root, fixed_time_len=1024)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# AST 모델 로드
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
raw_model = ASTForAudioClassification.from_pretrained(
    model_name,
    ignore_mismatched_sizes=True
)

# 모델 래퍼 클래스
class ASTWrapper(nn.Module):
    def __init__(self, model: ASTForAudioClassification):
        super().__init__()
        self.model = model

    def forward(self, spec: torch.Tensor):
        spec = spec.squeeze(1)
        return self.model(spec)

# 모델 설정 및 평가 모드
model = ASTWrapper(raw_model).to(device)
model.eval()

logits_list = []

# 🚩 전체 데이터셋에 대해 로짓 추출
with torch.no_grad():
    for idx, spec in enumerate(loader):
        spec = spec.to(device)
        outputs = model(spec)
        logits = outputs.logits.cpu().numpy()
        logits_list.append(logits)

        if idx % 10 == 0:
            print(f"{idx}/{len(loader)} batches processed...")

# 모든 배치의 로짓을 합쳐서 저장
logits_all = np.concatenate(logits_list, axis=0)
np.save('data/ast_logits.npy', logits_all)
print("✅ 전체 AST logits 저장 완료:", logits_all.shape)
