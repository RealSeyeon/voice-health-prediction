import pandas as pd
import shutil, os
import librosa
import soundfile as sf
from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise, Gain
from tqdm import tqdm
import numpy as np

# ✅ 폴더 정의 추가
base_dir = '/home/gpu_02/parkinson/data'
augmented_folder= os.path.join(base_dir, 'augmented')
folders = ['HC_M', 'HC_F', 'PD_M', 'PD_F']

# STEP 3️⃣: 스펙트로그램 생성 및 저장
spectrogram_folder = os.path.join(base_dir, 'spectrogram')
os.makedirs(spectrogram_folder, exist_ok=True)

# 스펙트로그램 파라미터 (정확히 논문 및 기존 코드 참고)
n_mels = 128
fixed_length = 10  # 초
sr = 16000         # 샘플링 주파수
n_fft = 400        # 25ms 윈도우
hop_length = 160   # 10ms 윈도우 이동

# 스펙트로그램 변환 함수 (최신 librosa 키워드 인자)
def audio_to_melspectrogram(audio, sr, n_mels=128, length=10, n_fft=400, hop_length=160):
    # 고정 길이 맞추기
    audio = librosa.util.fix_length(audio, size=sr * length)
    
    # 멜스펙트로그램 계산 (최신 librosa API 적용)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    
    # 데시벨 스케일로 변환
    S_db = librosa.power_to_db(S, ref=np.max)
    
    return S_db

for folder in folders:
    source_folder = os.path.join(augmented_folder, folder)
    spec_subfolder = os.path.join(spectrogram_folder, folder)
    os.makedirs(spec_subfolder, exist_ok=True)

    for filename in tqdm([f for f in os.listdir(source_folder) if f.endswith('.wav')], desc=f"Spec {folder}"):
        audio, _ = librosa.load(os.path.join(source_folder, filename), sr=16000)
        spec = audio_to_melspectrogram(audio, 16000)
        np.save(os.path.join(spec_subfolder, filename.replace('.wav', '.npy')), spec)

print("✅ 스펙트로그램 생성 완료!")

# STEP 4️⃣: 스펙트로그램 데이터 증강 (SpecAugment)
spectrogram_aug_folder = os.path.join(base_dir, 'spectrogram_augmented')
os.makedirs(spectrogram_aug_folder, exist_ok=True)

def spec_augment(spec, num_time_masks=2, num_freq_masks=2, max_time_mask=30, max_freq_mask=15):
    spec_aug = spec.copy()
    for _ in range(num_time_masks):
        t = np.random.randint(max_time_mask)
        t0 = np.random.randint(0, spec_aug.shape[1] - t)
        spec_aug[:, t0:t0+t] = 0
    for _ in range(num_freq_masks):
        f = np.random.randint(max_freq_mask)
        f0 = np.random.randint(0, spec_aug.shape[0] - f)
        spec_aug[f0:f0+f, :] = 0
    return spec_aug

for folder in folders:
    source_folder = os.path.join(spectrogram_folder, folder)
    aug_folder = os.path.join(spectrogram_aug_folder, folder)
    os.makedirs(aug_folder, exist_ok=True)

    for filename in tqdm([f for f in os.listdir(source_folder) if f.endswith('.npy')], desc=f"SpecAug {folder}"):
        spec = np.load(os.path.join(source_folder, filename))
        np.save(os.path.join(aug_folder, filename), spec)

        for i in range(4):
            spec_aug = spec_augment(spec)
            np.save(os.path.join(aug_folder, filename.replace('.npy', f'_aug_{i+1:02d}.npy')), spec_aug)

print("✅ 스펙트로그램 증강 완료!")

# 스펙트로그램 증강 후 총 개수 출력
total_augmented_files = sum(
    len([f for f in files if f.endswith('.npy')])
    for _, _, files in os.walk(spectrogram_aug_folder)
)

print(f"✅ 스펙트로그램 증강 후 총 데이터 개수: {total_augmented_files}개")