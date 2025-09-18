import os

data_root_before = 'data'
data_root_after = 'data/spectrogram_augmented'
folders = ['HC_M', 'HC_F', 'PD_M', 'PD_F']

# 증강 전 파일 개수
print('증강 전 파일 개수:')
for folder in folders:
    folder_path = os.path.join(data_root_before, folder)
    file_count = len([fn for fn in os.listdir(folder_path) if fn.endswith('.wav')])
    print(f'{folder}: {file_count}')

# 증강 후 파일 개수
print('\n증강 후 파일 개수:')
for folder in folders:
    folder_path = os.path.join(data_root_after, folder)
    file_count = len([fn for fn in os.listdir(folder_path) if fn.endswith('.npy')])
    print(f'{folder}: {file_count}')
