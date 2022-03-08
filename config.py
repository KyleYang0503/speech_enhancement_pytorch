import os

fs = 16000

chunk_length = 4 * 16000
win_size = 320
fft_num = 320
win_shift = 160

causal_flag = True

stage_number = 2
batch_size = 2
epoch = 200
lr = 0.0015


dataset_path = './Dataset_'
json_path = './Json'
loss_path = './Loss/darcn_loss_record.mat'
save_path = './Model'
check_point = 1
continue_from = ''
best_path = './Best_model/darcn_causal_final.pth'
os.makedirs(save_path, exist_ok=True)
os.makedirs('./Best_model', exist_ok=True)
os.makedirs('./Loss', exist_ok=True)