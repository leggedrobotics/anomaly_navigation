from generate_table import printTable
import numpy as np

FILE_AUC='anomaly_detection/log_incremental_rgbd_10/auc.npy'
FILE_FPR5='anomaly_detection/log_incremental_rgbd_10/fpr5.npy'

train_dat = [
'Base',    # 0
'+Sun',  # 1
'+Twilight',  # 2
'+Rain',           # 3 
]

val_dat = [
'Sun',          # 0
'Fire',     # 1
'Rain',         # 2
'Wet',          # 3
'Twilight',         # 4
]

val_order = [
0,1,4,2,3
]


auc = np.transpose(np.load(FILE_AUC), (0,2,1))
fpr5 = np.transpose(np.load(FILE_FPR5), (0,2,1))

printTable(auc, val_dat, train_dat)

print('\n\n\n\n')

printTable(fpr5, val_dat, train_dat)
