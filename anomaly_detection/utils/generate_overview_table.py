from generate_table import printTable
import numpy as np

FILE='anomaly_detection/log/9/auc.npy'

methods = [
'NVP Fixed\\\\Features',    # 0
'SVDD Soft\\\\Pretrained',  # 1
'SVDD Hard\\\\Pretrained',  # 2
'NVP\\\\No Pretraining',           # 3
'NVP\\\\Pretrained',        # 4
'SVDD Hard\\\\No Pretrained',     # 5
'SVDD Soft\\\\No Pretrained',     # 6
'Autoencoder'             # 7
]

method_order = [
7, 6, 5, 1, 2, 3, 4, 0
]

modalities = [
'RGB+G+A',      # 0
'RGB',          # 1
'IR',           # 2
'D',            # 3
'RGB+D',        # 4
'IR+D',         # 5
'RGB+IR+D',     # 6
'RGB+IR+D+N',   # 7
'RGB+IR+G+A',   # 8
'RGB+D+N',      # 9
'RGB+N',        # 10
'RGB+G',        # 11
'RGB+A',        # 12
'D+N',          # 13
'G+A',          # 14
'RGB+D+A',      # 15
'RGB+G+N',      # 16
'D+A',          # 17
]

mod_order = [
1, 3, 4, 11, 10, 12, 13, 17, 14, 9, 15, 16, 0, # No IR
2, 5, 6, 7, 8 #IR
]

val = np.load(FILE)

printTable(val, methods, modalities, lambda x: 'IR' not in x, method_order, mod_order)

print('\n\n\n\n')

printTable(val, methods, modalities, lambda x: 'IR' in x, method_order, mod_order)
