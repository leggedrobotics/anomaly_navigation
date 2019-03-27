python train_anomaly_detection.py selfsupervised StackConvNet log \
data/full --objective real-nvp \
--lr 0.0001 --n_epochs 150 --lr_milestone 100 --batch_size 200 --weight_decay 0.5e-6 \
--pretrain True --ae_lr 0.0001 --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 \
--ae_weight_decay 0.5e-6 --normal_class 1 --rgb --depth_3d --normals \
--train_folder train --val_pos_folder val/wangen_sun_3_pos --val_neg_folder val/wangen_sun_3_neg \
--fix_encoder