# Data augmentation on different data:
python gnnexp.py --dataset {dataset} --dataset_root {dataset_root}  --train_val_test_idx_save_root {train_val_test_idx_save_root}
# Train DISGEN on bbbp
python disentgnn.py --dataset {dataset} --dataset_root {dataset_root}  --train_val_test_idx_save_root {train_val_test_idx_save_root} --criterion {criterion}
# Train on other data (PROTEINS, GraphSST2 and NCI1)
python disentgnn_proteins.py --dataset {dataset} --dataset_root {dataset_root}  --train_val_test_idx_save_root {train_val_test_idx_save_root} --criterion {criterion}
