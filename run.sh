# Execute the following scripts to use a gnn explainer and generate augmented graphs:
cd DISGEN/src
python gnnexp.py --dataset_root DISGEN/dataset  --train_val_test_idx_save_root DISGEN/saved_idx  --dataset bbbp# Train DISGEN on bbbp

# Execute the following scripts to train DISGEN:
python disentgnn.py --dataset_root DISGEN/dataset  --train_val_test_idx_save_root DISGEN/saved_idx --saved_model DISGEN/saved_model --dataset bbbp --criterion pair_loss_cos --beta1 0.5 --beta2 1 --beta3 5e4 --lr 0.001
