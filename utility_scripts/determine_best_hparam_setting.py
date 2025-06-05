# given a directory of hyperparameter settings, determine the best hyperparameter setting
import os
import json

if __name__ == "__main__":
    lrs = ["1e-3", "1e-4", "1e-5"]
    batch_sizes = [32, 64, 128]
    emb_size = [32, 64]
    hid_size = [128, 256]
    
    base_dir = "output/presentation_results/models/barlines"
    best_hparam_setting = None
    best_model_dir = None
    best_loss = float("inf")
    
    for lr in lrs:
        for batch_size in batch_sizes:
            for emb in emb_size:
                for hid in hid_size:
                    hparam_dir = os.path.join(base_dir, f"lr_{lr}/b_size_{batch_size}/emb_{emb}/hid_{hid}")
                    loss_file = os.path.join(hparam_dir, "loss_vals.json")
                    with open(loss_file, "r") as f:
                        loss_vals = json.load(f)
                        if loss_vals["best_val_loss"] < best_loss:
                            best_loss = loss_vals["best_val_loss"]
                            best_model_dir = hparam_dir
                            best_hparam_setting = {
                                "lr": lr,
                                "batch_size": batch_size,
                                "emb_size": emb,
                                "hid_size": hid
                            }
    
    print(f"Best hyperparameter setting: {best_hparam_setting}")
    print(f"Best validation loss: {best_loss}")
    print(f"Best model directory: {best_model_dir}")