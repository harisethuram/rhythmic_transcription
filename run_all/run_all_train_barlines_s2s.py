import sys
import os
import subprocess
import json

if __name__ == "__main__":
    hidden_dims = [128, 256, 512]
    batch_sizes = [4, 16, 64]
    num_layers = [1, 2]
    lrs = [1e-4, 1e-3, 1e-2]
    
    i = 1
    total = len(hidden_dims) * len(batch_sizes) * len(num_layers) * len(lrs)
    best_val_loss = float("inf")
    best_hyperparams = None
    for hidden_dim in hidden_dims:
        for num_layer in num_layers:
            for batch_size in batch_sizes:
                for lr in lrs:
                    print(f"Running experiment {i}/{total}")
                    i += 1
                    output_path = f"models/barline_s2s_hyperparam/lr_{lr}/batch_size_{batch_size}/hidden_dim_{hidden_dim}/num_layers_{num_layer}"
                    
                    if not os.path.exists(output_path):
                        command = [
                            "bash",
                            "run/train_barlines2s.sh",
                            "processed_data/all/parallel_barlines",
                            output_path,
                            f"{hidden_dim}",
                            f"{num_layer}",
                            f"{batch_size}",
                            f"{lr}",
                        ]
                        
                        subprocess.run(command)
                        
                    with open(os.path.join(output_path, "loss_vals.json"), "r") as f:
                        loss_vals = json.load(f)
                    curr_val_loss = loss_vals["best_val_loss"]
                    
                    if curr_val_loss < best_val_loss:
                        best_val_loss = curr_val_loss
                        best_hyperparams = {
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layer,
                            "batch_size": batch_size,
                            "lr": lr,
                        }
                        
                        
    # now we want to find the best hyperparam setting
    print("Best hyperparameters:")
    print(best_hyperparams)
    print("Best val loss:", best_val_loss)
    