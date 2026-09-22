import json
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from src.model.smq import SMQModel
from src.model.utils import distance_joints
from src.model.eval_utils import evaluate_local_hungarian, evaluate_global_hungarian

class Trainer:
    
    """Trains SMQ and evaluates with MoF, Edit and F1 scores."""
    
    def __init__(self, in_channels, filters, num_layers, latent_dim, num_actions, 
                 num_joints, num_person, patch_size, kmeans, kmeans_metric, 
                 sampling_quantile, replacement_strategy, decay,
                 dead_code_threshold=10, tc_weight=0.0, tc_beta=1.0, grad_checkpoint=False):
        """Builds the model and loss.

        Args:
            in_channels: Input feature channels per joint (C).
            filters: Base temporal conv width.
            num_layers: Number of dilated residual layers per stage.
            latent_dim: Latent channels per joint (Z).
            num_actions: Codebook size (K).
            num_joints: Number of joints (V).
            patch_size: Temporal window length for VQ (W).
            kmeans: Whether to initialize codebook with KMeans.
            kmeans_metric: Metric for KMeans init ('euclidean' or 'dtw').
            decay: EMA decay for codebook updates.
        """
        
        # Init model and loss
        self.model = SMQModel(in_channels = in_channels, filters = filters, 
                           num_layers = num_layers, latent_dim = latent_dim, 
                           num_actions = num_actions, num_joints = num_joints, 
                           num_person = num_person, patch_size = patch_size,
                           kmeans = kmeans, kmeans_metric = kmeans_metric, 
                           sampling_quantile = sampling_quantile, 
                           replacement_strategy = replacement_strategy, 
                           decay=decay, dead_code_threshold=dead_code_threshold,
                           tc_weight=tc_weight, tc_beta=tc_beta)
        
        self.mse = nn.MSELoss(reduction='none')

        # Activation checkpointing in every TCN stage (memory only)
        for m in self.model.modules():
            if hasattr(m, "grad_checkpoint"):
                m.grad_checkpoint = grad_checkpoint

    def train(self, save_dir, batch_gen, num_epochs, batch_size,
              learning_rate, commit_weight, mse_loss_weight, device,
              joint_distance_recons=True, micro_batch_size=None, save_every=5):

        # Train mode
        self.model.train()
        self.model.to(device)

        num_batches = batch_gen.num_batches(batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Gradient accumulation: process the batch in smaller chunks to bound
        # peak activation memory, while the optimizer still steps once per
        # full `batch_size` (gradients summed, scaled by chunk/batch_size).
        # Note: the VQ codebook's EMA update still runs once per chunk
        # (equivalent to micro_batch_size-granularity for that mechanism),
        # only the encoder/decoder weight update reflects the full batch.
        chunk_size = micro_batch_size or batch_size

        # Heartbeat file: written every batch so progress can be monitored
        # externally without relying on stdout buffering (which can be
        # fully withheld until process exit under some invocation wrappers).
        save_dir.mkdir(parents=True, exist_ok=True)
        heartbeat_path = save_dir / "heartbeat.txt"

        def write_heartbeat(msg):
            with open(heartbeat_path, "w") as hb:
                hb.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

        for epoch in range(num_epochs):

            pbar = tqdm(
            total=num_batches,
            desc=f"Training [Epoch {epoch+1}]",
            unit="batch",
            leave=False)

            epoch_rec_loss = 0.0
            epoch_commit = 0.0
            epoch_tc = 0.0
            batch_idx = 0

            while batch_gen.has_next():
                write_heartbeat(f"epoch {epoch+1}/{num_epochs} batch {batch_idx+1}/{num_batches}")
                batch_input, mask = batch_gen.next_batch(batch_size)
                batch_input, mask = batch_input.to(device), mask.to(device)
                actual_batch_size = batch_input.shape[0]

                optimizer.zero_grad()

                batch_rec_loss = 0.0
                batch_commit_loss = 0.0
                batch_tc_loss = 0.0

                for start in range(0, actual_batch_size, chunk_size):
                    end = min(start + chunk_size, actual_batch_size)
                    chunk_input = batch_input[start:end]
                    chunk_mask = mask[start:end]
                    chunk_weight = (end - start) / actual_batch_size

                    # Forward pass
                    reconstructed = self.model(chunk_input, chunk_mask)

                    # Reconstruction in joint-distance space
                    if joint_distance_recons:
                        x, x_hat = distance_joints(chunk_input), distance_joints(reconstructed)

                    # Vanilla Reconstruction
                    else :
                        x, x_hat = chunk_input, reconstructed

                    # Calculate loss
                    rec_loss = mse_loss_weight * torch.mean(self.mse(x, x_hat))

                    commit_loss = commit_weight * self.model.commit_loss
                    tc_loss = self.model.vq.tc_loss
                    loss = (rec_loss + commit_loss + tc_loss) * chunk_weight

                    # Backprop (accumulate); optimizer steps once per full batch
                    loss.backward()

                    batch_rec_loss += rec_loss.item() * chunk_weight
                    batch_commit_loss += commit_loss.item() * chunk_weight
                    batch_tc_loss += float(tc_loss) * chunk_weight

                optimizer.step()

                epoch_rec_loss += batch_rec_loss
                epoch_commit += batch_commit_loss
                epoch_tc += batch_tc_loss

                pbar.update(1)
                batch_idx += 1

            batch_gen.reset()
            pbar.close()
            
            # Save every `save_every` epochs (default 5, as published) and always the last
            if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), save_dir / f"epoch-{epoch+1}.model")
                torch.save(optimizer.state_dict(), save_dir / f"epoch-{epoch+1}.opt")
            
            print("[epoch %d]: Reconstruction Loss = %f -- Commit Loss = %f" % 
                  (epoch + 1, epoch_rec_loss / num_batches, 
                   epoch_commit / num_batches))
            with open(save_dir / "losses.jsonl", "a") as lf:
                lf.write(json.dumps({"epoch": epoch + 1,
                                     "rec": epoch_rec_loss / num_batches,
                                     "commit": epoch_commit / num_batches,
                                     "tc": epoch_tc / num_batches}) + "\n")

    def eval(self, model_path, features_path, gt_path, mapping_file,
                epoch, vis , plot_dir, device) :
    
        # Eval mode
        self.model.eval()
        
        with torch.no_grad():
            # Load model
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))

            # --- Sequence Level Evaluation ---
            local_mof, local_edit, local_f1_vec, gt_all, prediction_all = evaluate_local_hungarian(
                model=self.model,
                features_path=features_path,
                gt_path=gt_path,
                mapping_file=mapping_file,
                epoch=epoch,
                device=device,
                verbose=True,
            )

            # --- Dataset Level Evaluation ---
            mof, edit, f1_vec, pr2gt = evaluate_global_hungarian(
                model=self.model,
                features_path=features_path,
                gt_path=gt_path,
                device=device,          
                mapping_file=mapping_file,
                epoch = epoch,
                vis=vis,                      
                plot_dir=plot_dir,
                gt_all=gt_all,
                prediction_all=prediction_all,
                verbose=True,
            )