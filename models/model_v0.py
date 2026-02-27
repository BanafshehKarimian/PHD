import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from models.visit_aggregator import *
from models.survival_module import *
import math

def cosine_ramp(p_min: float, p_max: float, epoch: int, max_epoch: int) -> float:
    """
    Cosine annealing from p_min -> p_max over [0, max_epoch-1].
    epoch can be 0-based or 1-based; this handles both safely.
    """
    e = float(max(0, min(epoch, max_epoch - 1)))
    t = e / float(max(1, max_epoch - 1))  # in [0,1]
    # 0 -> 1 with cosine
    return p_min + 0.5 * (p_max - p_min) * (1.0 - math.cos(math.pi * t))

class HistoryPredictor(nn.Module):
    """
    Input:  year-0 embedding  [B, D_in]
    Output: predicted history [B, 4, D_in] for years -4..-1
    """
    def __init__(self, d_in: int, d_hid: int = 1024, n_hist: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_hist = n_hist
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hid, n_hist * d_in),
        )

    def forward(self, x0):
        B, D = x0.shape
        y = self.net(x0).view(B, self.n_hist, D)  # [B,4,D]
        return y

class LoMaRMasked(nn.Module):
    def __init__(self, config):
        super(LoMaRMasked, self).__init__()

        # VisitAggregator:
        #   - Takes a longitudinal sequence of visit embeddings
        #   - Adds temporal/positional signal
        #   - Uses a transformer encoder to contextualize visits
        #   - Pools over time to produce a single patient/history representation
        self.visit_transformer = VisitAggregator(config)

        # SurvivalModule:
        #   - Maps the pooled patient embedding to time-dependent risk / hazard outputs
        #   - Produces one score per follow-up horizon (e.g., 1..T years)
        self.classifier = SurvivalModule(config)

        # Dropout applied to the pooled history embedding before the survival head.
        self.dropout = torch.nn.Dropout(p=config['global_do_rate'], inplace=False)

        # Cache device for consistent tensor placement.
        self.torch_device = config['torch_device']
        D = config["input_embedding_dim"]

        self.imputer = HistoryPredictor(
            d_in=D,
            d_hid=2048,
            n_hist=4,
            dropout=0.1,
        )

        

    def forward(self, batch, visit_key = 'visit_embeddings', mask_key = 'visit_mask', use_pred = False):
        # Expected batch fields:
        #   - visit_embeddings: [B, n_visits, embedding_dim]
        #   - visit_mask: [B, n_visits]

        # Move inputs to device and ensure floating type for projections/transformer ops.
        orig = batch["original_visit_embeddings"].to(self.torch_device).float()  # [B,5,D]
        mask = batch["original_visit_mask"].to(self.torch_device).float()  # [B,5]
        recon_mask = torch.zeros_like(mask)

        # assume ordering [-4,-3,-2,-1,0]
        x0 = orig[:, -1, :]                    # [B,D]
        pred_hist = self.imputer(x0)           # [B,4,D]
        pred_full = torch.cat([pred_hist, x0.unsqueeze(1)], dim=1)

        # Build reconstructed visits: [-4..-1] predicted, [0] real
        '''if not use_pred:
            p_use_pred = cosine_ramp(p_min=0.0, p_max=1.0, epoch=epoch, max_epoch=max_epoch)
            use_pred = (torch.rand((), device=orig.device) < p_use_pred).item()'''
        if use_pred:
            #visit_embeddings = torch.cat([pred_hist, x0.unsqueeze(1)], dim=1).to(self.torch_device).float()  # [B,5,D]
            m = mask.unsqueeze(-1)  # [B,5,1]
            visit_embeddings =  pred_full  # [B,5,D](1.0 - m) * orig + m *

            # if we imputed, treat imputed history as present for the transformer
            recon_mask = mask.clone()
            recon_mask[:, :-1] = 0.0  # history now "available" via imputation
            complete_history_embedding = self.visit_transformer(visit_embeddings, recon_mask)
        else:
            visit_embeddings = orig.to(self.torch_device).float()
            # 1) Pool the longitudinal visit sequence into a single history embedding per sample.
            # Output: [B, D_model]
            complete_history_embedding = self.visit_transformer(visit_embeddings, mask)


        # 2) Apply dropout + survival head to obtain time-dependent risk scores/logits.
        # Output: [B, n_followup_years]
        risk_prediction_logits = self.classifier(self.dropout(complete_history_embedding))
        return risk_prediction_logits, {"pred": pred_hist, "true": batch["original_visit_embeddings"][:,:-1,:], "mask": 1.0 - mask[:,:-1]}
