import torch
import torch.nn as nn
import torch.nn.functional as F

from models.visit_aggregator import *
from models.survival_module import *

class Mirai(nn.Module):
    def __init__(self, config):
        super(Mirai, self).__init__()
        
        # SurvivalModule:
        #   - Maps the pooled patient embedding to time-dependent risk / hazard outputs
        #   - Produces one score per follow-up horizon (e.g., 1..T years)
        self.classifier = SurvivalModule(config)

        # Dropout applied to the pooled history embedding before the survival head.
        self.dropout = torch.nn.Dropout(p=config['global_do_rate'], inplace=False)

        # Cache device for consistent tensor placement.
        self.torch_device = config['torch_device']

    def forward(self, batch):
        # Expected batch fields:
        #   - visit_embeddings: [B, n_visits, embedding_dim]
        #   - visit_mask: [B, n_visits]

        # Move inputs to device and ensure floating type for projections/transformer ops.
        visit_embeddings = batch['visit_embeddings'].to(self.torch_device).float()
        mask = batch['visit_mask'].to(self.torch_device).float() # Mask is cast to bool inside the visit aggregator.

        # 1) Apply dropout + survival head to obtain time-dependent risk scores/logits.
        # Output: [B, n_followup_years]
        risk_prediction_logits = self.classifier(self.dropout(visit_embeddings[:,-1]))
        return risk_prediction_logits, None
