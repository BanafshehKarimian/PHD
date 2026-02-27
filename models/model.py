import torch
import torch.nn as nn
import torch.nn.functional as F

from models.visit_aggregator import *
from models.survival_module import *

class LoMaR(nn.Module):
    def __init__(self, config, student = False):
        super(LoMaR, self).__init__()

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

        self.mapper = None
        if student:
            self.mapper = nn.Linear(config["model_embedding_dim"], config['teacher_embd'])

    def forward(self, batch, visit_key = 'visit_embeddings', mask_key = 'visit_mask'):
        # Expected batch fields:
        #   - visit_embeddings: [B, n_visits, embedding_dim]
        #   - visit_mask: [B, n_visits]

        # Move inputs to device and ensure floating type for projections/transformer ops.
        visit_embeddings = batch[visit_key].to(self.torch_device).float()
        #visit_embeddings[:,:-1,:] = torch.zeros((visit_embeddings.shape[0], 4, visit_embeddings.shape[2]))
        mask = batch[mask_key].to(self.torch_device).float() # Mask is cast to bool inside the visit aggregator.

        # 1) Pool the longitudinal visit sequence into a single history embedding per sample.
        # Output: [B, D_model]
        complete_history_embedding = self.visit_transformer(visit_embeddings, mask)

        # 2) Apply dropout + survival head to obtain time-dependent risk scores/logits.
        # Output: [B, n_followup_years]
        risk_prediction_logits = self.classifier(self.dropout(complete_history_embedding))
        if self.mapper:
            return risk_prediction_logits, self.mapper(complete_history_embedding)
        return risk_prediction_logits, complete_history_embedding
