import torch
import torch.nn as nn
import torch.nn.functional as F

from models.visit_aggregator import *
from models.survival_module import *
from vmramar.models.factory import load_model, RegisterModel, get_model_by_name
from types import SimpleNamespace

class VMRA(nn.Module):
    def __init__(self, config, student = False):
        super(VMRA, self).__init__()

        # VisitAggregator:
        #   - Takes a longitudinal sequence of visit embeddings
        #   - Adds temporal/positional signal
        #   - Uses a transformer encoder to contextualize visits
        #   - Pools over time to produce a single patient/history representation
        args_vmrnn = {'depths_downsample':[1], 'depths_upsample':[1], "embed_dim":612, "feature_resolution":(1, 1)}
        args_vmrnn = SimpleNamespace(**args_vmrnn)
        self.vmrnn = get_model_by_name('vmrnn1', False, args_vmrnn)

        # SurvivalModule:
        #   - Maps the pooled patient embedding to time-dependent risk / hazard outputs
        #   - Produces one score per follow-up horizon (e.g., 1..T years)
        self.classifier = SurvivalModule(config)

        # Dropout applied to the pooled history embedding before the survival head.
        self.dropout = torch.nn.Dropout(p=config['global_do_rate'], inplace=False)

        # Cache device for consistent tensor placement.
        self.torch_device = config['torch_device']


    def forward(self, batch, visit_key = 'visit_embeddings', mask_key = 'visit_mask'):
        # Expected batch fields:
        #   - visit_embeddings: [B, n_visits, embedding_dim]
        #   - visit_mask: [B, n_visits]

        orig = batch["original_visit_embeddings"].to(self.torch_device).float()  # [B,5,D]
        mask = batch["original_visit_mask"].to(self.torch_device).float()  # [B,5]
        fused_feats = orig * (1-mask).unsqueeze(-1)
        states = None
        vmrnn_outputs = []

        for t in range(5):
            feat_t = fused_feats[:, t]        # [B, D]
            feat_t = feat_t.unsqueeze(1)      # [B, 1, D]  (L = 1)

            temporal_output_t, states = self.vmrnn(feat_t, states_down=states)
            # temporal_output_t: [B, 1, D]
            vmrnn_outputs.append(temporal_output_t)
        vmrnn_outputs = torch.stack(vmrnn_outputs, dim=1) 
        combined_feats =  vmrnn_outputs[:, -1, 0, :]  #temporal_output.mean(dim=-1).mean(dim=-1)
        hidden_states = combined_feats
        # Output: [B, n_followup_years]
        risk_prediction_logits = self.classifier(self.dropout(hidden_states))
        return risk_prediction_logits, hidden_states
