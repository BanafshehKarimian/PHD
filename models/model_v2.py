import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from models.visit_aggregator import *
from models.survival_module import *
from models.model import LoMaR
import torch
import torch.nn as nn
import torch.nn.functional as F

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


        self.teacher_branch = []

        # Dropout applied to the pooled history embedding before the survival head.
        self.dropout = torch.nn.Dropout(p=config['global_do_rate'], inplace=False)

        # Cache device for consistent tensor placement.
        self.torch_device = config['torch_device']

        D = config["input_embedding_dim"]

        '''self.imputer = RecurrentStepBackImputer(d_in=D,d_hid=512)
        self.mapper = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(4*D, 4*D),
            nn.ReLU(inplace=True),
            nn.Linear(4*D, 4*D))'''
        self.imputer = HistoryPredictor(
            d_in=D,
            d_hid=512,
            n_hist=4,
            dropout=0.1,
        )
        for net in self.imputer.net:
            net.to(self.torch_device)
        self.config = config

    @staticmethod
    def _freeze_module(m: nn.Module):
        m.eval()  # optional, but common for frozen teacher-like modules
        for p in m.parameters():
            p.requires_grad = False

    def add_teacher_model(self, path):
        teacher = LoMaR(self.config)
        chkpnt = torch.load(path, map_location=self.torch_device)
        if isinstance(chkpnt, dict) and "state_dict" in chkpnt:
            chkpnt = chkpnt["state_dict"]
        elif isinstance(chkpnt, dict) and "model" in chkpnt:
            chkpnt = chkpnt["model"]
        teacher.load_state_dict(chkpnt, strict=True)
        # Freeze the copies
        self._freeze_module(teacher)
        self.teacher_branch.append(teacher.cuda())

    def load_model(self, path):
        chkpnt = torch.load(path, map_location=self.torch_device)
        if isinstance(chkpnt, dict) and "state_dict" in chkpnt:
            chkpnt = chkpnt["state_dict"]
        elif isinstance(chkpnt, dict) and "model" in chkpnt:
            chkpnt = chkpnt["model"]
        self.teacher_branch.load_state_dict(chkpnt, strict=True)
        # Freeze the copies
        self._freeze_module(self.teacher_branch)
    
    def load_imputer(self, path, strict=True):
        map_location = self.torch_device
        ckpt = torch.load(path, map_location=map_location)
        imputer_sd = {}
        for k, v in ckpt.items():
            if k.startswith("imputer."):
                imputer_sd[k[len("imputer."):]] = v
        self.imputer.load_state_dict(imputer_sd, strict=strict)
        #self._freeze_module(self.imputer)
        

    def forward(self, batch, visit_key = 'visit_embeddings', mask_key = 'visit_mask', train_imputer = False):
        # Expected batch fields:
        #   - visit_embeddings: [B, n_visits, embedding_dim]
        #   - visit_mask: [B, n_visits]

        # Move inputs to device and ensure floating type for projections/transformer ops.
        if train_imputer:
            visit_embeddings = batch[visit_key].to(self.torch_device).float()
            mask = batch[mask_key].to(self.torch_device).float() # Mask is cast to bool inside the visit aggregator.
            orig = batch["original_visit_embeddings"].to(self.torch_device).float()  # [B,5,D]
            mask = batch["original_visit_mask"].to(self.torch_device).float()  # [B,5]
            x0 = orig[:, -1, :]                    # [B,D]
            pred_hist = self.imputer(x0)           # [B,4,D]
            return pred_hist, batch["original_visit_embeddings"][:,:-1,:], 1.0 - mask[:, :-1 ]
        

        # 1) Pool the longitudinal visit sequence into a single history embedding per sample.
        # Output: [B, D_model]

        orig = batch["original_visit_embeddings"].to(self.torch_device).float()  # [B,5,D]
        mask = batch["original_visit_mask"].to(self.torch_device).float()  # [B,5]

        # assume ordering [-4,-3,-2,-1,0]
        recon_mask = mask.clone()
        #recon_mask[:, :-1] = 0.0  # history now "available" via imputation
        x0 = orig[:, -1, :]                    # [B,D]
        pred_hist = self.imputer(x0)           # [B,4,D]
        pred_hist = pred_hist.view(orig.shape[0], 4, orig.shape[2])#self.mapper()
        pred_full = torch.cat([pred_hist, x0.unsqueeze(1)], dim=1)
        with torch.no_grad():
            risk_prediction_logits_, complete_history_embedding_ = [], []
            for teacher in self.teacher_branch:
                l, v = teacher(
                    batch, visit_key="original_visit_embeddings", mask_key="original_visit_mask"
                )
                risk_prediction_logits_.append(l)
                complete_history_embedding_.append(v)
        complete_history_embedding = self.visit_transformer(pred_full, recon_mask)

        # 2) Apply dropout + survival head to obtain time-dependent risk scores/logits.
        # Output: [B, n_followup_years]
        risk_prediction_logits = self.classifier(self.dropout(complete_history_embedding))

        logits ={}
        logits["true"] = risk_prediction_logits_
        logits["pred"] = risk_prediction_logits
        return risk_prediction_logits, {"pred": pred_hist, "true": batch["original_visit_embeddings"][:,:-1,:], "mask": 1.0 - mask[:,:-1], "logits": logits, "true_f":complete_history_embedding_, "pred_f": complete_history_embedding}
