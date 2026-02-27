from dataset import BreastDataset 
from models.model import *
from models.mirai_model import *
from models.model_vmra import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import torch
import copy 
import os
import numpy as np 
import pandas as pd 
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import random
import os
from c_index import pycox_cindex_td_from_outputs
import json
import argparse
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR

model_dict = {"LoMaR": LoMaR, "Mirai": Mirai, "VMRA": VMRA}
def seed_function(seed, extra = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if extra:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(1)

def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


'''def get_loss(outputs, labels, y_mask):
    label_mask = (labels != -1).float()                 # [B, K]
    labels_safe = torch.clamp(labels, 0, 1)             # turn -1 into 0 (masked anyway)

    loss = F.binary_cross_entropy_with_logits(
        outputs, labels_safe, weight=label_mask, reduction="sum"
    ) / label_mask.sum().clamp(min=1.0)
    return loss'''

def compute_pos_weight_from_dataset(train_dataset, n_future_dx: int):
    """
    LoMaR-style class weighting for BCEWithLogits:
      pos_weight[k] = (#neg_known_k) / (#pos_known_k)
    computed over TRAIN ONLY, ignoring Unknown (-1).
    Returns torch.FloatTensor shape [K].
    """
    # Fast path: use the metadata dataframe if your Dataset exposes it
    if hasattr(train_dataset, "past_visits") and isinstance(train_dataset.past_visits, pd.DataFrame):
        df = train_dataset.past_visits
        dx_cols = [f"dx_{k}" for k in range(1, n_future_dx + 1)]
        if all(c in df.columns for c in dx_cols):
            label_map = {"Not Malignant": 0, "Malignant": 1, "Unknown": -1}
            pos = np.zeros(n_future_dx, dtype=np.float64)
            neg = np.zeros(n_future_dx, dtype=np.float64)

            for k, c in enumerate(dx_cols):
                v = df[c].map(label_map).astype("int32").to_numpy()
                known = v != -1
                pos[k] = np.sum(v[known] == 1)
                neg[k] = np.sum(v[known] == 0)

            # avoid division by zero: if no positives, weight=1 (won't matter much since no pos exist)
            pos_safe = np.maximum(pos, 1.0)
            w = neg / pos_safe
            w = np.clip(w, 1.0, 1000.0)  # prevent insane weights
            return torch.tensor(w, dtype=torch.float32)

    # Fallback: iterate over dataset __getitem__ (slower but safe)
    pos = torch.zeros(n_future_dx, dtype=torch.float64)
    neg = torch.zeros(n_future_dx, dtype=torch.float64)
    for i in range(len(train_dataset)):
        item = train_dataset[i]
        lab = item["label"]
        if not torch.is_tensor(lab):
            lab = torch.tensor(lab)
        lab = lab.view(-1).long()  # [K]
        known = lab != -1
        pos += ((lab == 1) & known).double()
        neg += ((lab == 0) & known).double()

    pos_safe = torch.clamp(pos, min=1.0)
    w = (neg / pos_safe).clamp(min=1.0, max=1000.0).float()
    return w


def get_loss(outputs, labels, pos_weight=None):
    """
    outputs: [B, K] logits
    labels:  [B, K] in {0,1,-1} where -1 is Unknown
    pos_weight: [K] tensor or None
    """
    label_mask = (labels != -1).float()            # [B, K]
    labels_safe = torch.clamp(labels, 0, 1).float()  # [B, K]

    # BCE with logits supports pos_weight (per class/horizon)
    per_elem = F.binary_cross_entropy_with_logits(
        outputs,
        labels_safe,
        pos_weight=pos_weight,     # None or [K] broadcast to [B,K]
        reduction="none"
    )  # [B, K]

    # mask out Unknowns and normalize by #known labels
    loss = (per_elem * label_mask).sum() / label_mask.sum().clamp(min=1.0)
    return loss

def get_example_config():  # without density, ver_1 data, visit aug off, no risk, with long early stop, survival.
    p = argparse.ArgumentParser(
        description="LoMaR/VMRA-MaR experiment config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # experiment
    p.add_argument("--exp-id", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--init-lr", type=float, default=1e-3)
    p.add_argument("--train-epoch", type=int, default=50)
    p.add_argument("--early", type=int, default=20)

    # model
    p.add_argument("--n-past-visits", type=int, default=5,
                   help="present point + 4 years of history")
    p.add_argument("--n-future-dx", type=int, default=5,
                   help="5 follow-up years")
    p.add_argument("--input-embedding-dim", type=int, default=612)
    p.add_argument("--model-embedding-dim", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--global-do-rate", type=float, default=0.25)
    p.add_argument("--model-weight-dir", type=str, default="",
                   help="path to model weights .pth (optional)")
    p.add_argument("--model-type", type=str, default="LoMaR",
                   help="what model to load")

    # data
    p.add_argument(
        "--path-to-csv",
        type=str,
        default="./lomar_like_metadata.csv",
    )
    p.add_argument("--n-pseudo", type=int, default=100,
                   help="number of pseudo test sets for evaluation")
    p.add_argument("--history-masking-id", type=int, default=None)

    # results
    p.add_argument("--results-root", type=str, default="./results/lomar",
                   help="root folder where the run folder will be created")
    p.add_argument("--run-name", type=str, default="",
                   help="optional override for results subfolder name")

    args = p.parse_args()

    # Build config dict with your original keys
    config = {
        "exp_id": args.exp_id,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "init_lr": args.init_lr,
        "train_epoch": args.train_epoch,

        "n_past_visits": args.n_past_visits,
        "n_future_dx": args.n_future_dx,
        "input_embedding_dim": args.input_embedding_dim,
        "model_embedding_dim": args.model_embedding_dim,
        "n_heads": args.n_heads,
        "global_do_rate": args.global_do_rate,
        "model_weight_dir": args.model_weight_dir,

        "path_to_csv": args.path_to_csv,
        "n_pseudo": args.n_pseudo,
        "results_root": args.results_root,
        "early": args.early,
        "model_type": args.model_type,
        "history_masking_id": args.history_masking_id,
    }
    return config

def compute_metrics(outputs, labels, fpr = None):
    """
    Compute ROC-AUC metrics for a multi-horizon prediction setting.

    Args:
        outputs: numpy array of shape [B, K] with predicted scores/logits per horizon.
        labels:  numpy array of shape [B, K] with binary labels {0,1} or -1 for unknown.

    Returns:
        List: [avg_rocauc, rocauc_year1, rocauc_year2, ...]
              avg_rocauc ignores NaNs (years where ROC-AUC is undefined).
    """
    rocauc_scores = []

    for i in range(labels.shape[1]):
        # Slice labels/predictions for horizon i (e.g., year i+1).
        year_labels = labels[:, i]
        year_predictions = outputs[:, i]

        # Only evaluate on valid labels (here: -1 means Unknown).
        mask = (year_labels != -1)

        # ROC-AUC is defined only if both classes {0,1} are present in the filtered labels.
        if len(np.unique(year_labels[mask])) == 2:
            if fpr is None:
                rocauc_score = roc_auc_score(year_labels[mask], year_predictions[mask])
            else:
                rocauc_score = roc_auc_score(year_labels[mask], year_predictions[mask], max_fpr=fpr)
            rocauc_scores.append(rocauc_score)
        else:
            rocauc_scores.append(float('nan'))

    # Average ROC-AUC across horizons, ignoring undefined horizons (NaNs).
    avg_rocauc = np.nanmean(rocauc_scores)
    
    # Return average first, then per-horizon scores.
    return [avg_rocauc] + rocauc_scores

def evalute(model, test_dataset, test_loader, config, fpr = None):
        """
        Run a single forward pass over the entire test_loader (assumed full-batch),
        then compute pseudo-group ROC-AUC metrics based on dataset-provided boolean
        columns (pseudo_0, pseudo_1, ...).

        Note:
            - This function currently takes only the *first batch* from test_loader.
              In the demo() setup, batch_size=len(test_dataset), so this corresponds
              to evaluating the full dataset in one pass.
        """
        device = torch.device(config['torch_device'])
        model.eval()
        outputs = []
        labels = []
        with torch.no_grad():
            # Fetch a single batch (in this demo: the full dataset).
            for batch in tqdm(test_loader):
                # Labels: [B, K] where K = number of future horizons (e.g., 5 years).
                batch = to_device(batch, device)
                labels.append(batch['label'])
                # Model outputs: expected [B, K].
                outputs.append(model(batch)[0].cpu().detach())
        outputs = torch.cat(outputs, dim = 0)
        labels = torch.cat(labels, dim = 0)
        
        # Collect results per pseudo split/group.
        pseudo_results = pd.DataFrame()
        for i_pseudo in range(config['n_pseudo']):
            # Ensure there is a boolean column pseudo_i in the dataset dataframe.
            # If absent, default to True for all rows (i.e., use all samples).
            
            # Note that if you are using a subject multiple times in your meta csv, 
            # you must set the pseudo indices so that each subject is used only once 
            # in a pseudo test set. Otherwise your evaluation will be biased. Please 
            # refer to the paper for more info.
            if config['n_pseudo'] > 1 and "pseudo_"+str(i_pseudo) in test_dataset.past_visits.columns:
                pass 
            else: 
                test_dataset.past_visits["pseudo_"+str(i_pseudo)] = True

            # Boolean indexing mask over rows/samples.
            #indices = list(test_dataset.past_visits[test_dataset.past_visits["pseudo_"+str(i_pseudo)]].index)
            indices = np.where(test_dataset.past_visits[f"pseudo_{i_pseudo}"].values)[0]

            # Filter predictions/labels for this pseudo group and compute ROC-AUCs.
            pseudo_preds = outputs[indices].detach().cpu().numpy()
            pseudo_labels = labels[indices].detach().cpu().numpy()
            pseudo_rocauc = compute_metrics(pseudo_preds, pseudo_labels, fpr = fpr)
            pseudo_cind = pycox_cindex_td_from_outputs(pseudo_preds, pseudo_labels)

            # Store results in a dataframe for easy averaging/printing.
            pseudo_results.loc[i_pseudo, 'history_masking_id'] = test_dataset.history_masking_id
            cols = ['1_year', '2_year', '3_year', '4_year', '5_year']
            pseudo_results.loc[i_pseudo, cols] = pseudo_rocauc[1:6]
            pseudo_results.loc[i_pseudo, ['c_index']] = pseudo_cind

        # Average across pseudo groups; indexed by history_masking_id for convenience.
        average_pseudo_results = pd.DataFrame(columns=pseudo_results.columns)
        average_pseudo_results.loc[test_dataset.history_masking_id] = pseudo_results.mean()

        # Pack outputs for downstream analysis / saving.
        res = {}
        #res['test_labels'] = labels.detach().cpu().numpy()
        #res['test_outputs'] = outputs.detach().cpu().numpy()
        res['test_average_pseudo_results'] = average_pseudo_results.to_dict()
        return res



def train(model, train_loader, val_loader, config):
    run_dir = f'{config["exp_id"]}_{config["history_masking_id"]}_{config["init_lr"]}_{config["weight_decay"]}_{config["model_embedding_dim"]}_{config["n_heads"]}'

    config["results_dir"] = os.path.join(config["results_root"], run_dir)
    os.makedirs(config["results_dir"], exist_ok=True)
    device = torch.device(config['torch_device'])
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["init_lr"],
        weight_decay=config["weight_decay"]
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["train_epoch"],   # full cosine over all epochs
        eta_min=config.get("min_lr", 1e-6)
    )
    k = 3
    cols = ['1_year', '2_year', '3_year', '4_year', '5_year']
    per_epoch_results = pd.DataFrame()


    best_metric = -100000
    noimprove = 0
    
    pos_weight = compute_pos_weight_from_dataset(train_loader.dataset, config["n_future_dx"]).to(device)
    print("pos_weight (per horizon):", pos_weight.detach().cpu().numpy())

    for ep in range(config['train_epoch']):
        model.train()
        loss_log = 0.0
        ep_pred, ep_label = [], []

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            batch = to_device(batch, device)
            labels = batch['label'].to(device).float()     # [B, K]
            label_mask = (labels != -1).float()            # [B, K]
            labels_safe = torch.clamp(labels, 0, 1)        # [B, K]

            outputs, _ = model(batch)                         # [B, K]

            loss = get_loss(outputs, labels, pos_weight=pos_weight)
            
            '''F.binary_cross_entropy_with_logits(
                outputs, labels_safe, weight=label_mask, reduction='sum'
            ) / label_mask.sum().clamp(min=1.0)'''

            loss.backward()
            optimizer.step()   # <-- important

            loss_log += loss.item() * labels.size(0)

            ep_pred.append(outputs.detach().cpu())
            ep_label.append(labels.detach().cpu())

        #scheduler.step()
        
        ep_pred = torch.cat(ep_pred, dim=0).numpy()
        ep_label = torch.cat(ep_label, dim=0).numpy()

        rocauc = compute_metrics(ep_pred, ep_label)
        per_epoch_results.loc[ep, cols] = rocauc[1:6]
        
        cind = pycox_cindex_td_from_outputs(ep_pred, ep_label)

        val_loss_log = 0.0
        ep_pred, ep_label = [], []

        model.eval()
        for batch in tqdm(val_loader):
            batch = to_device(batch, device)
            labels = batch['label'].to(device).float()     # [B, K]
            label_mask = (labels != -1).float()            # [B, K]
            labels_safe = torch.clamp(labels, 0, 1)        # [B, K]
            with torch.no_grad():
                outputs, _ = model(batch)                         # [B, K]

                loss = get_loss(outputs, labels, pos_weight=pos_weight)
            
                val_loss_log += loss.item() * labels.size(0)

            ep_pred.append(outputs.detach().cpu())
            ep_label.append(labels.detach().cpu())

        ep_pred = torch.cat(ep_pred, dim=0).numpy()
        ep_label = torch.cat(ep_label, dim=0).numpy()

        val_rocauc = compute_metrics(ep_pred, ep_label)
        val_cind = pycox_cindex_td_from_outputs(ep_pred, ep_label)
        print(ep, loss_log / len(train_loader.dataset), cind, val_cind, val_loss_log)

        val_loss = val_loss_log / len(val_loader.dataset)
        if best_metric < -1 * val_loss:
            best_metric = -1 * val_loss
            torch.save(model.state_dict(), f"{config['results_dir']}/best.pth")
            noimprove = 0
        else:
            noimprove += 1
        if noimprove > config["early"]:
            break

    return {"train_results": per_epoch_results.to_dict(), "best_metric": best_metric}



def run_train(config):

    # Pick device automatically (GPU if available).
    config['torch_device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(config['torch_device'])
    print("config:")
    print(config)
  
    # Create model and load pretrained weights.
    model = model_dict[config["model_type"]](config).to(device)
    '''if config['model_weight_dir']:
        model = torch.load(config['model_weight_dir'])'''
    

    history_masking_id = config["history_masking_id"]
    train_dataset =  BreastDataset(config, history_masking_id, split="train", random_masking = False)
    val_dataset =  BreastDataset(config, history_masking_id, split="val", random_masking = False)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    
    print("Starting Training")
    start_time = time.time()
    train_results = train(model, train_loader, val_loader, config)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    print(train_results['train_results'], train_results["best_metric"])
    
    train_results['config'] = config 
    with open(f"{config['results_dir']}/log.json", "w") as json_file:
        json.dump(train_results, json_file)
    return train_results["best_metric"], f"{config['results_dir']}/best.pth"


def run_test(path, config, fpr = None):

    # Pick device automatically (GPU if available).
    config['torch_device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(config['torch_device'])
    print("config:")
    print(config)
  
    # Create model and load pretrained weights.
    model = model_dict[config["model_type"]](config).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    config["best_model_path"] = path
    

    history_masking_id = config["history_masking_id"]
    test_dataset =  BreastDataset(config, history_masking_id, split="test", random_masking = False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    
    inference_log = evalute(model, test_dataset, test_loader, config, fpr = fpr)
    print(inference_log["test_average_pseudo_results"])
    inference_log['config'] = config 
    with open(f"{config['results_root']}/test_log_{config['exp_id']}_{fpr}_{config['history_masking_id']}.json", "w") as json_file:
        json.dump(inference_log, json_file)

seed_function(0)    
config_tmp = get_example_config()
value, data = [], []
d_list = [config_tmp["model_embedding_dim"]]#[128, 256, 512]#
if config_tmp["model_type"] in ["Mirai", "VMRA"]:
    d_list = [612]
n_heads = [config_tmp["n_heads"]]#[1, 4, 8]#
Lrate = [1e-5]#[config_tmp["weight_decay"]]#
for d in d_list:
    for h in n_heads:
        for l in Lrate:
            config = copy.deepcopy(config_tmp)
            config["model_embedding_dim"] = d
            config["n_heads"] = h
            config["weight_decay"] = l
            v, p = run_train(config)
            value.append(v)
            data.append((p, d, h, l))
idx = np.argmax(value)
config["model_embedding_dim"] = data[idx][1]
config["n_heads"] = data[idx][2]
config["weight_decay"] = data[idx][3]
for i in [None,0,1,2,3]:
    config['history_masking_id'] = i
    for fpr in [None]:
        run_test(data[idx][0], config, fpr)
    