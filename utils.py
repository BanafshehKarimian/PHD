from lifelines.utils import concordance_index
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def kd_prob_loss(student_logits, teacher_logits, labels, tau=4.0):
    """
    KD for K independent Bernoulli heads (per-horizon).
    student_logits, teacher_logits: [B,K]
    labels: [B,K] with -1 unknown (mask uses labels)
    """
    mask = (labels != -1).float()  # [B,K]

    # soften
    t_prob = torch.sigmoid(teacher_logits / tau)      # [B,K]
    s_logit = student_logits / tau                    # [B,K]

    per_elem = F.binary_cross_entropy_with_logits(s_logit, t_prob, reduction="none")  # [B,K]
    loss = (per_elem * mask).sum() / mask.sum().clamp(min=1.0)

    return loss * (tau ** 2)

def kd_logit_loss(student_logits, teacher_logits, labels):
    mask = (labels != -1).float()
    per_elem = F.smooth_l1_loss(student_logits, teacher_logits, reduction="none")  # [B,K]
    return (per_elem * mask).sum() / mask.sum().clamp(min=1.0)


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)
def kd_loss_binary(student_logits, teacher_logits, temperature=4.0, mask=None, logit_stand=False):
    """
    Multi-label KD for logits shaped [B, K] (independent binary tasks).

    mask: [B, K] with 1 = include, 0 = ignore (e.g. labels != -1)
    """
    if logit_stand:
        student_logits = normalize(student_logits)
        teacher_logits = normalize(teacher_logits)

    # soften
    p_t = torch.sigmoid(teacher_logits / temperature)   # [B,K]
    p_s = torch.sigmoid(student_logits / temperature)   # [B,K]

    # BCE between probabilities (not logits) for distillation
    per = F.binary_cross_entropy(p_s, p_t, reduction="none")  # [B,K]
    per = per * (temperature ** 2)

    if mask is not None:
        per = per * mask
        return per.sum() / mask.sum().clamp(min=1.0)

    return per.mean()
def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand, mask = None):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none")#.sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd.sum(1).mean()
def kd_loss_multilabel(student_logits, teacher_logits, temperature=4.0, mask=None):
    teacher_logits = teacher_logits.detach()

    p_t = torch.sigmoid(teacher_logits / temperature)
    p_s = torch.sigmoid(student_logits / temperature)

    per = F.binary_cross_entropy(p_s, p_t, reduction="none") * (temperature ** 2)

    if mask is not None:
        per = per * mask
        return per.sum() / mask.sum().clamp(min=1.0)
    return per.mean()
def dx_to_surv(dy):  # dy shape [N, K] values in {0,1,-1}
    K = dy.shape[1]
    event_time = np.full(len(dy), K, dtype=float)   # default: censored at K
    event_observed = np.zeros(len(dy), dtype=int)

    for i in range(len(dy)):
        row = dy[i]
        known = row[row != -1]
        if len(known) == 0:
            # no information, you may want to drop these from c-index
            event_time[i] = np.nan
            event_observed[i] = 0
            continue

        # find first year with a 1 among known horizons
        ones = np.where(row == 1)[0]
        if len(ones) > 0:
            event_time[i] = ones[0] + 1   # years are 1..K
            event_observed[i] = 1
        else:
            # censored at last known horizon (not necessarily K)
            last_known = np.where(row != -1)[0].max()
            event_time[i] = last_known + 1
            event_observed[i] = 0

    return event_time, event_observed

def c_index_lifelines(outputs, labels):
    # outputs: [N,K], labels: [N,K]
    t, e = dx_to_surv(labels)

    # drop rows with no known labels
    ok = ~np.isnan(t)
    t = t[ok]
    e = e[ok]

    risk = outputs[ok, -1]  # one scalar per person
    cind = concordance_index(t, -risk, e)  # negate because lifelines expects higher=better survival
    return cind
