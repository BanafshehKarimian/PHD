import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv

def pycox_cindex_td_from_outputs(outputs_np, labels_np, eps=1e-8):
    """
    Compute time-dependent C-index using pycox EvalSurv, but *remove Unknown effect* by:
      1) dropping samples with no known horizons at all
      2) optionally dropping samples where the known part is too short/degenerate
      3) making sure survival matrix has no NaNs/Infs and is strictly in (0,1)

    outputs_np: [N, T] logits (higher -> higher risk)
    labels_np:  [N, T] with {0,1,-1} where -1=Unknown
    """
    outputs_np = np.asarray(outputs_np)
    labels_np  = np.asarray(labels_np)

    N, T = labels_np.shape

    durations = []
    events = []
    keep = []

    # Convert dx horizons -> (duration, event), but DROP fully-unknown rows
    for i in range(N):
        row = labels_np[i]
        known = np.where(row != -1)[0]
        if len(known) == 0:
            continue  # drop this row entirely

        pos = np.where(row == 1)[0]
        if len(pos) > 0:
            durations.append(float(pos[0] + 1))
            events.append(1)
        else:
            durations.append(float(known.max() + 1))
            events.append(0)

        keep.append(i)

    if len(keep) < 2:
        return float("nan")

    keep = np.asarray(keep, dtype=int)
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)

    # Survival proxy: sigmoid(-logit) = 1/(1+exp(logit))
    surv_np = 1.0 / (1.0 + np.exp(outputs_np[keep]))  # [M, T]

    # Clean numeric issues (can cause NaN c-index)
    surv_np = np.nan_to_num(surv_np, nan=0.5, posinf=1.0 - eps, neginf=eps)
    surv_np = np.clip(surv_np, eps, 1.0 - eps)

    # Optional: enforce non-increasing survival over time (helps EvalSurv behave)
    # survival(t) should not increase with t
    surv_np = np.minimum.accumulate(surv_np, axis=1)

    times = np.arange(1, T + 1, dtype=float)
    surv_df = pd.DataFrame(surv_np.T, index=times)  # [T, M]

    ev = EvalSurv(surv_df, durations, events, censor_surv="km")
    return float(ev.concordance_td())

def pycox_cindex_td_from_outputs_(outputs_np, labels_np):
    """
    outputs_np: [N, T] model outputs (logits/cum hazard-ish)
    labels_np:  [N, T] with {0,1,-1} (Unknown=-1)
    """
    N, T = labels_np.shape

    # Convert labels -> durations, events
    durations = np.zeros(N, dtype=float)
    events = np.zeros(N, dtype=int)

    for i in range(N):
        row = labels_np[i]
        known = np.where(row != -1)[0]
        if len(known) == 0:
            durations[i] = T
            events[i] = 0
            continue

        pos = np.where(row == 1)[0]
        if len(pos) > 0:
            durations[i] = pos[0] + 1
            events[i] = 1
        else:
            durations[i] = known.max() + 1
            events[i] = 0

    # Build survival probabilities from outputs.
    # If outputs are cumulative hazard logits, a simple monotone survival proxy is:
    # surv(t) = sigmoid(-output_t)
    # (This guarantees survival decreases with increasing output.)
    surv_np = 1.0 / (1.0 + np.exp(outputs_np))  # sigmoid(-outputs)

    # Ensure shape for EvalSurv: [T, N] as DataFrame
    times = np.arange(1, T + 1, dtype=float)  # must be increasing
    surv_df = pd.DataFrame(surv_np.T, index=times)

    ev = EvalSurv(surv_df, durations, events, censor_surv='km')
    return ev.concordance_td()
from lifelines import KaplanMeierFitter

def censoring_dist_from_TE(T, E):
    # Uno IPCW wants G(t)=P(C>=t). Fit KM where censoring is the "event".
    kmf = KaplanMeierFitter()
    censor_event = 1 - E  # 1 if censored
    kmf.fit(T, event_observed=censor_event)
    uniq = np.unique(T).astype(int)
    return {int(t): float(kmf.predict(t)) for t in uniq}

def outputs_to_surv_matrix(outputs_np):
    """
    outputs_np: [M,K] logits
    return: [M,K+1] survival probs where surv[:,t] corresponds to horizon t (t=1..K)
    """
    M, K = outputs_np.shape
    surv = 1.0 / (1.0 + np.exp(outputs_np))   # sigmoid(-logit)
    pred = np.zeros((M, K + 1), dtype=float)
    pred[:, 1:] = surv
    return pred

import numpy as np

def dx_to_TE(labels_np):
    """
    labels_np: [N,K] with values {1,0,-1}
    Returns:
      T: [M] times in {1..K}
      E: [M] events in {0,1}
      keep: [M] indices into original arrays
    """
    N, K = labels_np.shape
    T_list, E_list, keep = [], [], []

    for i in range(N):
        row = labels_np[i]
        known = np.where(row != -1)[0]
        if len(known) == 0:
            continue

        pos = np.where(row == 1)[0]
        if len(pos) > 0:
            t = int(pos[0] + 1)
            e = 1
        else:
            t = int(known.max() + 1)
            e = 0

        T_list.append(t)
        E_list.append(e)
        keep.append(i)

    return np.asarray(T_list, dtype=float), np.asarray(E_list, dtype=float), np.asarray(keep, dtype=int)

## Adapted from: https://raw.githubusercontent.com/CamDavidsonPilon/lifelines/master/lifelines/utils/concordance.py
## Modified to weight by ipcw (inverse probality of censor weight) to fit Uno's C-index
## Modified to use a time-dependent score

import numpy as np
from lifelines.utils.btree import _BTree
from lifelines import KaplanMeierFitter
import pdb

def get_censoring_dist(train_dataset):
    _dataset = train_dataset.dataset.past_visits
    _dataset = _dataset.dropna()
    _dataset = _dataset[_dataset['time_at_event']>=0]
    times, event_observed = np.array(_dataset['time_at_event']), np.array(_dataset['y'])
    all_observed_times = set(times)
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed)

    censoring_dist = {time: kmf.predict(time) for time in all_observed_times}
    return censoring_dist

def concordance_index(event_times, predicted_scores, event_observed=None, censoring_dist=None):
    """
    Calculates the concordance index (C-index) between two series
    of event times. The first is the real survival times from
    the experimental data, and the other is the predicted survival
    times from a model of some kind.

    The c-index is the average of how often a model says X is greater than Y when, in the observed
    data, X is indeed greater than Y. The c-index also handles how to handle censored values
    (obviously, if Y is censored, it's hard to know if X is truly greater than Y).


    The concordance index is a value between 0 and 1 where:

    - 0.5 is the expected result from random predictions,
    - 1.0 is perfect concordance and,
    - 0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)

    Parameters
    ----------
    event_times: iterable
         a length-n iterable of observed survival times.
    predicted_scores: iterable
        a length-n iterable of predicted scores - these could be survival times, or hazards, etc. See https://stats.stackexchange.com/questions/352183/use-median-survival-time-to-calculate-cph-c-statistic/352435#352435
    event_observed: iterable, optional
        a length-n iterable censorship flags, 1 if observed, 0 if not. Default None assumes all observed.

    Returns
    -------
    c-index: float
      a value between 0 and 1.

    References
    -----------
    Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in
    developing models, evaluating assumptions and adequacy, and measuring and
    reducing errors. Statistics in Medicine 1996;15(4):361-87.

    Examples
    --------

    >>> from lifelines.utils import concordance_index
    >>> cph = CoxPHFitter().fit(df, 'T', 'E')
    >>> concordance_index(df['T'], -cph.predict_partial_hazard(df), df['E'])

    """
    event_times = np.asarray(event_times, dtype=float)
    predicted_scores = 1 - np.asarray(predicted_scores, dtype=float)


    if event_observed is None:
        event_observed = np.ones(event_times.shape[0], dtype=float)
    else:
        event_observed = np.asarray(event_observed, dtype=float).ravel()
        if event_observed.shape != event_times.shape:
            raise ValueError("Observed events must be 1-dimensional of same length as event times")

    num_correct, num_tied, num_pairs = _concordance_summary_statistics(event_times, predicted_scores, event_observed, censoring_dist)

    return _concordance_ratio(num_correct, num_tied, num_pairs)


def _concordance_ratio(num_correct, num_tied, num_pairs):
    if num_pairs == 0:
        raise ZeroDivisionError("No admissable pairs in the dataset.")
    return (num_correct + num_tied / 2) / num_pairs


def _concordance_summary_statistics(
    event_times, predicted_event_times, event_observed, censoring_dist
):  # pylint: disable=too-many-locals
    """Find the concordance index in n * log(n) time.

    Assumes the data has been verified by lifelines.utils.concordance_index first.
    """
    # Here's how this works.
    #
    # It would be pretty easy to do if we had no censored data and no ties. There, the basic idea
    # would be to iterate over the cases in order of their true event time (from least to greatest),
    # while keeping track of a pool of *predicted* event times for all cases previously seen (= all
    # cases that we know should be ranked lower than the case we're looking at currently).
    #
    # If the pool has O(log n) insert and O(log n) RANK (i.e., "how many things in the pool have
    # value less than x"), then the following algorithm is n log n:
    #
    # Sort the times and predictions by time, increasing
    # n_pairs, n_correct := 0
    # pool := {}
    # for each prediction p:
    #     n_pairs += len(pool)
    #     n_correct += rank(pool, p)
    #     add p to pool
    #
    # There are three complications: tied ground truth values, tied predictions, and censored
    # observations.
    #
    # - To handle tied true event times, we modify the inner loop to work in *batches* of observations
    # p_1, ..., p_n whose true event times are tied, and then add them all to the pool
    # simultaneously at the end.
    #
    # - To handle tied predictions, which should each count for 0.5, we switch to
    #     n_correct += min_rank(pool, p)
    #     n_tied += count(pool, p)
    #
    # - To handle censored observations, we handle each batch of tied, censored observations just
    # after the batch of observations that died at the same time (since those censored observations
    # are comparable all the observations that died at the same time or previously). However, we do
    # NOT add them to the pool at the end, because they are NOT comparable with any observations
    # that leave the study afterward--whether or not those observations get censored.
    if np.logical_not(event_observed).all():
        return (0, 0, 0)

    observed_times = set(event_times)


    died_mask = event_observed.astype(bool)
    # TODO: is event_times already sorted? That would be nice...
    died_truth = event_times[died_mask]
    ix = np.argsort(died_truth)
    died_truth = died_truth[ix]

    died_pred = predicted_event_times[died_mask][ix]

    censored_truth = event_times[~died_mask]
    ix = np.argsort(censored_truth)
    censored_truth = censored_truth[ix]
    censored_pred = predicted_event_times[~died_mask][ix]

    censored_ix = 0
    died_ix = 0
    times_to_compare = {}
    for time in observed_times:
        times_to_compare[time] = _BTree(np.unique(died_pred[:, int(time)]))
    num_pairs = np.int64(0)
    num_correct = np.int64(0)
    num_tied = np.int64(0)

    # we iterate through cases sorted by exit time:
    # - First, all cases that died at time t0. We add these to the sortedlist of died times.
    # - Then, all cases that were censored at time t0. We DON'T add these since they are NOT
    #   comparable to subsequent elements.
    while True:
        has_more_censored = censored_ix < len(censored_truth)
        has_more_died = died_ix < len(died_truth)
        # Should we look at some censored indices next, or died indices?
        if has_more_censored and (not has_more_died or died_truth[died_ix] > censored_truth[censored_ix]):
            pairs, correct, tied, next_ix, weight = _handle_pairs(censored_truth, censored_pred, censored_ix, times_to_compare, censoring_dist)
            censored_ix = next_ix
        elif has_more_died and (not has_more_censored or died_truth[died_ix] <= censored_truth[censored_ix]):
            pairs, correct, tied, next_ix, weight = _handle_pairs(died_truth, died_pred, died_ix, times_to_compare, censoring_dist)
            for pred in died_pred[died_ix:next_ix]:
                for time in observed_times:
                    times_to_compare[time].insert(pred[int(time)])
            died_ix = next_ix
        else:
            assert not (has_more_died or has_more_censored)
            break

        num_pairs += pairs * weight
        num_correct += correct * weight
        num_tied += tied * weight

    return (num_correct, num_tied, num_pairs)


def _handle_pairs(truth, pred, first_ix, times_to_compare, censoring_dist):
    """
    Handle all pairs that exited at the same time as truth[first_ix].

    Returns
    -------
      (pairs, correct, tied, next_ix)
      new_pairs: The number of new comparisons performed
      new_correct: The number of comparisons correctly predicted
      next_ix: The next index that needs to be handled
    """
    next_ix = first_ix
    truth_time = truth[first_ix]
    weight = 1./(censoring_dist[truth_time]**2)
    while next_ix < len(truth) and truth[next_ix] == truth[first_ix]:
        next_ix += 1
    pairs = len(times_to_compare[truth_time]) * (next_ix - first_ix)
    correct = np.int64(0)
    tied = np.int64(0)
    for i in range(first_ix, next_ix):
        rank, count = times_to_compare[truth_time].rank(pred[i][int(truth_time)])
        correct += rank
        tied += count

    return (pairs, correct, tied, next_ix, weight)
