"""
Supervised diagnostic readouts shared by the vocabulary programme (Experiments 1 and 3).

These are DIAGNOSTICS, not unsupervised methods: they use fitting-recording labels. Their scores are
not upper bounds. All tuning (regularisation, early-stopping epoch) uses an INNER group split of the
fitting recordings; outer-fold labels are never used for selection.

Readout families (same family and same grid for every input representation compared):
  linear : multinomial logistic regression, L2 weight penalty, class-balanced loss
  mlp    : one hidden layer (128 ReLU), weight decay + dropout, class-balanced loss, Adam, early stopping

Everything is aggregated as PER-RECORDING confusion matrices (rows = truth, cols = prediction) so paired
group bootstraps (participants for LARa, recordings for BABEL) need no recomputation.
"""
import numpy as np
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LIN_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]                                  # L2 coefficient (5 configs)
MLP_GRID = [(wd, dr) for wd in (1e-4, 1e-3, 1e-2) for dr in (0.0, 0.5)]    # 6 configs
MLP_HIDDEN = 128
MLP_MAX_EPOCHS = 60
MLP_BATCH = 256
INNER_VAL_FRAC = 0.25
INNER_SEED = 777
LIN_ITERS = 2000
LINEAR_SOLVER = 'float64_lbfgs_2000_history50'


# ----------------------------------------------------------------------------- metrics
def bal_acc_f1(conf):
    """conf (C, C) rows = truth. Classes with no true support are ignored. Returns (balanced acc, macro-F1)."""
    conf = np.asarray(conf, dtype=np.float64)
    sup = conf.sum(1)
    tp = np.diag(conf)
    pred = conf.sum(0)
    ok = sup > 0
    rec = np.where(ok, tp / np.maximum(sup, 1e-12), 0.0)
    prec = np.where(pred > 0, tp / np.maximum(pred, 1e-12), 0.0)
    f1 = np.where(rec + prec > 0, 2 * prec * rec / np.maximum(rec + prec, 1e-12), 0.0)
    return float(rec[ok].mean()), float(f1[ok].mean())


def metrics_from_weighted(conf_rec, W):
    """conf_rec (R, C, C); W (B, R) recording weights -> balanced acc, macro-F1 arrays of shape (B,)."""
    T = np.einsum("br,rij->bij", W, conf_rec)
    sup = T.sum(2)
    tp = np.einsum("bii->bi", T)
    pred = T.sum(1)
    ok = sup > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(ok, tp / sup, 0.0)
        prec = np.where(pred > 0, tp / pred, 0.0)
        f1 = np.where(rec + prec > 0, 2 * prec * rec / (rec + prec), 0.0)
    n = ok.sum(1)
    return (rec * ok).sum(1) / n, (f1 * ok).sum(1) / n


def per_class_recall(conf):
    conf = np.asarray(conf, dtype=np.float64)
    sup = conf.sum(1)
    return np.where(sup > 0, np.diag(conf) / np.maximum(sup, 1e-12), np.nan)


def confusion_by_recording(y, pred, rec, n_rec, C, w=None):
    """(n_rec, C, C) confusion; w = optional per-item weights (durations)."""
    out = np.zeros((n_rec, C, C))
    np.add.at(out, (rec, y, pred), 1.0 if w is None else w)
    return out


# ----------------------------------------------------------------------------- models
def _class_weights(y, C):
    cnt = np.bincount(y, minlength=C).astype(np.float64)
    w = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), 0.0)
    return w / w[cnt > 0].mean() if (cnt > 0).any() else w


def _to_t(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEV)


def fit_linear(X, y, C, l2, seed=0):
    """Full-batch L-BFGS softmax regression. Loss = class-balanced mean CE + 0.5*l2*||W||^2 (bias unpenalised)."""
    torch.manual_seed(seed)
    Xt, yt = _to_t(X, torch.float64), _to_t(y, torch.long)
    cw = _to_t(_class_weights(y, C), torch.float64)
    lin = torch.nn.Linear(X.shape[1], C).to(device=DEV,dtype=torch.float64)
    torch.nn.init.zeros_(lin.weight); torch.nn.init.zeros_(lin.bias)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=LIN_ITERS, history_size=50,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        ce = torch.nn.functional.cross_entropy(lin(Xt), yt, weight=cw, reduction="sum") / cw[yt].sum()
        loss = ce + 0.5 * l2 * (lin.weight ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    lin.optim_loss = float(closure().detach().cpu())
    lin.optim_iterations = int(opt.state[lin.weight].get('n_iter', 0))
    lin.optim_gradient_max = float(max(p.grad.detach().abs().max().item() for p in lin.parameters()))
    return lin


def fit_mlp(X, y, C, wd, dropout, seed, Xv=None, yv=None, epochs=MLP_MAX_EPOCHS):
    """Mini-batch Adam. If (Xv, yv) given, returns (model, best_epoch, best_val_bal_acc) chosen by validation
    balanced accuracy; otherwise trains for `epochs` epochs and returns (model, epochs, None)."""
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    torch.manual_seed(seed)
    Xt, yt = _to_t(X), _to_t(y, torch.long)
    cw = _to_t(_class_weights(y, C))
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], MLP_HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                              torch.nn.Linear(MLP_HIDDEN, C)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    n = len(Xt)
    best, best_ep, state = -1.0, epochs, None
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=g).to(DEV)
        for s in range(0, n, MLP_BATCH):
            idx = perm[s:s + MLP_BATCH]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(Xt[idx]), yt[idx], weight=cw)
            loss.backward()
            opt.step()
        if Xv is not None:
            net.eval()
            with torch.no_grad():
                pv = net(_to_t(Xv)).argmax(1).cpu().numpy()
            ba = bal_acc_f1(np.bincount(yv * C + pv, minlength=C * C).reshape(C, C))[0]
            if ba > best:
                best, best_ep = ba, ep
                state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if ep - best_ep >= 10:
                break
    if Xv is not None and state is not None:
        net.load_state_dict(state)
    net.eval()
    return net, best_ep, (best if Xv is not None else None)


def predict(model, X):
    with torch.no_grad():
        return model(_to_t(X, next(model.parameters()).dtype)).argmax(1).cpu().numpy()


# ----------------------------------------------------------------------------- inner split + selection
def inner_split(groups_fit, seed=INNER_SEED, frac=INNER_VAL_FRAC):
    """Split FITTING items by group (participant / recording) -> boolean validation mask over items."""
    ug = np.unique(groups_fit)
    rng = np.random.default_rng(seed)
    nval = max(1, int(round(frac * len(ug))))
    val_g = set(rng.permutation(ug)[:nval].tolist())
    return np.array([g in val_g for g in groups_fit])


def select_and_fit(family, X, y, groups, C, seeds=(0, 1, 2)):
    """Tune on an inner group split of (X, y), then refit on all items with the chosen config.
    Returns dict(models=[...], config=..., inner=[(cfg, val_bal_acc, epoch)], family)."""
    val = inner_split(groups)
    Xtr, ytr, Xva, yva = X[~val], y[~val], X[val], y[val]
    log = []
    if family == "linear":
        for l2 in LIN_GRID:
            m = fit_linear(Xtr, ytr, C, l2)
            pv = predict(m, Xva)
            log.append((dict(l2=l2), bal_acc_f1(np.bincount(yva * C + pv, minlength=C * C).reshape(C, C))[0], None))
        i = int(np.argmax([l[1] for l in log]))
        cfg = log[i][0]
        return dict(family=family, config=cfg, inner=log, models=[fit_linear(X, y, C, cfg["l2"])])
    for wd, dr in MLP_GRID:
        _, ep, ba = fit_mlp(Xtr, ytr, C, wd, dr, seed=seeds[0], Xv=Xva, yv=yva)
        log.append((dict(wd=wd, dropout=dr), ba, ep))
    i = int(np.argmax([l[1] for l in log]))
    cfg, ep = log[i][0], log[i][2]
    models = [fit_mlp(X, y, C, cfg["wd"], cfg["dropout"], seed=s, epochs=ep)[0] for s in seeds]
    cfg = dict(cfg, epochs=ep)
    return dict(family=family, config=cfg, inner=log, models=models)


def standardise_fit(Xfit):
    mu = Xfit.mean(0, keepdims=True)
    sd = Xfit.std(0, keepdims=True)
    floor = 1e-3 * np.median(sd[sd > 0]) if np.any(sd > 0) else 1.0
    return mu.astype(np.float32), np.maximum(sd, floor).astype(np.float32)


# ----------------------------------------------------------------------------- baselines (analytic, exact)
def baseline_confusions(y_fit, y_true, rec, n_rec, C, w=None):
    """Per-recording expected confusion of (a) majority class of the fitting set, (b) class-prior sampling
    (predict class c with probability = fitting prior; exact expectation), (c) uniform guessing."""
    cnt = np.bincount(y_fit, minlength=C).astype(np.float64)
    prior = cnt / cnt.sum()
    maj = int(cnt.argmax())
    ww = np.ones(len(y_true)) if w is None else w
    T = np.zeros((n_rec, C))
    np.add.at(T, (rec, y_true), ww)                                   # true mass per recording and class
    out = {}
    m = np.zeros((n_rec, C, C)); m[:, :, maj] = T; out["majority"] = m
    out["prior"] = T[:, :, None] * prior[None, None, :]
    out["uniform"] = T[:, :, None] * np.full((1, 1, C), 1.0 / C)
    return out
