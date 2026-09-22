"""Nested, group-separated readouts; all transforms fitted inside their training split."""
import numpy as np
from script.vocab import readout as D


def run(X, y, groups, Xeval, C, family='linear', scale=True, seeds=(0, 1, 2),basis_blocks=None,progress_label=None):
    val = D.inner_split(groups)
    assert not set(groups[val]) & set(groups[~val])
    def scaler(a):
        return D.standardise_fit(a) if scale else (0., 1.)
    mi, si = scaler(X[~val])
    from script.vocab import linear_projection as LP
    if basis_blocks is None:
        xt = ((X[~val] - mi) / si).astype(np.float32)
        xv = ((X[val] - mi) / si).astype(np.float32)
    else:
        assert family=='linear'
        projectors=LP.make_projectors(basis_blocks,mi,si)
        xt=LP.project(X[~val],mi,si,projectors);xv=LP.project(X[val],mi,si,projectors)
    log = []
    grid = D.LIN_GRID if family == 'linear' else D.MLP_GRID
    for cfg in grid:
        if progress_label:print('INNER START',progress_label,cfg,'solver_dim',xt.shape[1],flush=True)
        if family == 'linear':
            model = D.fit_linear(xt, y[~val], C, cfg)
            pred = D.predict(model, xv)
            ba = D.bal_acc_f1(np.bincount(y[val]*C+pred, minlength=C*C).reshape(C,C))[0]
            ep = None
        else:
            model, ep, ba = D.fit_mlp(xt, y[~val], C, *cfg, seed=0, Xv=xv, yv=y[val])
        entry=dict(config=cfg, ba=ba, epoch=ep)
        if family=='linear':entry.update(iterations=model.optim_iterations,gradient_max=model.optim_gradient_max,training_objective=model.optim_loss)
        log.append(entry)
        if progress_label:print('INNER DONE',progress_label,entry,flush=True)
        del model
    selected = log[int(np.argmax([l['ba'] for l in log]))]
    del xt, xv
    mu, sd = scaler(X)
    if basis_blocks is None:
        xt = ((X-mu)/sd).astype(np.float32)
        xe = ((Xeval-mu)/sd).astype(np.float32)
        projection_info=None
    else:
        projectors=LP.make_projectors(basis_blocks,mu,sd)
        xt=LP.project(X,mu,sd,projectors);xe=LP.project(Xeval,mu,sd,projectors)
        projection_info=LP.info(projectors)
    predictions = [];optimization=[]
    for seed in ((0,) if family == 'linear' else seeds):
        if progress_label:print('REFIT START',progress_label,selected['config'],flush=True)
        if family == 'linear':
            model = D.fit_linear(xt, y, C, selected['config'], seed)
        else:
            model = D.fit_mlp(xt, y, C, *selected['config'], seed=seed, epochs=selected['epoch'])[0]
        predictions.append(D.predict(model, xe))
        if family=='linear':optimization.append(dict(iterations=model.optim_iterations,gradient_max=model.optim_gradient_max,training_objective=model.optim_loss))
        del model
    return dict(predictions=np.asarray(predictions), selected=selected, tuning=log,optimization=optimization,
                solver=D.LINEAR_SOLVER if family=='linear' else 'adamw_128',solver_dimension=xt.shape[1],projection=projection_info,
                solver_parameters=(xt.shape[1]+1)*C if family=='linear' else (X.shape[1]+1)*D.MLP_HIDDEN+(D.MLP_HIDDEN+1)*C,
                validation_groups=np.unique(groups[val]), training_groups=np.unique(groups[~val]),
                parameters=(X.shape[1]+1)*C if family=='linear' else (X.shape[1]+1)*D.MLP_HIDDEN+(D.MLP_HIDDEN+1)*C)
