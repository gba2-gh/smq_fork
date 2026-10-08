"""The dimensionality-reduction variants: what sits between the global standardizer and K-means-512.

Every variant is a linear map applied to the standardized window vector x (D-dim):

    z = ((x - mean_) @ components_.T) / divisor_          (d-dim)

so only two things can differ between variants: WHICH subspace is kept (the row space of `components_`) and HOW its
axes are scaled (`divisor_`). A rotation inside the kept subspace changes nothing downstream -- K-means, the centroid
lookup and the mean-centred cosine OT cost are all invariant to it -- which is why ICA (a rotation of whitened PCA) is
not a variant here.

Methods
  pca       randomized PCA to `dim` (the anchor path). Whitening alpha in {0, 1} goes through sklearn's own
            `PCA(whiten=...)` so the anchor reproduces `timescale_inspection.tokenizers.fit_population` exactly;
            any other alpha divides component k by explained_variance_k ** (alpha / 2).
  pca_var   full-SVD PCA, keeping the fewest components whose cumulative explained variance reaches `var`.
  identity  no reduction: K-means / OT on the standardized vector itself.
  random    Gaussian random projection to `dim` (JL control: is PCA's variance ordering doing anything?).
  sfa       linear slow feature analysis: whitened PCA to `sfa_pre` dims, then the `dim` directions whose
            window-to-window change (adjacent windows of the same recording) has the least variance.
            Unit variance by construction.
  kpca      kernel PCA with an RBF kernel, Nystrom form: `n_landmarks` windows drawn from the fitting sample, bandwidth
            sigma = median pairwise distance between landmarks (label-free), exact kernel PCA on the landmarks, every
            other window projected through its kernel row to the landmarks. Component scaling follows PCA (variance
            proportional to the eigenvalue; alpha > 0 whitens it the same way).
  wrpca     within-recording PCA: axes fitted on windows minus their recording's mean (so variance shared by a whole
            recording -- subject, sensor placement -- cannot claim an axis); the data are projected uncentred.

Whitening for non-PCA methods (random, wrpca): alpha > 0 divides each output coordinate by its fitting-sample
standard deviation ** alpha. For PCA axes this is identical to PCA whitening; for random axes it is per-axis
standardization (they stay correlated).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA

METHODS = ("pca", "pca_var", "identity", "random", "sfa", "wrpca", "kpca")


@dataclass(frozen=True)
class Variant:
    key: str
    stage: str
    method: str
    dim: int | None = 64
    whiten: str = "anchor"  # anchor | flip | half | on | off
    var: float | None = None  # pca_var only
    standardize: bool = True  # False: centre only, no per-coordinate scaling
    recenter: bool = False  # subtract each recording's mean window vector before everything (transductive, label-free)
    sfa_pre: int = 128
    solver: str = "randomized"  # pca only: randomized | full | arpack
    n_landmarks: int = 2000  # kpca only
    label: str = ""

    def alpha(self, anchor_whiten: bool) -> float:
        if self.method == "identity":
            return 0.0
        if self.method == "sfa":
            return 1.0
        return {"anchor": float(anchor_whiten), "flip": float(not anchor_whiten), "half": 0.5, "on": 1.0,
                "off": 0.0}[self.whiten]


VARIANTS: tuple[Variant, ...] = (
    # M -- why PCA: the anchor against no reduction, an uninformed linear reduction and a temporal criterion
    Variant("base", "M", "pca", 64, label="PCA-64, randomized SVD (anchor)"),
    Variant("identity", "M", "identity", None, label="no reduction"),
    Variant("random64", "M", "random", 64, label="random projection-64"),
    Variant("sfa64", "M", "sfa", 64, whiten="on", label="slow features-64 (whitened)"),
    Variant("kpca64", "M", "kpca", 64, label="kernel PCA-64 (Nystrom RBF)"),
    # S -- why randomized SVD: decoded only if `solver_check` shows the tokens differ
    Variant("pca_full", "S", "pca", 64, solver="full", label="PCA-64, exact (full) SVD"),
    # X -- not in the default run (declared, so a later run can add them without a new identity)
    Variant("whiten_flip", "X", "pca", 64, whiten="flip", label="PCA-64, whitening toggled"),
    Variant("pca16", "X", "pca", 16, label="PCA-16"),
    Variant("pca32", "X", "pca", 32, label="PCA-32"),
    Variant("pca128", "X", "pca", 128, label="PCA-128"),
    Variant("wrpca64", "X", "wrpca", 64, label="within-recording PCA-64"),
    Variant("rc_pca64", "X", "pca", 64, recenter=True, label="recording-centred data, PCA-64"),
    Variant("nostd_pca64", "X", "pca", 64, standardize=False, label="no per-coordinate scaling, PCA-64"),
    Variant("pca8", "X", "pca", 8, label="PCA-8"),
    Variant("pca256", "X", "pca", 256, label="PCA-256"),
    Variant("var95", "X", "pca_var", None, var=0.95, label="PCA, 95% variance"),
    Variant("whiten_half", "X", "pca", 64, whiten="half", label="PCA-64, half whitening"),
)
BY_KEY = {v.key: v for v in VARIANTS}
STAGES = ("M", "S", "X")
DEFAULT_STAGES = ("M",)
assert len(BY_KEY) == len(VARIANTS)


@dataclass
class Reducer:
    method: str
    mean_: np.ndarray  # (D,)
    components_: np.ndarray | None  # (d, D); None for identity
    divisor_: np.ndarray | None  # (d,); None = 1
    sk_pca: PCA | None = None  # exact sklearn path (plain PCA, alpha 0 or 1)
    kernel: dict | None = None  # kpca: landmarks, sigma, centring statistics, projection
    info: dict = field(default_factory=dict)

    @property
    def out_dim(self) -> int:
        if self.sk_pca is not None:
            return int(self.sk_pca.n_components_)
        if self.kernel is not None:
            return int(self.kernel["alphas"].shape[1])
        return int(self.mean_.shape[0] if self.components_ is None else self.components_.shape[0])

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.sk_pca is not None:
            return self.sk_pca.transform(x)
        if self.kernel is not None:
            return _kpca_transform(self.kernel, x)
        if self.components_ is None:
            return x
        z = (np.asarray(x, dtype=np.float64) - self.mean_) @ self.components_.T
        return z / self.divisor_ if self.divisor_ is not None else z

    def basis(self) -> np.ndarray:
        """(d, D) rows spanning the kept subspace (in standardized coordinates)."""
        if self.sk_pca is not None:
            return self.sk_pca.components_.astype(np.float64)
        if self.kernel is not None:
            raise ValueError("a kernel reducer has no linear basis")
        if self.components_ is None:
            return np.eye(self.mean_.shape[0])
        return self.components_


def _sd(z: np.ndarray) -> np.ndarray:
    sd = z.std(axis=0, ddof=1)
    return np.where(sd > 1e-12, sd, 1.0)


def _pca(sample_std: np.ndarray, dim: int, alpha: float, seed: int,
         solver: str = "randomized") -> tuple[Reducer, np.ndarray]:
    n_components = min(dim, sample_std.shape[0], sample_std.shape[1])
    if solver == "arpack":
        n_components = min(n_components, min(sample_std.shape) - 1)
    if alpha in (0.0, 1.0):
        # line for line `tokenizers.fit_population`: K-means is fitted on fit_transform's output
        pca = PCA(n_components=n_components, whiten=bool(alpha), svd_solver=solver, random_state=seed)
        out = pca.fit_transform(sample_std)
        return Reducer("pca", pca.mean_, None, None, sk_pca=pca), out
    pca = PCA(n_components=n_components, whiten=False, svd_solver=solver, random_state=seed).fit(sample_std)
    red = Reducer("pca", pca.mean_.astype(np.float64), pca.components_.astype(np.float64),
                  np.power(pca.explained_variance_.astype(np.float64), alpha / 2.0))
    return red, red.transform(sample_std)


def _pca_var(sample_std: np.ndarray, var: float, alpha: float) -> tuple[Reducer, np.ndarray]:
    pca = PCA(svd_solver="full").fit(sample_std)
    cum = np.cumsum(pca.explained_variance_ratio_)
    d = int(min(np.searchsorted(cum, var) + 1, len(cum)))
    div = np.power(pca.explained_variance_[:d].astype(np.float64), alpha / 2.0) if alpha else None
    red = Reducer("pca_var", pca.mean_.astype(np.float64), pca.components_[:d].astype(np.float64), div,
                  info=dict(chosen_dim=d, variance_target=var, cumulative_at_dim=float(cum[d - 1])))
    return red, red.transform(sample_std)


def _random(sample_std: np.ndarray, dim: int, alpha: float, seed: int) -> tuple[Reducer, np.ndarray]:
    D = sample_std.shape[1]
    R = np.random.default_rng(seed).standard_normal((dim, D)) / np.sqrt(dim)
    red = Reducer("random", np.zeros(D), R, None)
    z = red.transform(sample_std)
    if alpha:
        red.divisor_ = np.power(_sd(z), alpha)
        z = z / red.divisor_
    return red, z


def _wrpca(sample_std: np.ndarray, sample_rec_means: np.ndarray, dim: int, alpha: float,
           seed: int) -> tuple[Reducer, np.ndarray]:
    n_components = min(dim, sample_std.shape[0], sample_std.shape[1])
    within = sample_std - sample_rec_means
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit(within)
    red = Reducer("wrpca", sample_std.mean(axis=0).astype(np.float64), pca.components_.astype(np.float64), None)
    z = red.transform(sample_std)
    # share of each kept axis's (uncentred) variance that is between-recording
    between = (sample_rec_means - red.mean_) @ red.components_.T
    red.info = dict(between_recording_share=float(between.var(axis=0).sum() / max(z.var(axis=0).sum(), 1e-12)),
                    within_explained_variance_ratio=float(pca.explained_variance_ratio_.sum()))
    if alpha:
        red.divisor_ = np.power(_sd(z), alpha)
        z = z / red.divisor_
    return red, z


def _sfa(sample_std: np.ndarray, pairs_prev: np.ndarray, pairs_next: np.ndarray, dim: int, pre: int,
         seed: int) -> tuple[Reducer, np.ndarray]:
    """`pairs_prev` / `pairs_next`: standardized vectors of adjacent windows (p, p+1) of one recording."""
    if len(pairs_prev) < 2 * dim:
        raise RuntimeError(f"SFA needs >= {2 * dim} adjacent window pairs, found {len(pairs_prev)}")
    d0 = min(max(pre, dim), sample_std.shape[0] - 1, sample_std.shape[1])
    pca = PCA(n_components=d0, whiten=True, svd_solver="randomized", random_state=seed).fit(sample_std)
    delta = pca.transform(pairs_next).astype(np.float64) - pca.transform(pairs_prev).astype(np.float64)
    cov = delta.T @ delta / len(delta)
    evals, evecs = np.linalg.eigh(cov)  # ascending: slowest first
    V = evecs[:, :dim]
    pre_map = pca.components_.astype(np.float64) / np.sqrt(pca.explained_variance_.astype(np.float64))[:, None]
    red = Reducer("sfa", pca.mean_.astype(np.float64), V.T @ pre_map, None,
                  info=dict(pre_dim=int(d0), n_pairs=int(len(delta)),
                            slowness_kept=[float(evals[0]), float(evals[dim - 1])],
                            slowness_dropped_max=float(evals[-1])))
    return red, red.transform(sample_std)


def _rbf(x: np.ndarray, landmarks: np.ndarray, lm_sq: np.ndarray, sigma: float) -> np.ndarray:
    d2 = (x * x).sum(1)[:, None] + lm_sq[None, :] - 2.0 * (x @ landmarks.T)
    return np.exp(-np.maximum(d2, 0.0) / (2.0 * sigma ** 2))


def _kpca_transform(k: dict, x: np.ndarray, chunk: int = 8192) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty((len(x), k["alphas"].shape[1]), dtype=np.float64)
    for s in range(0, len(x), chunk):
        kr = _rbf(x[s:s + chunk], k["landmarks"], k["lm_sq"], k["sigma"])
        kc = kr - kr.mean(axis=1, keepdims=True) - k["col_mean"][None, :] + k["all_mean"]
        out[s:s + chunk] = (kc @ k["alphas"]) / k["divisor"]
    return out


def _kpca(sample_std: np.ndarray, dim: int, n_landmarks: int, alpha: float, seed: int) -> tuple[Reducer, np.ndarray]:
    rng = np.random.default_rng(seed)
    m = min(n_landmarks, len(sample_std))
    lm = np.asarray(sample_std, dtype=np.float64)[rng.choice(len(sample_std), size=m, replace=False)]
    lm_sq = (lm * lm).sum(1)
    d2 = np.maximum(lm_sq[:, None] + lm_sq[None, :] - 2.0 * (lm @ lm.T), 0.0)
    sigma = float(np.sqrt(np.median(d2[np.triu_indices(m, 1)])))  # median pairwise distance
    K = np.exp(-d2 / (2.0 * sigma ** 2))
    col_mean, all_mean = K.mean(axis=0), float(K.mean())
    Kc = K - col_mean[None, :] - K.mean(axis=1, keepdims=True) + all_mean
    evals, evecs = np.linalg.eigh(Kc)
    order = np.argsort(evals)[::-1]
    evals, evecs = np.maximum(evals[order], 1e-12), evecs[:, order]
    d = min(dim, m - 1)
    var = evals[:d] / m  # variance of each component over the landmarks (plain PCA's eigenvalue analogue)
    # unwhitened axis j has variance lambda_j / m over the landmarks (plain PCA scaling); alpha > 0 whitens
    kernel = dict(landmarks=lm, lm_sq=lm_sq, sigma=sigma, col_mean=col_mean, all_mean=all_mean,
                  alphas=evecs[:, :d] / np.sqrt(evals[:d])[None, :], divisor=np.power(var, alpha / 2.0))
    red = Reducer("kpca", np.zeros(sample_std.shape[1]), None, None, kernel=kernel,
                  info=dict(sigma=sigma, n_landmarks=int(m), spectrum_share=float(evals[:d].sum() / evals.sum())))
    return red, red.transform(sample_std)


def fit_reducer(variant: Variant, alpha: float, sample_std: np.ndarray, seed: int, *,
                sample_rec_means: np.ndarray | None = None, pairs_prev: np.ndarray | None = None,
                pairs_next: np.ndarray | None = None) -> tuple[Reducer, np.ndarray]:
    """Returns the fitted reducer and its output on the fitting sample (what K-means is fitted on)."""
    m = variant.method
    if m == "pca":
        return _pca(sample_std, variant.dim, alpha, seed, variant.solver)
    if m == "pca_var":
        return _pca_var(sample_std, variant.var, alpha)
    if m == "kpca":
        return _kpca(sample_std, variant.dim, variant.n_landmarks, alpha, seed)
    if m == "identity":
        return Reducer("identity", np.zeros(sample_std.shape[1]), None, None), sample_std
    if m == "random":
        return _random(sample_std, variant.dim, alpha, seed)
    if m == "wrpca":
        if sample_rec_means is None:
            raise ValueError("wrpca needs the recording mean of every fitting window")
        return _wrpca(sample_std, sample_rec_means, variant.dim, alpha, seed)
    if m == "sfa":
        if pairs_prev is None or pairs_next is None:
            raise ValueError("sfa needs adjacent window pairs")
        return _sfa(sample_std, pairs_prev, pairs_next, variant.dim, variant.sfa_pre, seed)
    raise ValueError(m)


def retained_variance(reducer: Reducer, sample_std: np.ndarray) -> float:
    """Share of the standardized sample's total variance lying in the kept subspace (scaling ignored)."""
    if reducer.kernel is not None:  # share of the centred kernel spectrum on the landmarks, not of input variance
        return float(reducer.info["spectrum_share"])
    B = reducer.basis()
    if B.shape[0] >= B.shape[1]:
        return 1.0
    Q, _ = np.linalg.qr(B.T)  # (D, d) orthonormal basis of the row space
    xc = sample_std - sample_std.mean(axis=0)
    return float(((xc @ Q).var(axis=0).sum()) / max(xc.var(axis=0).sum(), 1e-12))
