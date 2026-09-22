"""
Shared segment representation.

For a segment of L frames with frame-level (already standardised) features z[0:L]:
  1. its normalised temporal extent [0,1] is divided into B=8 equal bins;
  2. L >= B : each bin is the exact area-average of the piecewise-constant frame signal
              over the bin's (possibly fractional) frame extent, computed from prefix sums
              (equals the plain block mean when B divides L);
     L <  B : each bin is the linear interpolation of the frame values (placed at frame
              centres) at the bin centre; clamped at the ends. L = 1 gives a constant vector;
  3. the B bins are flattened in temporal order into one B*D vector.
Deterministic; no learned parameters and no labels.
"""
import numpy as np

B_DEFAULT = 8


def fit_scaler(frame_arrays):
    """Per-dimension mean/std of FRAME-level latents over the given (fitting) recordings."""
    n = 0
    s = None
    s2 = None
    for z in frame_arrays:
        z = np.asarray(z, dtype=np.float64)
        if s is None:
            s = np.zeros(z.shape[1]); s2 = np.zeros(z.shape[1])
        s += z.sum(0); s2 += (z ** 2).sum(0); n += len(z)
    mu = s / n
    var = np.maximum(s2 / n - mu ** 2, 0.0)
    sd = np.sqrt(var)
    floor = 1e-3 * (np.median(sd[sd > 0]) if np.any(sd > 0) else 1.0)   # guards constant dims
    return mu.astype(np.float32), np.maximum(sd, floor).astype(np.float32)


def bin_segments(z, edges, B=B_DEFAULT):
    """z: (T, D) standardised frames; edges: partition. Returns (n_seg, B, D) float32."""
    z = np.asarray(z, dtype=np.float64)
    T, D = z.shape
    n = len(edges) - 1
    C = np.concatenate([np.zeros((1, D)), np.cumsum(z, 0)], 0)            # C[t] = sum_{u<t} z[u]

    def cint(x):                                                            # integral of piecewise-const signal on [0,x]
        i = np.minimum(np.floor(x).astype(np.int64), T - 1)
        i = np.maximum(i, 0)
        frac = x - i
        return C[i] + frac[..., None] * z[i]

    out = np.empty((n, B, D), dtype=np.float64)
    a = edges[:-1].astype(np.float64)
    L = np.diff(edges).astype(np.float64)
    long = L >= B
    if long.any():
        idx = np.flatnonzero(long)
        x0 = a[idx, None] + (np.arange(B)[None, :] * L[idx, None] / B)
        x1 = a[idx, None] + ((np.arange(B)[None, :] + 1) * L[idx, None] / B)
        # exact end at T when x1 hits the segment end (avoid floor overflow)
        m = (cint(x1.ravel()) - cint(x0.ravel())) / (x1.ravel() - x0.ravel())[:, None]
        out[idx] = m.reshape(len(idx), B, D)
    for i in np.flatnonzero(~long):
        Li = int(L[i]); ai = int(a[i])
        seg = z[ai:ai + Li]
        t = np.clip((np.arange(B) + 0.5) * Li / B - 0.5, 0, Li - 1)      # position in frame-centre coordinates
        i0 = np.floor(t).astype(np.int64)
        i1 = np.minimum(i0 + 1, Li - 1)
        w = (t - i0)[:, None]
        out[i] = seg[i0] * (1 - w) + seg[i1] * w
    return out.astype(np.float32)


def segment_vectors(z_raw, edges, mu, sd, B=B_DEFAULT):
    z = (np.asarray(z_raw, dtype=np.float32) - mu) / sd
    return bin_segments(z, edges, B).reshape(len(edges) - 1, -1)
