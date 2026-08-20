"""
State-space learning-curve models for per-animal plateau detection.

Step 1: univariate Gaussian local level model (Smith et al. 2004 adapted to
continuous observations, i.e. the classical local level / random-walk-plus-noise
model), fitted by EM.

    x_k = x_{k-1} + eps_k,   eps_k ~ N(0, sigma2_eps)      state (smoothness prior)
    y_k = x_k + v_k,         v_k   ~ N(0, sigma2_v / n_k)  observation

`x_k` is the latent per-session performance level on the transformed metric
scale; `n_k` is the session's trial count, so a session with more trials gets a
more precise observation. `sigma2_v` is therefore the observation variance of a
session with a typical number of trials.

The intercept `mu` of the design doc is absorbed into `x`, with a diffuse prior
on x_1 -- carrying both a free `mu` and a free state level would be redundant.

Everything here is pure NumPy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.behavior_metrics import load_metrics_table, PHASES


# --- metric transforms -----------------------------------------------------
# `sign` records which direction on the RAW metric means "better at the task".
# The latent state is left on the metric's own transformed scale so that Step 1
# can be eyeballed against the existing four-panel figures; `sign` is what the
# Step 3 criterion will use to decide which direction counts as improvement.

METRIC_TRANSFORMS = {
    'reentry_index': dict(forward=np.log,   inverse=np.exp, name='log', sign=-1),
    'consumption':   dict(forward=np.log,   inverse=np.exp, name='log', sign=-1),
    'pct_engaged':   dict(forward=None,     inverse=None,   name='logit', sign=+1),
    'realized_rr':   dict(forward=np.log,   inverse=np.exp, name='log', sign=+1),
}


def _logit(p):
    return np.log(p / (1.0 - p))


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


METRIC_TRANSFORMS['pct_engaged']['forward'] = _logit
METRIC_TRANSFORMS['pct_engaged']['inverse'] = _expit

# pct_engaged includes a travel-time penalty in its numerator and can exceed 1,
# which the logit cannot take. Clip into the open unit interval before transform.
LOGIT_CLIP = 1e-4


def transform_metric(values, metric):
    """Applies the variance-stabilising transform for `metric`, NaN-safe."""
    values = np.asarray(values, dtype=float)
    spec = METRIC_TRANSFORMS[metric]

    with np.errstate(divide='ignore', invalid='ignore'):
        if spec['name'] == 'logit':
            clipped = np.clip(values, LOGIT_CLIP, 1.0 - LOGIT_CLIP)
            out = spec['forward'](clipped)
        else:
            out = np.where(values > 0,
                           spec['forward'](np.where(values > 0, values, np.nan)),
                           np.nan)

    return np.where(np.isfinite(out), out, np.nan)


# --- reduce the per-phase table to one observation per session -------------

def pool_phases(metrics_df, metric):
    """
    Collapses the two phase rows of each session into a single per-session value.

    For the two ratio metrics the pooling is done on the underlying counts, which
    is the trial-weighted mean rather than the unweighted mean of the two blocks:

        reentry_index : total context exits / total trials
        realized_rr   : total rewards / total engaged time

    `consumption` and `pct_engaged` are averaged with n_trials as weights.

    Returns one row per (animal_id, session_idx) with columns
    animal_id, session_idx, session_date, session_type, <metric>, n_trials.
    """
    df = metrics_df.copy()
    df['_w'] = df['n_trials'].astype(float)

    if metric == 'reentry_index':
        # exits_j = reentry_index_j * n_trials_j
        df['_num'] = df['reentry_index'] * df['n_trials']
        df['_den'] = df['n_trials'].astype(float)
    elif metric == 'realized_rr':
        # engaged_j = n_rewards_j / realized_rr_j
        with np.errstate(divide='ignore', invalid='ignore'):
            df['_num'] = df['n_rewards'].astype(float)
            df['_den'] = np.where(df['realized_rr'] > 0,
                                  df['n_rewards'] / df['realized_rr'], np.nan)
    else:
        df['_num'] = df[metric] * df['_w']
        df['_den'] = np.where(df[metric].notna(), df['_w'], np.nan)

    # A phase row with a NaN metric contributes to neither numerator nor denominator.
    bad = df['_num'].isna() | df['_den'].isna()
    df.loc[bad, ['_num', '_den']] = np.nan

    grouped = df.groupby(['animal_id', 'session_idx'], sort=True)
    pooled = grouped.agg(
        session_date=('session_date', 'first'),
        session_type=('session_type', 'first'),
        _num=('_num', 'sum'),
        _den=('_den', 'sum'),
        n_trials=('n_trials', 'sum'),
    ).reset_index()

    with np.errstate(divide='ignore', invalid='ignore'):
        pooled[metric] = np.where(pooled['_den'] > 0,
                                  pooled['_num'] / pooled['_den'], np.nan)

    return pooled.drop(columns=['_num', '_den'])


def prepare_series(metrics_df, animal_id, metric='reentry_index',
                   session_type='pre-surgery'):
    """
    Builds the observation vector for one animal, one metric, one segment.

    Returns a dict with:
        session_idx : (K,) int    -- pipeline-aligned session index
        raw         : (K,) float  -- pooled metric on its natural scale
        y           : (K,) float  -- transformed observation (NaN = missing)
        n_trials    : (K,) float  -- trials per session, the observation weight
        metric, session_type, animal_id
    """
    animal_df = metrics_df[metrics_df['animal_id'] == animal_id]
    if session_type is not None:
        animal_df = animal_df[animal_df['session_type'] == session_type]

    pooled = pool_phases(animal_df, metric).sort_values('session_idx')

    return dict(
        animal_id=animal_id,
        metric=metric,
        session_type=session_type,
        session_idx=pooled['session_idx'].to_numpy(dtype=int),
        session_date=pooled['session_date'].to_numpy(),
        raw=pooled[metric].to_numpy(dtype=float),
        y=transform_metric(pooled[metric].to_numpy(dtype=float), metric),
        n_trials=pooled['n_trials'].to_numpy(dtype=float),
    )


# --- Kalman filter + RTS smoother ------------------------------------------

def kalman_filter_smoother(y, w, sigma2_eps, sigma2_v, m0, P0):
    """
    Forward filter and fixed-interval (RTS) smoother for the local level model
    with per-observation noise scaling R_k = sigma2_v / w_k.

    NaN entries of `y` are treated as missing: the update step is skipped and
    the prediction is carried forward, so the state is still interpolated there.

    Returns filtered and smoothed means and variances, the lag-one smoothed
    covariances Cov(x_{k+1}, x_k | y_{1:K}), and the log-likelihood.

    The lag-one covariances are what the slope-based plateau criterion needs:
        Var(x_{k+1} - x_k | Y) = P_smooth[k+1] + P_smooth[k] - 2*lag_one_cov[k]
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    K = len(y)
    observed = ~np.isnan(y)

    x_pred = np.zeros(K); P_pred = np.zeros(K)
    x_filt = np.zeros(K); P_filt = np.zeros(K)
    loglik = 0.0

    # --- forward pass ---
    for k in range(K):
        if k == 0:
            x_pred[k] = m0
            P_pred[k] = P0
        else:
            x_pred[k] = x_filt[k - 1]
            P_pred[k] = P_filt[k - 1] + sigma2_eps

        if observed[k]:
            R_k = sigma2_v / w[k]
            S_k = P_pred[k] + R_k          # innovation variance
            e_k = y[k] - x_pred[k]         # innovation
            G_k = P_pred[k] / S_k          # Kalman gain
            x_filt[k] = x_pred[k] + G_k * e_k
            P_filt[k] = (1.0 - G_k) * P_pred[k]
            loglik += -0.5 * (np.log(2.0 * np.pi * S_k) + e_k ** 2 / S_k)
        else:
            x_filt[k] = x_pred[k]
            P_filt[k] = P_pred[k]

    # --- backward pass ---
    x_smooth = np.zeros(K); P_smooth = np.zeros(K)
    A = np.zeros(max(K - 1, 0))

    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for k in range(K - 2, -1, -1):
        A[k] = P_filt[k] / P_pred[k + 1]
        x_smooth[k] = x_filt[k] + A[k] * (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + A[k] ** 2 * (P_smooth[k + 1] - P_pred[k + 1])

    # Cov(x_k, x_{k+1} | Y) = A_k * P_smooth[k+1]
    lag_one_cov = A * P_smooth[1:] if K > 1 else np.zeros(0)

    return dict(x_pred=x_pred, P_pred=P_pred,
                x_filt=x_filt, P_filt=P_filt,
                x_smooth=x_smooth, P_smooth=P_smooth,
                lag_one_cov=lag_one_cov,
                loglik=loglik, observed=observed)


# --- EM --------------------------------------------------------------------

def fit_local_level_em(y, w=None, max_iter=1000, tol=1e-8, var_floor=1e-10,
                       sigma2_eps_init=None, sigma2_v_init=None,
                       P0_scale=1e6, verbose=False):
    """
    Maximum-likelihood estimation of (sigma2_eps, sigma2_v) by EM.

    Both M-steps are closed form for the Gaussian observation model:

        sigma2_eps = (1/(K-1)) sum_k E[(x_k - x_{k-1})^2 | Y]
        sigma2_v   = (1/K_obs) sum_k w_k E[(y_k - x_k)^2 | Y]

    Initialisation is diffuse (m0 = first observation, P0 large), so x_1 is
    effectively determined by the data rather than by a prior. One consequence:
    the reported log-likelihood carries a large constant offset from the k=1
    term, so compare log-likelihoods only across fits with the same P0_scale.
    """
    y = np.asarray(y, dtype=float)
    K = len(y)

    if w is None:
        w = np.ones(K)
    w = np.asarray(w, dtype=float)
    w = w / np.nanmean(w)      # normalised, so sigma2_v is "typical session" variance

    observed = ~np.isnan(y)
    if observed.sum() < 3:
        raise ValueError(f"Need at least 3 observed sessions, got {observed.sum()}.")
    y_obs = y[observed]

    if sigma2_v_init is None:
        sigma2_v_init = max(0.5 * np.var(y_obs), var_floor)
    if sigma2_eps_init is None:
        sigma2_eps_init = max(0.5 * np.var(np.diff(y_obs)), var_floor)

    sigma2_eps = float(sigma2_eps_init)
    sigma2_v = float(sigma2_v_init)

    m0 = float(y_obs[0])
    P0 = float(P0_scale * np.var(y_obs)) if np.var(y_obs) > 0 else float(P0_scale)

    loglik_history = []
    prev_ll = -np.inf
    converged = False

    for iteration in range(max_iter):
        # --- E-step ---
        s = kalman_filter_smoother(y, w, sigma2_eps, sigma2_v, m0, P0)
        ll = s['loglik']
        loglik_history.append(ll)

        x_s = s['x_smooth']; P_s = s['P_smooth']; C = s['lag_one_cov']

        # --- M-step ---
        # E[(x_k - x_{k-1})^2] = P_s[k] + x_s[k]^2 + P_s[k-1] + x_s[k-1]^2
        #                        - 2*(Cov(x_k,x_{k-1}) + x_s[k]*x_s[k-1])
        state_ss = np.sum(P_s[1:] + x_s[1:] ** 2 + P_s[:-1] + x_s[:-1] ** 2
                          - 2.0 * (C + x_s[1:] * x_s[:-1]))
        sigma2_eps = max(state_ss / (K - 1), var_floor)

        obs_ss = np.sum(w[observed] * ((y[observed] - x_s[observed]) ** 2 + P_s[observed]))
        sigma2_v = max(obs_ss / observed.sum(), var_floor)

        if iteration > 0 and abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
            converged = True
            break
        prev_ll = ll

    if verbose:
        print(f"  EM: {iteration + 1} iterations, converged={converged}, "
              f"sigma2_eps={sigma2_eps:.5f}, sigma2_v={sigma2_v:.5f}")
    if not converged:
        print(f"  ⚠️ EM hit max_iter={max_iter} without converging "
              f"(last dloglik={abs(ll - prev_ll):.2e}).")

    fit = kalman_filter_smoother(y, w, sigma2_eps, sigma2_v, m0, P0)
    fit.update(sigma2_eps=sigma2_eps, sigma2_v=sigma2_v,
               m0=m0, P0=P0, w=w, y=y,
               n_iter=iteration + 1, converged=converged,
               loglik_history=np.array(loglik_history),
               signal_to_noise=sigma2_eps / sigma2_v)
    return fit


def fit_metric_series(series, **kwargs):
    """Convenience wrapper: fits the local level model to a `prepare_series` dict."""
    fit = fit_local_level_em(series['y'], w=series['n_trials'], **kwargs)
    fit['series'] = series
    return fit


def smoothed_band(fit, level=0.95):
    """Posterior mean and credible band of the latent state, on the y scale."""
    z = 1.959963984540054 if level == 0.95 else abs(_norm_ppf((1 - level) / 2))
    sd = np.sqrt(fit['P_smooth'])
    return fit['x_smooth'], fit['x_smooth'] - z * sd, fit['x_smooth'] + z * sd


def _norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    from math import sqrt, log
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = sqrt(-2 * log(p))
        return ((((( c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = sqrt(-2 * log(1 - p))
        return -((((( c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# --- plotting --------------------------------------------------------------

def plot_local_level_fit(fit, axes=None, level=0.95, bin_size=5, color='tab:purple'):
    """
    Two panels for eyeballing the fit against the existing four-panel figures:
      top    -- transformed scale: observations, smoothed state, credible band
      bottom -- raw metric scale: observations, back-transformed state and band,
                plus the old `bin_size`-session binned means for comparison.
    """
    import matplotlib.pyplot as plt

    series = fit['series']
    metric = series['metric']
    spec = METRIC_TRANSFORMS[metric]
    k = series['session_idx']

    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    x_s, lo, hi = smoothed_band(fit, level=level)

    # --- transformed scale ---
    ax = axes[0]
    ax.plot(k, series['y'], 'o', ms=4, color='0.6', alpha=0.8, label='observed')
    ax.plot(k, x_s, '-', lw=2, color=color, label='smoothed state')
    ax.fill_between(k, lo, hi, color=color, alpha=0.2,
                    label=f'{int(level*100)}% credible band')
    ax.set_ylabel(f"{spec['name']}({metric})")
    ax.legend(fontsize='x-small', loc='best')
    ax.set_title(f"{series['animal_id']} — {metric} — {series['session_type']}  "
                 f"(sigma2_eps={fit['sigma2_eps']:.4f}, sigma2_v={fit['sigma2_v']:.4f}, "
                 f"SNR={fit['signal_to_noise']:.3f})", fontsize=9)

    # --- raw scale ---
    ax = axes[1]
    ax.plot(k, series['raw'], 'o', ms=4, color='0.6', alpha=0.8, label='observed')
    ax.plot(k, spec['inverse'](x_s), '-', lw=2, color=color, label='smoothed state')
    ax.fill_between(k, spec['inverse'](lo), spec['inverse'](hi), color=color, alpha=0.2)

    # old-style binned means, for direct comparison with the existing panels
    rank = np.arange(len(k))
    binned = pd.DataFrame({'group': rank // bin_size, 'k': k, 'raw': series['raw']})
    gm = binned.groupby('group').agg(k=('k', 'mean'), raw=('raw', 'mean'))
    ax.plot(gm['k'], gm['raw'], 's--', ms=5, color='tab:orange', alpha=0.9,
            label=f'{bin_size}-session bins (old)')

    ax.set_ylabel(metric)
    ax.set_xlabel('session_idx')
    ax.legend(fontsize='x-small', loc='best')

    for a in axes:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    return axes


# ===========================================================================
# STEP 2 -- Multivariate dynamic factor model
#
#     y_{k,j} = c_j * x_k + v_{k,j},   v_{k,j} ~ N(0, sigma2_j / n_k)
#     x_k     = x_{k-1} + eps_k,       eps_k   ~ N(0, sigma2_eps)
#
# One shared latent "task expertise" state x_k drives all J channels. Each
# channel gets its own loading c_j and its own noise variance sigma2_j, so a
# noisy channel is automatically down-weighted instead of being averaged in.
#
# Identifiability
# ---------------
# Two redundancies have to be removed:
#   * Scale/sign: (x, c) and (a*x, c/a) are indistinguishable. Fixed by pinning
#     the anchor channel's loading. We pin it to METRIC_TRANSFORMS[anchor]['sign'],
#     i.e. c_reentry = -1 rather than +1. This is the same one-parameter
#     constraint the design doc calls for, but it also orients the state so that
#     INCREASING x means IMPROVING, which is what the Step 3 criterion
#     Pr(x_{k+1} > x_k) needs to read literally. The doc's expected-sign check is
#     unchanged: c_reentry < 0, c_consumption < 0, c_pct_engaged > 0.
#   * Level: (x + a, mu_j - c_j*a) is indistinguishable if every mu_j is free.
#     Removed by z-scoring each channel and fixing mu_j = 0 -- which is exactly
#     what the z-scoring in the design doc is for. The absolute level of x is
#     arbitrary anyway; both plateau criteria depend only on differences of x.
# ===========================================================================

DEFAULT_CHANNELS = ['reentry_index', 'consumption', 'pct_engaged']

# Held back from the model on purpose: realized reward rate is a partial
# function of investment-port leave time, which is what the RL pipeline is
# scored on. Used only by validate_against_heldout().
HELDOUT_CHANNEL = 'realized_rr'


def prepare_multivariate_series(metrics_df, animal_id, channels=None,
                                session_type='pre-surgery',
                                zscore_scope='pre-surgery'):
    """
    Builds the (K, J) observation matrix for one animal, one segment.

    Each channel is transformed (log / logit) and then z-scored.

    `zscore_scope` decides which sessions the z-scoring mean and sd come from:
        'pre-surgery' (default) -- always the animal's pre-surgery sessions.
          Post-surgery states are then on the SAME scale as pre-surgery ones, so
          the Step 4 figure can show whether the animal came back below its own
          pre-surgery asymptote. This is what makes the surgery-gap comparison
          meaningful.
        'segment' -- the fitted segment only. Each segment is self-contained and
          the two are not comparable in level.
        'all'     -- every session of that animal.

    Returns a dict with Y (K, J), n_trials (K,), session_idx (K,), channels,
    plus the z-scoring stats so raw values can be recovered.
    """
    channels = list(DEFAULT_CHANNELS if channels is None else channels)
    animal_df = metrics_df[metrics_df['animal_id'] == animal_id]

    if zscore_scope == 'pre-surgery':
        ref_df = animal_df[animal_df['session_type'] == 'pre-surgery']
    elif zscore_scope == 'segment':
        ref_df = animal_df[animal_df['session_type'] == session_type]
    elif zscore_scope == 'all':
        ref_df = animal_df
    else:
        raise ValueError(f"Unknown zscore_scope: {zscore_scope}")

    if ref_df.empty:
        raise ValueError(f"No reference sessions for {animal_id} "
                         f"(zscore_scope={zscore_scope}).")

    per_channel = {}
    z_mean = {}
    z_sd = {}

    for metric in channels:
        # z-scoring stats come from the reference scope...
        ref_y = transform_metric(pool_phases(ref_df, metric)[metric].to_numpy(float), metric)
        z_mean[metric] = float(np.nanmean(ref_y))
        z_sd[metric] = float(np.nanstd(ref_y))
        if not np.isfinite(z_sd[metric]) or z_sd[metric] <= 0:
            raise ValueError(f"{animal_id}/{metric}: zero variance in reference scope.")

        # ...but the fitted series comes from the requested segment.
        per_channel[metric] = prepare_series(metrics_df, animal_id, metric=metric,
                                             session_type=session_type)

    session_idx = per_channel[channels[0]]['session_idx']
    for metric in channels[1:]:
        if not np.array_equal(per_channel[metric]['session_idx'], session_idx):
            raise ValueError("Channels disagree on session_idx -- check the metrics table.")

    Y = np.column_stack([(per_channel[m]['y'] - z_mean[m]) / z_sd[m] for m in channels])

    return dict(
        animal_id=animal_id,
        channels=channels,
        session_type=session_type,
        zscore_scope=zscore_scope,
        session_idx=session_idx,
        session_date=per_channel[channels[0]]['session_date'],
        n_trials=per_channel[channels[0]]['n_trials'],
        Y=Y,
        raw={m: per_channel[m]['raw'] for m in channels},
        z_mean=z_mean, z_sd=z_sd,
    )


def dfm_filter_smoother(Y, w, c, sigma2, sigma2_eps, m0, P0):
    """
    Kalman filter + RTS smoother for the one-factor dynamic factor model.

    The state is scalar, so the vector update is done in information form:

        1 / P_filt = 1 / P_pred + sum_j c_j^2 / R_{k,j}
        x_filt     = x_pred + P_filt * sum_j c_j * e_{k,j} / R_{k,j}

    which is O(J) per session and handles per-channel missingness for free --
    a NaN in channel j at session k simply drops out of both sums. The
    log-likelihood uses the matrix determinant lemma on the rank-one update, so
    no J x J matrix is ever formed or inverted.
    """
    Y = np.asarray(Y, dtype=float)
    w = np.asarray(w, dtype=float)
    c = np.asarray(c, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)

    K, J = Y.shape
    observed = ~np.isnan(Y)

    x_pred = np.zeros(K); P_pred = np.zeros(K)
    x_filt = np.zeros(K); P_filt = np.zeros(K)
    loglik = 0.0

    for k in range(K):
        if k == 0:
            x_pred[k] = m0
            P_pred[k] = P0
        else:
            x_pred[k] = x_filt[k - 1]
            P_pred[k] = P_filt[k - 1] + sigma2_eps

        o = observed[k]
        if o.any():
            R_k = sigma2[o] / w[k]                 # (m,) observation variances
            c_k = c[o]
            e_k = Y[k, o] - c_k * x_pred[k]        # innovation

            F = np.sum(c_k ** 2 / R_k)             # c' R^-1 c   (information)
            g = np.sum(c_k * e_k / R_k)            # c' R^-1 e
            denom = 1.0 + P_pred[k] * F

            # log N(e; 0, c c' P_pred + R) via determinant lemma / Woodbury
            quad = np.sum(e_k ** 2 / R_k) - P_pred[k] * g ** 2 / denom
            logdet = np.sum(np.log(R_k)) + np.log(denom)
            loglik += -0.5 * (o.sum() * np.log(2.0 * np.pi) + logdet + quad)

            P_filt[k] = P_pred[k] / denom
            x_filt[k] = x_pred[k] + P_filt[k] * g
        else:
            x_filt[k] = x_pred[k]
            P_filt[k] = P_pred[k]

    x_smooth = np.zeros(K); P_smooth = np.zeros(K)
    A = np.zeros(max(K - 1, 0))
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for k in range(K - 2, -1, -1):
        A[k] = P_filt[k] / P_pred[k + 1]
        x_smooth[k] = x_filt[k] + A[k] * (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + A[k] ** 2 * (P_smooth[k + 1] - P_pred[k + 1])

    lag_one_cov = A * P_smooth[1:] if K > 1 else np.zeros(0)

    return dict(x_pred=x_pred, P_pred=P_pred,
                x_filt=x_filt, P_filt=P_filt,
                x_smooth=x_smooth, P_smooth=P_smooth,
                lag_one_cov=lag_one_cov,
                loglik=loglik, observed=observed)


def fit_dynamic_factor_em(Y, w=None, anchor=0, anchor_loading=-1.0,
                          min_noise_share=0.05,          # <-- NEW
                          max_iter=2000, tol=1e-9, var_floor=1e-10,
                          P0=1e6, verbose=False):
    """
    EM for the one-factor dynamic factor model. All M-steps are closed form:

        sigma2_eps = (1/(K-1)) sum_k E[(x_k - x_{k-1})^2]
        c_j        = sum_k w_k y_kj E[x_k] / sum_k w_k E[x_k^2]      (j != anchor)
        sigma2_j   = (1/K_j) sum_k w_k E[(y_kj - c_j x_k)^2]

    `anchor_loading` is held fixed (identifiability); only its sigma2 is updated.
    """
    Y = np.asarray(Y, dtype=float)
    K, J = Y.shape

    if w is None:
        w = np.ones(K)
    w = np.asarray(w, dtype=float)
    w = w / np.nanmean(w)

    observed = ~np.isnan(Y)
    if observed[:, anchor].sum() < 3:
        raise ValueError("Anchor channel needs at least 3 observed sessions.")
    # --- Heywood guard -------------------------------------------------
    # The Gaussian likelihood is UNBOUNDED as any sigma2_j -> 0: the state
    # collapses onto y_j/c_j, that channel's density becomes a delta, and the
    # random walk absorbs the resulting path. EM will crawl up that ridge
    # forever (it shows up as max_iter with no convergence). Flooring each
    # channel's noise at a share of its own variance blocks the ridge and, on
    # simulated healthy data, leaves the fit bit-identical.
    channel_var = np.array([np.var(Y[observed[:, j], j]) for j in range(J)])
    noise_floor = np.maximum(min_noise_share * channel_var, var_floor)

    # --- initialisation: anchor channel implies a first guess at the state ---
    x_anchor = np.where(observed[:, anchor], Y[:, anchor] / anchor_loading, np.nan)
    good = ~np.isnan(x_anchor)
    x_init = np.interp(np.arange(K), np.arange(K)[good], x_anchor[good])

    c = np.zeros(J)
    sigma2 = np.zeros(J)
    for j in range(J):
        o_j = observed[:, j]
        if j == anchor:
            c[j] = anchor_loading
        else:
            c[j] = np.sum(Y[o_j, j] * x_init[o_j]) / np.sum(x_init[o_j] ** 2)
        sigma2[j] = max(0.5 * np.var(Y[o_j, j]), noise_floor[j])
    sigma2_eps = max(0.5 * np.var(np.diff(x_init)), var_floor)

    loglik_history = []
    prev_ll = -np.inf
    converged = False

    for iteration in range(max_iter):
        # --- E-step ---
        s = dfm_filter_smoother(Y, w, c, sigma2, sigma2_eps, 0.0, P0)
        ll = s['loglik']
        loglik_history.append(ll)

        x_s = s['x_smooth']; P_s = s['P_smooth']; C = s['lag_one_cov']
        Ex2 = P_s + x_s ** 2

        # --- M-step ---
        state_ss = np.sum(P_s[1:] + x_s[1:] ** 2 + P_s[:-1] + x_s[:-1] ** 2
                          - 2.0 * (C + x_s[1:] * x_s[:-1]))
        sigma2_eps = max(state_ss / (K - 1), var_floor)

        for j in range(J):
            o_j = observed[:, j]
            if j != anchor:
                c[j] = (np.sum(w[o_j] * Y[o_j, j] * x_s[o_j])
                        / np.sum(w[o_j] * Ex2[o_j]))
            obs_ss = np.sum(w[o_j] * (Y[o_j, j] ** 2
                                      - 2.0 * c[j] * Y[o_j, j] * x_s[o_j]
                                      + c[j] ** 2 * Ex2[o_j]))
            sigma2[j] = max(obs_ss / o_j.sum(), noise_floor[j])

        if iteration > 0 and abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
            converged = True
            break
        prev_ll = ll

    at_floor = sigma2 <= noise_floor * 1.0001
    info = c ** 2 / sigma2
    info_share = info / info.sum()

    if at_floor.any():
        print(f"  ⚠️ noise floor active on channel(s) {np.where(at_floor)[0].tolist()} "
              f"-- Heywood degeneracy was blocked, but treat this animal's fit as suspect.")
    if info_share.max() > 0.90:
        print(f"  ⚠️ channel {int(np.argmax(info_share))} carries "
              f"{info_share.max():.1%} of the information -- the 'shared' state is "
              f"effectively that one channel.")
    if not converged:
        print(f"  ⚠️ DFM EM hit max_iter={max_iter} without converging.")
    if verbose:
        print(f"  DFM EM: {iteration + 1} iterations, converged={converged}")

    fit = dfm_filter_smoother(Y, w, c, sigma2, sigma2_eps, 0.0, P0)
    fit.update(c=c, sigma2=sigma2, sigma2_eps=sigma2_eps,
               anchor=anchor, anchor_loading=anchor_loading,
               noise_floor=noise_floor, at_floor=at_floor,
               min_noise_share=min_noise_share,
               degenerate=bool(at_floor.any() or info_share.max() > 0.90),
               w=w, Y=Y, n_iter=iteration + 1, converged=converged,
               loglik_history=np.array(loglik_history),
               total_information=float(info.sum()))
    return fit


def fit_factor_model(mv_series, **kwargs):
    """Fits the DFM to a `prepare_multivariate_series` dict, anchoring on channel 0."""
    channels = mv_series['channels']
    anchor_loading = float(METRIC_TRANSFORMS[channels[0]]['sign'])
    fit = fit_dynamic_factor_em(mv_series['Y'], w=mv_series['n_trials'],
                                anchor=0, anchor_loading=anchor_loading, **kwargs)
    fit['series'] = mv_series
    return fit


# --- diagnostics -----------------------------------------------------------

def _spearman(a, b):
    """Spearman rho and a two-sided p-value from the t approximation."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 4:
        return np.nan, np.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    rho = np.corrcoef(ra, rb)[0, 1]
    if not np.isfinite(rho) or abs(rho) >= 1:
        return rho, 0.0
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    # two-sided p from the t distribution, via its relation to the normal for n>10
    p = 2.0 * (1.0 - _norm_cdf(abs(t)))
    return rho, p


def _norm_cdf(z):
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def factor_diagnostics(fit):
    """
    Per-channel table for checking the one-factor assumption.

        loading         c_j (anchor is fixed, not fitted)
        expected_sign   what the design doc predicts
        sign_ok         whether the fitted loading agrees
        sigma2_j        idiosyncratic noise variance
        var_explained   share of the channel's variance carried by the shared state
        info_share      c_j^2/sigma2_j as a fraction of the total -- how much this
                        channel actually contributes to locating the state
        resid_rho/_p    Spearman correlation of standardised residual with session.
                        A significant trend means the shared state does NOT capture
                        this channel's time course -- the one-factor assumption is
                        failing for it.
    """
    series = fit['series']
    Y = fit['Y']; x_s = fit['x_smooth']; w = fit['w']
    K, J = Y.shape
    rows = []

    for j, metric in enumerate(series['channels']):
        o_j = fit['observed'][:, j]
        resid = Y[:, j] - fit['c'][j] * x_s
        std_resid = resid / np.sqrt(fit['sigma2'][j] / w)
        rho, p = _spearman(series['session_idx'][o_j], std_resid[o_j])

        var_y = np.var(Y[o_j, j])
        # The shared state is constrained, so it must fit channel j NO BETTER
        # than a dedicated per-channel state does. sigma2_ratio well below 1 is
        # the signature of a Heywood collapse.
        solo_j = fit_local_level_em(Y[:, j], w=w)['sigma2_v']
        rows.append({
            'channel': metric,
            'loading': fit['c'][j],
            'expected_sign': METRIC_TRANSFORMS[metric]['sign'],
            'sign_ok': bool(np.sign(fit['c'][j]) == METRIC_TRANSFORMS[metric]['sign']),
            'is_anchor': j == fit['anchor'],
            'sigma2_j': fit['sigma2'][j],
            'var_explained': 1.0 - fit['sigma2'][j] / var_y if var_y > 0 else np.nan,
            'info_share': (fit['c'][j] ** 2 / fit['sigma2'][j]) / fit['total_information'],
            'resid_rho': rho,
            'resid_p': p,
            'n_obs': int(o_j.sum()),
            'sigma2_solo': solo_j,
            'sigma2_ratio': fit['sigma2'][j] / solo_j if solo_j > 0 else np.nan,
            'at_floor': bool(fit['at_floor'][j]),
        })

    return pd.DataFrame(rows)


def compare_to_univariate(fit, metrics_df):
    """
    Fits each channel on its own and compares its latent state to the shared one.

    If the one-factor assumption holds, every channel's solo state should be
    highly correlated with the shared state (after orienting by the loading sign).
    A channel whose solo state correlates weakly is telling you it plateaus on a
    different schedule, and the shared state is a compromise for it.
    """
    series = fit['series']
    x_shared = fit['x_smooth']
    rows = []

    for j, metric in enumerate(series['channels']):
        solo_series = prepare_series(metrics_df, series['animal_id'], metric=metric,
                                     session_type=series['session_type'])
        solo = fit_local_level_em(solo_series['y'], w=solo_series['n_trials'])
        # orient the solo state so "up = better", matching the shared state
        oriented = METRIC_TRANSFORMS[metric]['sign'] * solo['x_smooth']
        rows.append({
            'channel': metric,
            'corr_with_shared': float(np.corrcoef(oriented, x_shared)[0, 1]),
            'solo_sigma2_eps': solo['sigma2_eps'],
            'solo_sigma2_v': solo['sigma2_v'],
            'solo_snr': solo['signal_to_noise'],
        })

    return pd.DataFrame(rows)


def validate_against_heldout(fit, metrics_df, metric=HELDOUT_CHANNEL):
    """
    Checks the shared state against a channel that was deliberately kept OUT of
    the model. Realized reward rate is excluded because it is partly a function
    of investment-port leave time, which is what the RL pipeline is scored on --
    letting it define the plateau would select the test set for restricted range.

    A positive correlation here is convergent evidence that the shared state is
    tracking task expertise rather than an artefact of the three chosen channels.
    Nothing about this number feeds back into the fit.
    """
    series = fit['series']
    heldout = prepare_series(metrics_df, series['animal_id'], metric=metric,
                             session_type=series['session_type'])
    y = heldout['y']
    sign = METRIC_TRANSFORMS[metric]['sign']
    ok = np.isfinite(y)

    return dict(
        metric=metric,
        corr_with_shared=float(np.corrcoef(sign * y[ok], fit['x_smooth'][ok])[0, 1]),
        n_obs=int(ok.sum()),
        note='positive = held-out channel improves as the shared state rises',
    )


# --- plotting --------------------------------------------------------------

def plot_dfm_fit(fit, level=0.95, color='tab:green'):
    """
    Top panel: the shared latent state with its credible band.
    One panel per channel: z-scored observations with the implied c_j * x_k fit.
    """
    import matplotlib.pyplot as plt

    series = fit['series']
    channels = series['channels']
    k = series['session_idx']
    J = len(channels)

    fig, axes = plt.subplots(J + 1, 1, figsize=(9, 2.1 * (J + 1)), sharex=True)

    z = 1.959963984540054 if level == 0.95 else abs(_norm_ppf((1 - level) / 2))
    sd = np.sqrt(fit['P_smooth'])

    ax = axes[0]
    ax.plot(k, fit['x_smooth'], '-', lw=2, color=color)
    ax.fill_between(k, fit['x_smooth'] - z * sd, fit['x_smooth'] + z * sd,
                    color=color, alpha=0.2)
    ax.set_ylabel('shared state x')
    ax.set_title(f"{series['animal_id']} — {series['session_type']} — one-factor DFM\n"
                 f"sigma2_eps={fit['sigma2_eps']:.4f}, "
                 f"loadings=[{', '.join(f'{v:+.2f}' for v in fit['c'])}]  "
                 f"(higher x = better)", fontsize=9)

    for j, metric in enumerate(channels):
        ax = axes[j + 1]
        ax.plot(k, series['Y'][:, j], 'o', ms=4, color='0.6', alpha=0.8)
        ax.plot(k, fit['c'][j] * fit['x_smooth'], '-', lw=2, color=color)
        ax.fill_between(k,
                        fit['c'][j] * (fit['x_smooth'] - z * sd),
                        fit['c'][j] * (fit['x_smooth'] + z * sd),
                        color=color, alpha=0.2)
        ax.set_ylabel(f"z({metric})", fontsize=8)
        ax.text(0.99, 0.92, f"c={fit['c'][j]:+.2f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=8)

    axes[-1].set_xlabel('session_idx')
    for a in axes:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    return fig, axes

# ===========================================================================
# STEP 3 -- Plateau criteria
#
# Primary: SLOPE, terminal form.  Pr(x_K > x_k | all data) > 0.95.
#   The design doc specifies the one-step form Pr(x_{k+1} > x_k). Simulation
#   showed that version has essentially no power -- one session of improvement
#   is smaller than the posterior sd of a one-step difference, so it fires in
#   only ~38% of replicates and returns a plateau ~25 sessions too early when it
#   does. The terminal form is parameter-free, fires 100% of the time, recovers
#   the true plateau, and is the more faithful reading of Smith et al. (2004),
#   whose IO(0.95) is explicitly about performance "from that trial to the end
#   of the experiment". m-step forms are kept as a reported sensitivity.
#
# Sensitivity: ASYMPTOTE. First session from which the 95% band on x_k stays
#   within +/- delta of the terminal level for the balance of the segment.
#
# READ THIS BEFORE TRUSTING ANY PLATEAU INDEX
# -------------------------------------------
# On a finite segment, BOTH criteria fire even for an animal that never
# plateaus: near the end of any segment there is little remaining change left
# to detect, so a monotonically-improving animal still gets a "plateau" in its
# last fifth. Verified in simulation -- a pure linear ramp yields a terminal
# plateau at position 56/68 and an asymptote plateau at the same place.
#
# The statistic that separates them is `drift_ratio`: the state's mean drift
# per session AFTER the plateau, over its mean drift BEFORE.
#       true plateau : 0.16  (IQR 0.14-0.27)
#       linear ramp  : 0.93  (IQR 0.76-1.20)
# No overlap. Always check drift_ratio and frac_after before using a plateau.
# ===========================================================================

PLATEAU_PROB_THRESHOLD = 0.95
ASYMPTOTE_DELTAS = (0.50, 0.75, 1.00)     # z units of the shared state
DRIFT_RATIO_MAX = 0.50                    # above this, the "plateau" is a ramp tail
FRAC_AFTER_MIN = 0.25                     # plateau must leave a usable stretch

_Z_ONE_SIDED_95 = 1.6448536269514722
_Z_TWO_SIDED_95 = 1.959963984540054


def smoother_gains(fit):
    """A_k = P_filt[k] / P_pred[k+1], the backward-recursion gains."""
    P_filt, P_pred = fit['P_filt'], fit['P_pred']
    K = len(P_filt)
    return np.array([P_filt[k] / P_pred[k + 1] for k in range(K - 1)])


def _suffix_gain_products(A):
    """sp[k] = prod(A[k:]) -- the gain product from k out to the last session."""
    K = len(A) + 1
    sp = np.ones(K)
    for k in range(K - 2, -1, -1):
        sp[k] = A[k] * sp[k + 1]
    return sp


def improvement_prob(fit, horizon='terminal'):
    """
    Pr(x_target > x_k | all data) for every session k, from the smoother's
    cross-covariances:

        Cov(x_k, x_{k+m} | Y) = (prod_{i=k}^{k+m-1} A_i) * P_smooth[k+m]
        Var(x_{k+m} - x_k | Y) = P_smooth[k+m] + P_smooth[k] - 2*Cov

    Verified exact at every lag against brute-force inversion of the joint
    Gaussian precision matrix.

    horizon : 'terminal' -> compare against the last session of the segment
              int m      -> compare against session k + m
    Returns (prob, delta_mean, delta_sd), each length K, NaN where undefined.
    """
    x_s, P_s = fit['x_smooth'], fit['P_smooth']
    K = len(x_s)
    A = smoother_gains(fit)

    prob = np.full(K, np.nan)
    delta_mean = np.full(K, np.nan)
    delta_sd = np.full(K, np.nan)

    if horizon == 'terminal':
        sp = _suffix_gain_products(A)
        for k in range(K - 1):
            cov = sp[k] * P_s[K - 1]
            var = max(P_s[K - 1] + P_s[k] - 2.0 * cov, 1e-12)
            delta_mean[k] = x_s[K - 1] - x_s[k]
            delta_sd[k] = np.sqrt(var)
            prob[k] = _norm_cdf(delta_mean[k] / delta_sd[k])
    else:
        m = int(horizon)
        for k in range(K - m):
            cov = np.prod(A[k:k + m]) * P_s[k + m]
            var = max(P_s[k + m] + P_s[k] - 2.0 * cov, 1e-12)
            delta_mean[k] = x_s[k + m] - x_s[k]
            delta_sd[k] = np.sqrt(var)
            prob[k] = _norm_cdf(delta_mean[k] / delta_sd[k])

    return prob, delta_mean, delta_sd


def drift_ratio(fit, position):
    """
    |mean drift per session after `position`| / |mean drift per session before|.

    THE guard statistic. A real plateau gives ~0.15; a monotone ramp whose tail
    was mistaken for a plateau gives ~1.0.
    """
    x_s = fit['x_smooth']
    K = len(x_s)
    if position < 2 or position > K - 3:
        return np.nan
    pre = (x_s[position] - x_s[0]) / position
    post = (x_s[K - 1] - x_s[position]) / (K - 1 - position)
    return abs(post) / abs(pre) if abs(pre) > 1e-9 else np.nan


def plateau_slope(fit, horizon='terminal', threshold=PLATEAU_PROB_THRESHOLD):
    """
    Plateau = the first session after which the animal is no longer credibly
    still improving, i.e. one past the LAST session whose improvement
    probability exceeds `threshold`. Mirrors Smith's "and remains ... for the
    balance of the experiment" clause, which is load-bearing here: several
    animals have an intermediate flat stretch that a first-crossing rule would
    fire on.
    """
    prob, delta_mean, delta_sd = improvement_prob(fit, horizon)
    K = len(fit['x_smooth'])

    defined = np.where(np.isfinite(prob))[0]
    above = np.where(np.isfinite(prob) & (prob > threshold))[0]

    if len(above) == 0:
        position, status = 0, 'immediate'      # never credibly improving
    else:
        position = int(above.max()) + 1
        last_defined = int(defined.max())
        # still improving at the last comparison we can make -> no plateau
        status = 'not_reached' if (int(above.max()) >= last_defined or position >= K - 1) \
                 else 'reached'

    det = _Z_ONE_SIDED_95 * delta_sd[position] if position < K and np.isfinite(delta_sd[position]) else np.nan

    return dict(position=position, status=status, prob=prob,
                n_after=K - position,
                frac_after=(K - position) / K,
                drift_ratio=drift_ratio(fit, position),
                detectable_total_z=det)


def plateau_asymptote(fit, delta=0.75, n_terminal=5, level=0.95):
    """
    Plateau = the first session from which the `level` credible band on x_k
    stays entirely within +/- delta of the terminal level x_bar (posterior mean
    over the last `n_terminal` sessions) for the balance of the segment.

    delta MUST exceed the band half-width or the criterion can never fire --
    `min_usable_delta` reports that floor.
    """
    x_s, P_s = fit['x_smooth'], fit['P_smooth']
    K = len(x_s)
    z = _Z_TWO_SIDED_95 if level == 0.95 else abs(_norm_ppf((1 - level) / 2))
    sd = np.sqrt(P_s)

    x_bar = float(np.mean(x_s[-n_terminal:]))
    inside = ((x_s - z * sd) >= x_bar - delta) & ((x_s + z * sd) <= x_bar + delta)
    min_usable_delta = float(z * sd.max())

    if not inside[-1]:
        return dict(position=K, status='not_reached', x_bar=x_bar, n_after=0,
                    frac_after=0.0, drift_ratio=np.nan,
                    min_usable_delta=min_usable_delta)

    position = K - 1
    while position > 0 and inside[position - 1]:
        position -= 1

    return dict(position=position,
                status='immediate' if position == 0 else 'reached',
                x_bar=x_bar, n_after=K - position, frac_after=(K - position) / K,
                drift_ratio=drift_ratio(fit, position),
                min_usable_delta=min_usable_delta)


def plateau_table(fit, threshold=PLATEAU_PROB_THRESHOLD,
                  m_horizons=(5, 10), deltas=ASYMPTOTE_DELTAS):
    """
    One row per criterion for a single fitted segment.

        animal_id | segment | plateau_session_idx | criterion | threshold |
        n_sessions | status | n_after | frac_after | drift_ratio |
        detectable_total_z | credible

    `credible` is the guard: False means the index is probably the tail of a
    still-rising curve rather than a real plateau. Never use a plateau index
    with credible=False as a train/test boundary without looking at the figure.
    """
    series = fit['series']
    session_idx = series['session_idx']
    K = len(session_idx)
    rows = []

    def _row(res, criterion, thr):
        pos = res['position']
        reached = res['status'] == 'reached'
        return {
            'animal_id': series['animal_id'],
            'segment': series['session_type'],
            'plateau_session_idx': int(session_idx[pos]) if (reached and pos < K) else pd.NA,
            'criterion': criterion,
            'threshold': thr,
            'n_sessions': K,
            'status': res['status'],
            'n_after': res.get('n_after', np.nan),
            'frac_after': res.get('frac_after', np.nan),
            'drift_ratio': res.get('drift_ratio', np.nan),
            'detectable_total_z': res.get('detectable_total_z', np.nan),
            'credible': bool(reached
                             and np.isfinite(res.get('drift_ratio', np.nan))
                             and res['drift_ratio'] < DRIFT_RATIO_MAX
                             and res.get('frac_after', 0) >= FRAC_AFTER_MIN),
        }

    rows.append(_row(plateau_slope(fit, 'terminal', threshold),
                     'slope_terminal', threshold))
    for m in m_horizons:
        rows.append(_row(plateau_slope(fit, m, threshold), f'slope_m{m}', threshold))
    for d in deltas:
        rows.append(_row(plateau_asymptote(fit, delta=d), f'asymptote_d{d:.2f}', d))

    return pd.DataFrame(rows)


def plot_plateau_diagnosis(fit, threshold=PLATEAU_PROB_THRESHOLD, delta=0.75):
    """Top: state, band, both plateau markers. Bottom: the improvement probability."""
    import matplotlib.pyplot as plt

    series = fit['series']
    k = series['session_idx']
    slope = plateau_slope(fit, 'terminal', threshold)
    asym = plateau_asymptote(fit, delta=delta)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})

    sd = np.sqrt(fit['P_smooth'])
    ax = axes[0]
    ax.plot(k, fit['x_smooth'], '-', lw=2, color='tab:green')
    ax.fill_between(k, fit['x_smooth'] - _Z_TWO_SIDED_95 * sd,
                    fit['x_smooth'] + _Z_TWO_SIDED_95 * sd, color='tab:green', alpha=0.2)
    ax.axhline(asym['x_bar'], color='0.5', ls=':', lw=1)
    ax.axhspan(asym['x_bar'] - delta, asym['x_bar'] + delta, color='0.85', alpha=0.4, zorder=0)

    if slope['status'] == 'reached':
        ax.axvline(k[slope['position']], color='crimson', lw=2,
                   label=f"slope_terminal (idx {k[slope['position']]})")
    if asym['status'] == 'reached':
        ax.axvline(k[asym['position']], color='tab:blue', lw=2, ls='--',
                   label=f"asymptote d={delta} (idx {k[asym['position']]})")
    ax.set_ylabel('shared state x')
    ax.legend(fontsize='x-small', loc='lower right')
    ax.set_title(f"{series['animal_id']} — {series['session_type']} — "
                 f"drift_ratio={slope['drift_ratio']:.2f}, "
                 f"frac_after={slope['frac_after']:.2f}, "
                 f"credible={slope['drift_ratio'] < DRIFT_RATIO_MAX if np.isfinite(slope['drift_ratio']) else False}",
                 fontsize=9)

    ax = axes[1]
    ax.plot(k, slope['prob'], '-', color='crimson')
    ax.axhline(threshold, color='0.4', ls='--', lw=1)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(r'Pr($x_K > x_k$)')
    ax.set_xlabel('session_idx')

    for a in axes:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    return fig, axes

# --- Reporting figure: plateau overlaid on the familiar four metrics ---------

REPORT_PANELS = [
    ('reentry_index', 'Re-entry index',      'in model'),
    ('consumption',   'Consumption (s)',     'in model'),
    ('pct_engaged',   '% engaged',           'held out'),
    ('realized_rr',   'Reward rate (rew/s)', 'held out'),
]
C_LOW, C_HIGH, C_STATE, C_MARK = '#56B4E9', '#E69F00', '#0072B2', '#111111'


def plot_animal_summary(metrics_df, animal_id,
                        channels=('reentry_index', 'consumption'),
                        threshold=PLATEAU_PROB_THRESHOLD):
    """
    One reporting figure per animal:
      top    -- shared latent state + 95% band for each segment, plateau marked
      below  -- the four familiar per-session metrics by block, same x-axis,
                with the plateau line carried down so the model's answer can be
                read directly against the curves you already know.

    Channels used to fit are labelled IN MODEL; the other two are labelled
    held out, which is the point: they were never allowed to influence the
    plateau, so their agreement is independent evidence.
    """
    import matplotlib.pyplot as plt

    animal_df = metrics_df[metrics_df['animal_id'] == animal_id]
    fits, plateaus = {}, {}

    for segment in ('pre-surgery', 'post-surgery'):
        if not (animal_df['session_type'] == segment).any():
            continue
        mv = prepare_multivariate_series(metrics_df, animal_id,
                                         channels=list(channels),
                                         session_type=segment,
                                         zscore_scope='pre-surgery')
        fit = fit_factor_model(mv)
        res = plateau_slope(fit, 'terminal', threshold)
        fits[segment] = fit
        plateaus[segment] = res

    post = animal_df[animal_df['session_type'] == 'post-surgery']
    boundary = int(post['session_idx'].min()) if not post.empty else None

    fig, axes = plt.subplots(5, 1, figsize=(9.5, 10.2), sharex=True,
                             gridspec_kw={'height_ratios': [1.5, 1, 1, 1, 1],
                                          'hspace': 0.18})

    # --- latent state ---
    ax = axes[0]
    for segment, fit in fits.items():
        k = fit['series']['session_idx']
        sd = np.sqrt(fit['P_smooth'])
        ax.plot(k, fit['x_smooth'], '-', lw=2.4, color=C_STATE)
        ax.fill_between(k, fit['x_smooth'] - 1.96 * sd, fit['x_smooth'] + 1.96 * sd,
                        color=C_STATE, alpha=0.22)
    ax.set_ylabel('shared latent state\n(higher = better)')
    ax.set_title(f'{animal_id} — state-space plateau vs. the four session metrics',
                 fontweight='bold', fontsize=11, loc='left')

    # --- the four familiar metrics ---
    for ax_i, (metric, label, tag) in zip(axes[1:], REPORT_PANELS):
        for phase, color, name in (('0.4', C_LOW, 'Low'), ('0.8', C_HIGH, 'High')):
            phase_df = animal_df[animal_df['phase'] == phase].sort_values('session_idx')
            for segment in ('pre-surgery', 'post-surgery'):
                seg_df = phase_df[phase_df['session_type'] == segment]
                if seg_df.empty:
                    continue
                ax_i.plot(seg_df['session_idx'], seg_df[metric], 'o-', ms=2.6, lw=0.9,
                          color=color, alpha=0.75,
                          label=name if segment == 'pre-surgery' else None)
        ax_i.set_ylabel(label, fontsize=8.5)
        ax_i.text(0.995, 0.95,
                  'IN MODEL' if tag == 'in model' else 'held out — validation',
                  transform=ax_i.transAxes, ha='right', va='top', fontsize=7.5,
                  color='#111111' if tag == 'in model' else '#8C8C8C',
                  fontweight='bold' if tag == 'in model' else 'normal',
                  style='normal' if tag == 'in model' else 'italic')
    axes[1].legend(fontsize=8, frameon=False, ncol=2, loc='upper center',
                   bbox_to_anchor=(0.52, 1.04), title='block', title_fontsize=8)

    # --- surgery line and plateau lines on every panel ---
    for ax_i in axes:
        if boundary is not None:
            ax_i.axvline(boundary - 0.5, color='#999999', ls=':', lw=1.5)
        for segment, res in plateaus.items():
            if res['status'] == 'reached':
                idx = fits[segment]['series']['session_idx'][res['position']]
                ax_i.axvline(idx, color=C_MARK, ls='--', lw=2)

    lo, hi = axes[0].get_ylim()
    if boundary is not None:
        axes[0].text(boundary - 0.5, hi, ' surgery ', fontsize=8, color='#666666',
                     va='top', ha='center', bbox=dict(fc='white', ec='none', pad=1))

    for segment, res in plateaus.items():
        k = fits[segment]['series']['session_idx']
        if res['status'] == 'reached':
            idx = k[res['position']]
            axes[0].annotate(f'plateau\nsession {idx}',
                             xy=(idx, lo + 0.12 * (hi - lo)),
                             xytext=(idx - 0.28 * (k[-1] - k[0]), lo + 0.12 * (hi - lo)),
                             fontsize=9, fontweight='bold', color=C_MARK,
                             va='center', ha='center',
                             arrowprops=dict(arrowstyle='->', color=C_MARK, lw=1.3))
        else:
            note = ('flat from session 1 →\nthe whole block is plateau'
                    if res['status'] == 'immediate'
                    else 'still improving at the end →\nno plateau in this block')
            axes[0].annotate(f'{segment}: {res["status"]}\n{note}',
                             xy=(float(np.mean(k)), lo + 0.62 * (hi - lo)), xycoords='data',
                             xytext=(0.97, 0.06), textcoords='axes fraction',
                             fontsize=8.5, color=C_MARK, ha='right', va='bottom',
                             arrowprops=dict(arrowstyle='->', color='#888888', lw=1.1,
                                             connectionstyle='arc3,rad=-0.25'),
                             bbox=dict(fc='#F2F2F2', ec='#CCCCCC', boxstyle='round,pad=0.35'))

    axes[-1].set_xlabel('session_idx  (chronological, matches 00_pool_animal_transitions.py)')
    return fig, axes


def main_step3(animal_id='SZ036', session_type='pre-surgery',
               channels=('reentry_index', 'consumption')):
    import matplotlib.pyplot as plt

    metrics_df = load_metrics_table()
    mv = prepare_multivariate_series(metrics_df, animal_id, channels=list(channels),
                                     session_type=session_type)
    fit = fit_factor_model(mv, verbose=True)
    table = plateau_table(fit)
    print()
    print(table.to_string(index=False))
    plot_plateau_diagnosis(fit)
    plt.tight_layout()
    plt.show()
    return fit, table


def main_step2(animal_id='SZ036', session_type='pre-surgery', channels=None):
    import matplotlib.pyplot as plt

    metrics_df = load_metrics_table()
    mv = prepare_multivariate_series(metrics_df, animal_id, channels=channels,
                                     session_type=session_type)
    fit = fit_factor_model(mv, verbose=True)

    print(f"\n{animal_id} / {session_type}: {mv['Y'].shape[0]} sessions, "
          f"{mv['Y'].shape[1]} channels")
    print(f"  sigma2_eps      = {fit['sigma2_eps']:.5f}")
    print(f"  total info      = {fit['total_information']:.3f}  "
          f"(sum_j c_j^2 / sigma2_j)")
    print(f"  median band sd  = {np.median(np.sqrt(fit['P_smooth'])):.4f}")
    print()
    print(factor_diagnostics(fit).to_string(index=False))
    print()
    print(compare_to_univariate(fit, metrics_df).to_string(index=False))
    print()
    print(validate_against_heldout(fit, metrics_df))

    plot_dfm_fit(fit)
    plt.tight_layout()
    plt.show()

    return fit


def main(animal_id='RK008', metric='reentry_index', session_type='pre-surgery'):
    import matplotlib.pyplot as plt

    metrics_df = load_metrics_table()
    series = prepare_series(metrics_df, animal_id, metric=metric, session_type=session_type)

    print(f"{animal_id} / {metric} / {session_type}: "
          f"{len(series['y'])} sessions, {int(np.isnan(series['y']).sum())} missing")

    fit = fit_metric_series(series, verbose=True)

    print(f"  sigma2_eps (state)       = {fit['sigma2_eps']:.5f}")
    print(f"  sigma2_v   (observation) = {fit['sigma2_v']:.5f}")
    print(f"  signal-to-noise          = {fit['signal_to_noise']:.4f}")
    print(f"  log-likelihood           = {fit['loglik']:.2f}  ({fit['n_iter']} EM iters)")

    plot_local_level_fit(fit)
    plt.tight_layout()
    plt.show()

    return fit


if __name__ == "__main__":
    # main('SZ036', 'reentry_index', session_type='post-surgery')
    # main_step2('RK008', 'pre-surgery', channels={'reentry_index', 'consumption'})
    # fit, table = main_step3('RK008', 'post-surgery')
    metrics_df = load_metrics_table()
    for animal_id in ['SZ036', 'SZ037', 'SZ038', 'SZ039', 'SZ042', 'SZ043', 'RK008']:
        fig, _ = plot_animal_summary(metrics_df, animal_id)
        fig.savefig(f'plateau_summary_{animal_id}.pdf', bbox_inches='tight')
        plt.show()