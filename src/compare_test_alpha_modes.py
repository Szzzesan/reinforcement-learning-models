"""
compare_test_alpha_modes.py -- side-by-side comparison of the target-session results with the agent
FROZEN (alpha = 0) vs. LEARNING (the animal's fitted alpha kept on the target sessions).

Needs both versions of the evaluation outputs for the active experiment:
    frozen   -> outputs/<exp>/3_evaluation_metrics/                  (run_pipeline.py --from 03)
    learning -> outputs/<exp>/3_evaluation_metrics/learning_alpha/   (run_pipeline.py --test-alpha learning --from 03)

Writes to outputs/<exp>/4_outputs/<YYYY_MM_DD>/frozen_vs_learning/:
    leave_time_scatter_frozen_vs_learning.png   observed vs predicted leave time, one panel per mode
    leave_time_mae_by_session.png               MAE per target session and animal, both modes overlaid
    td_error_vs_time_in_port.png                TD error at investment-port rewards, one panel per mode
    frozen_vs_learning_by_animal.csv            MAE / bias / r per animal and mode
    frozen_vs_learning_all_animals.csv          same pooled over animals
    frozen_vs_learning_by_session.csv           same per animal x target session
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.current_experiment_config import get_eval_metrics_dir, get_dated_output_dir, ACTIVE_EXP_ID

ANIMALS = ['SZ036', 'SZ037', 'SZ038', 'SZ039', 'SZ042', 'SZ043', 'RK007', 'RK008']
MODES = ['frozen', 'learning']
MODE_TITLES = {'frozen': 'Frozen (alpha = 0)', 'learning': 'Learning (fitted alpha)'}
MODE_COLORS = {'frozen': '#4C72B0', 'learning': '#D62728'}
CONTEXT_COLORS = {0: sns.color_palette('Set2')[0], 1: sns.color_palette('Set2')[1]}  # same as the scatter plots
CONTEXT_LABELS = {0: 'Low', 1: 'High'}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_master_predictions_for_mode(mode, filename="master_leave_time_predictions.json"):
    path = Path(get_eval_metrics_dir(mode)) / filename
    if not path.exists():
        print(f"❌ Missing {mode} predictions: {path}")
        return None
    with open(path, 'r') as f:
        df = pd.DataFrame(json.load(f))
    df['mode'] = mode
    # Trial order within an animal is identical in both modes (same target transitions)
    df['trial_index'] = df.groupby('animal_id').cumcount()
    return df


def load_td_rewards_for_mode(mode):
    frames = []
    for animal in ANIMALS:
        path = Path(get_eval_metrics_dir(mode)) / f"tde_reward_features_{animal}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df['animal'] = animal
            frames.append(df)
    if not frames:
        print(f"❌ No {mode} TD-error files in {get_eval_metrics_dir(mode)}")
        return None
    out = pd.concat(frames, ignore_index=True)
    out['mode'] = mode
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics(g):
    err = g['predicted'] - g['actual']
    r = np.corrcoef(g['actual'], g['predicted'])[0, 1] if len(g) > 1 else np.nan
    return pd.Series({'N_trials': len(g), 'MAE': err.abs().mean(), 'bias': err.mean(), 'Pearson_r': r})


def build_tables(pred):
    by_animal = (pred.groupby(['animal_id', 'mode']).apply(_metrics)
                 .unstack('mode').swaplevel(axis=1).sort_index(axis=1))
    overall = pred.groupby('mode').apply(_metrics)  # rows: frozen / learning
    by_session = pred.groupby(['animal_id', 'session_id', 'mode']).apply(_metrics).reset_index()
    return by_animal, overall, by_session


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_scatter_side_by_side(pred, save_dir):
    lim = np.percentile(np.concatenate([pred['actual'], pred['predicted']]), 99) + 1.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    for ax, mode in zip(axes, MODES):
        d = pred[pred['mode'] == mode]
        for ctx in [0, 1]:
            dc = d[d['context'] == ctx]
            ax.plot(dc['actual'], dc['predicted'], 'o', color=CONTEXT_COLORS[ctx], alpha=0.4,
                    markersize=4, markeredgewidth=0, label=CONTEXT_LABELS[ctx])
        ax.plot([0, lim], [0, lim], 'k:', alpha=0.3, label='y = x (Perfect)')
        m = _metrics(d)
        ax.set_title(f"{MODE_TITLES[mode]}\nMAE = {m.MAE:.2f} s | bias = {m.bias:+.2f} s | r = {m.Pearson_r:.2f}")
        ax.set_xlabel('Observed Leave Time (s)')
        ax.set_xlim(-0.2, lim)
        ax.set_ylim(-0.2, lim)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.2)
    axes[0].set_ylabel('Predicted Leave Time (s)')
    axes[0].legend(title='Context Block', loc='upper left')
    fig.suptitle(f"{ACTIVE_EXP_ID}: held-out sessions, all animals")
    plt.tight_layout()
    fig.savefig(save_dir / "leave_time_scatter_frozen_vs_learning.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_mae_by_session(by_session, save_dir):
    animals = [a for a in ANIMALS if a in set(by_session['animal_id'])]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    for ax, animal in zip(axes.flatten(), animals):
        d = by_session[by_session['animal_id'] == animal]
        for mode in MODES:
            dm = d[d['mode'] == mode].sort_values('session_id')
            ax.plot(np.arange(1, len(dm) + 1), dm['MAE'], 'o-', color=MODE_COLORS[mode],
                    label=MODE_TITLES[mode], markersize=4)
        ax.set_title(animal)
        ax.set_xlabel('Target session #')
        ax.grid(True, linestyle='--', alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel('Leave-time MAE (s)')
    for ax in axes.flatten()[len(animals):]:
        ax.set_visible(False)
    axes.flatten()[0].legend(loc='upper left', fontsize=9)
    fig.suptitle(f"{ACTIVE_EXP_ID}: leave-time error across held-out sessions")
    plt.tight_layout()
    fig.savefig(save_dir / "leave_time_mae_by_session.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_td_vs_time_in_port(td, save_dir, bin_edges=(0, 1, 2, 3, 4, 6, 8, 12, 30)):
    td = td.copy()
    td['tip_bin'] = pd.cut(td['time_in_port'], bins=list(bin_edges))
    # animal means first, then mean +/- SEM across animals (animals are the unit of replication)
    per_animal = (td.groupby(['mode', 'context', 'tip_bin', 'animal'], observed=True)['td_error']
                  .mean().reset_index())
    summary = (per_animal.groupby(['mode', 'context', 'tip_bin'], observed=True)['td_error']
               .agg(['mean', 'sem']).reset_index())
    centers = {b: b.mid for b in summary['tip_bin'].unique()}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, mode in zip(axes, MODES):
        for ctx in [0, 1]:
            d = summary[(summary['mode'] == mode) & (summary['context'] == ctx)]
            x = [centers[b] for b in d['tip_bin']]
            ax.errorbar(x, d['mean'], yerr=d['sem'], fmt='o-', color=CONTEXT_COLORS[ctx],
                        label=CONTEXT_LABELS[ctx], capsize=3)
        ax.axhline(0, color='grey', linewidth=0.8)
        ax.set_title(MODE_TITLES[mode])
        ax.set_xlabel('Time in port at reward (s)')
        ax.grid(True, linestyle='--', alpha=0.3)
    axes[0].set_ylabel('TD error at reward (mean ± SEM across animals)')
    axes[0].legend(title='Context Block')
    fig.suptitle(f"{ACTIVE_EXP_ID}: TD error at investment-port rewards, held-out sessions")
    plt.tight_layout()
    fig.savefig(save_dir / "td_error_vs_time_in_port.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    save_dir = Path(get_dated_output_dir("frozen_vs_learning", mode_subfolder=False))

    preds = [load_master_predictions_for_mode(m) for m in MODES]
    if any(p is None for p in preds):
        print("⚠️ Skipping the leave-time comparison: run the pipeline in both modes first "
              "(python run_pipeline.py --test-alpha learning --from 03).")
    else:
        pred = pd.concat(preds, ignore_index=True)
        # Sanity check: both modes must describe the same trials
        chk = pred.pivot_table(index=['animal_id', 'trial_index'], columns='mode', values='actual')
        if chk.isna().any().any() or not np.allclose(chk['frozen'], chk['learning']):
            raise ValueError("Frozen and learning predictions are not on the same trials; re-run 00-03 for both modes.")

        by_animal, overall, by_session = build_tables(pred)
        print("\nLeave-time fit on held-out sessions (frozen vs learning):")
        with pd.option_context('display.width', 200):
            print(by_animal.round(3).to_string())
            print("\nAll animals:")
            print(overall.round(3).to_string())
        by_animal.to_csv(save_dir / "frozen_vs_learning_by_animal.csv")
        overall.to_csv(save_dir / "frozen_vs_learning_all_animals.csv")
        by_session.to_csv(save_dir / "frozen_vs_learning_by_session.csv", index=False)

        plot_scatter_side_by_side(pred, save_dir)
        plot_mae_by_session(by_session, save_dir)

    tds = [load_td_rewards_for_mode(m) for m in MODES]
    if any(t is None for t in tds):
        print("⚠️ Skipping the TD-error comparison (missing 03 outputs for one of the modes).")
    else:
        plot_td_vs_time_in_port(pd.concat(tds, ignore_index=True), save_dir)

    print(f"\n💾 Comparison outputs saved to {save_dir}")


if __name__ == "__main__":
    main()
