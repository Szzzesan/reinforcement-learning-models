"""
check_port_swap.py -- standalone QC for the swapped-port animal (RK007).

Question: after processing, does every session follow the convention the RL pipeline assumes
(data_loader.convert_behavior_data_to_state_transitions uses port_map = {2: context, 1: gambling})?

RK007 was trained with the ports physically mirrored, so its raw port numbers are the opposite
of every other animal and the processing step has to swap them. This script checks that WITHOUT
trusting any of our own labels, using three independent sources of truth:

  1. TASK-STRUCTURE FINGERPRINT (processed data only).
     In every trial the context port pays out exactly 4 rewards, and all of them arrive before
     the first gambling-port reward. The gambling port never shows that pattern. So for each
     port we compute
         ctx_score = fraction of trials (with rewards at both ports) in which THIS port delivered
                     exactly 4 rewards, all before any reward at the other port.
     The real context port scores ~1.0, the real gambling port ~0.0, whatever the labels say.

  2. RAW FILE HEADER (ground truth written by the rig).
     Line 2 of every raw data_*.txt lists which port_num carried the 'exp_decreasing' distribution
     and which carried 'background'. That tells us whether the raw file needed a swap.

  3. PIPELINE OUTPUT (pooled transitions from 00_pool_animal_transitions.py).
     Every investment-port reward in the state-transition list should happen while
     rewards_in_context == 4 and gambling_disabled == 0. A port mix-up breaks this immediately.

It also flags one known hazard: the rig writes 'trial' events with a FIXED port label
(start = 2, end = 1) that does not follow the physical port. For normal animals that happens
to agree with the head-entry port; for RK007 the processing swap flips it, so every trial-start
row ends up labelled as the gambling port. data_loader treats key=='trial' & value==1 as an
entry event, so the check below confirms the head-entry row still wins in each 0.1 s bin.

Run from the repo root (while or after 00 runs):
    python -m src.check_port_swap
Nothing is modified; one CSV summary is written to outputs/qc_port_swap_check.csv.
"""
import os
import re
import glob
import pickle

import numpy as np
import pandas as pd

import src.config as config
from src.quality_control import port_swapped

CONTEXT_PORT_EXPECTED = 2   # data_loader port_map: Data Context=2 -> Env 0
GAMBLING_PORT_EXPECTED = 1  # data_loader port_map: Data Gambling=1 -> Env 1
CONTEXT_REWARDS_MAX = 4
DT = 0.1

PASS_HIGH = 0.8   # context port should score above this
PASS_LOW = 0.2    # gambling port should score below this


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_events(path):
    """Loads a processed .parquet or a raw data_*.txt into the same event-table shape."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, na_values=['None'], skiprows=3)


def read_raw_header_ports(raw_path):
    """
    Parses line 2 of a raw data_*.txt, e.g.
      RK007,...,{'distribution': <function exp_decreasing ...>, ..., 'port_num': 2},{'distribution': 'background', ..., 'port_num': 1},...
    Returns {'gambling': port_num, 'context': port_num} as written by the rig (i.e. RAW numbering).
    """
    with open(raw_path, 'r') as f:
        f.readline()
        header = f.readline()
    out = {}
    for block in re.findall(r"\{[^{}]*\}", header):
        m = re.search(r"'port_num':\s*(\d+)", block)
        if m is None:
            continue
        if 'exp_decreasing' in block:
            out['gambling'] = int(m.group(1))
        elif "'background'" in block:
            out['context'] = int(m.group(1))
    return out


def _session_datetime(path):
    """
    Extracts a comparable datetime from any of the naming schemes in use:
      RK007_2025-05-01_14-47-45_pi_events_processed.parquet   (pre-surgery processed)
      RK007_2025-06-17T15_02_pi_events_processed.parquet      (post-surgery processed, minute resolution)
      data_2025-06-17_15-03-44.txt                            (raw)
    """
    base = os.path.basename(path)
    m = re.search(r"(\d{4}-\d{2}-\d{2})[T_](\d{2})[-_](\d{2})(?:[-_](\d{2}))?", base)
    if m is None:
        return None
    date, hh, mm, ss = m.groups()
    return pd.Timestamp(f"{date} {hh}:{mm}:{ss or '00'}")


def find_raw_file(processed_path, raw_dir, max_gap_min=10):
    """Pre-surgery names match exactly; post-surgery raw files start ~1-3 min after the FP/processed stamp."""
    t_proc = _session_datetime(processed_path)
    candidates = glob.glob(os.path.join(raw_dir, "data_*.txt"))
    if t_proc is None or not candidates:
        return None
    gaps = [(abs((_session_datetime(c) - t_proc).total_seconds()), c) for c in candidates if _session_datetime(c)]
    if not gaps:
        return None
    gap, best = min(gaps)
    return best if gap <= max_gap_min * 60 else None


# ---------------------------------------------------------------------------
# Check 1: task-structure fingerprint
# ---------------------------------------------------------------------------
def fingerprint_ports(df):
    """
    Returns {port: ctx_score} plus the number of trials scored.
    ctx_score(p) = fraction of trials with rewards at both ports where port p paid exactly 4
    rewards and all of them came before the first reward at the other port.
    """
    r = df[(df['key'] == 'reward') & (df['value'] == 1)].dropna(subset=['port', 'trial'])
    r = r.sort_values('task_time')
    ports = sorted(r['port'].unique())
    if len(ports) != 2:
        return {}, 0
    p_a, p_b = ports
    hits = {p_a: 0, p_b: 0}
    n = 0
    for _, s in r.groupby('trial'):
        seq = s['port'].values
        if not ((seq == p_a).any() and (seq == p_b).any()):
            continue
        n += 1
        for p, other in [(p_a, p_b), (p_b, p_a)]:
            first_other = np.argmax(seq == other)
            if (seq == p).sum() == CONTEXT_REWARDS_MAX and (seq[:first_other] == p).sum() == CONTEXT_REWARDS_MAX:
                hits[p] += 1
    scores = {int(p): (hits[p] / n if n else np.nan) for p in hits}
    return scores, n


def classify(scores):
    """Returns the port number that behaves like the context port, or None if ambiguous."""
    if len(scores) != 2:
        return None
    ctx = [p for p, s in scores.items() if s >= PASS_HIGH]
    gam = [p for p, s in scores.items() if s <= PASS_LOW]
    if len(ctx) == 1 and len(gam) == 1:
        return ctx[0]
    return None


# ---------------------------------------------------------------------------
# Check 2: 'trial' event port labels vs the head entry they coincide with
# ---------------------------------------------------------------------------
def check_trial_event_ports(df):
    """
    For each trial-start row (key=='trial', value==1): which port did the mouse actually enter at
    that moment (nearest head entry), does the trial row's own 'port' agree, and would the trial
    row be the FIRST entry-type row in its 0.1 s bin (which is what data_loader reads)?
    """
    d = df[df['key'].isin(['trial', 'head'])].sort_values('task_time')
    starts = d[(d['key'] == 'trial') & (d['value'] == 1)]
    entries = d[(d['key'] == 'head') & (d['value'] == 1)]
    if starts.empty or entries.empty:
        return dict(n_trial_starts=0, trial_label_agrees=np.nan, trial_row_first_and_wrong=0)

    agree, first_and_wrong = 0, 0
    t0 = df['task_time'].min()  # data_loader bins from the first task_time
    ent_t = entries['task_time'].values
    for _, row in starts.iterrows():
        k = np.argmin(np.abs(ent_t - row['task_time']))
        entry_port = entries['port'].iloc[k]
        same = (row['port'] == entry_port)
        agree += same
        # Same 0.1 s bin and the trial row sorts first -> data_loader would take the trial row's port
        b_trial = np.floor((row['task_time'] - t0) / DT)
        b_entry = np.floor((ent_t[k] - t0) / DT)
        if (not same) and b_trial == b_entry and row['task_time'] < ent_t[k]:
            first_and_wrong += 1
    return dict(n_trial_starts=len(starts),
                trial_label_agrees=agree / len(starts),
                trial_row_first_and_wrong=first_and_wrong)


# ---------------------------------------------------------------------------
# Check 3: pooled transitions produced by 00
# ---------------------------------------------------------------------------
def check_pooled_transitions(pkl_path):
    """
    In the state-transition list (env ports: 0=context, 1=gambling, 2=travel):
      - every reward received at port 1 should occur with rewards_in_context == 4 and gambling_disabled == 0
      - rewards_in_context should never exceed 4
    Returns per-session fractions.
    """
    with open(pkl_path, 'rb') as f:
        transitions = pickle.load(f)
    rows = []
    for t in transitions:
        obs, _, reward, _, _ = t[:5]
        info = t[5] if len(t) > 5 else {}
        rows.append((info.get('session_id', -1), info.get('session_type', ''), obs[0], obs[4], obs[5], reward))
    tr = pd.DataFrame(rows, columns=['session_id', 'session_type', 'port', 'rewards_in_context',
                                     'gambling_disabled', 'reward'])
    out = []
    for (sid, stype), g in tr.groupby(['session_id', 'session_type']):
        gam_r = g[(g['port'] == 1) & (g['reward'] > 0)]
        ctx_r = g[(g['port'] == 0) & (g['reward'] > 0)]
        ok = ((gam_r['rewards_in_context'] == CONTEXT_REWARDS_MAX) & (gam_r['gambling_disabled'] == 0)).mean()
        out.append(dict(session_id=sid, session_type=stype,
                        n_ctx_rewards=len(ctx_r), n_gamble_rewards=len(gam_r),
                        gamble_rewards_when_enabled=ok if len(gam_r) else np.nan,
                        max_rewards_in_context=g['rewards_in_context'].max()))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def check_animal_processed(animal_id):
    """Runs checks 1 + 2 (+ header) on every processed session of one animal."""
    animal_dir = os.path.join(config.MAIN_DATA_ROOT, animal_id)
    folders = [
        ('pre-surgery', config.PRETRAINING_PROCESSED_DATA_SUBDIR, config.PRETRAINING_RAW_DATA_SUBDIR),
        ('post-surgery', config.PROCESSED_DATA_SUBDIR, config.RAW_DATA_SUBDIR),
    ]
    results = []
    for session_type, proc_sub, raw_sub in folders:
        proc_files = sorted(glob.glob(os.path.join(animal_dir, proc_sub, "*_pi_events_processed.parquet")))
        raw_dir = os.path.join(animal_dir, raw_sub)
        for path in proc_files:
            df = load_events(path)
            scores, n_trials = fingerprint_ports(df)
            ctx_port = classify(scores)

            if ctx_port is None:
                verdict = 'AMBIGUOUS'
            elif ctx_port == CONTEXT_PORT_EXPECTED:
                verdict = 'OK'
            else:
                verdict = 'SWAPPED'

            raw_path = find_raw_file(path, raw_dir)
            header = read_raw_header_ports(raw_path) if raw_path else {}
            raw_needed_swap = (header.get('context') != CONTEXT_PORT_EXPECTED) if header else np.nan

            results.append(dict(
                animal=animal_id,
                session_type=session_type,
                session=os.path.basename(path).replace('_pi_events_processed.parquet', ''),
                n_trials_scored=n_trials,
                score_port1=scores.get(1, np.nan),
                score_port2=scores.get(2, np.nan),
                context_port_by_behavior=ctx_port,
                verdict=verdict,
                raw_file=os.path.basename(raw_path) if raw_path else None,
                raw_context_port=header.get('context'),
                raw_gambling_port=header.get('gambling'),
                raw_needed_swap=raw_needed_swap,
                **check_trial_event_ports(df),
            ))
    return pd.DataFrame(results)


def report_unprocessed_raw(animal_id):
    """Lists pre-surgery raw sessions with no processed file (they silently drop out of the pipeline)."""
    animal_dir = os.path.join(config.MAIN_DATA_ROOT, animal_id)
    raw = glob.glob(os.path.join(animal_dir, config.PRETRAINING_RAW_DATA_SUBDIR, "data_*.txt"))
    proc = glob.glob(os.path.join(animal_dir, config.PRETRAINING_PROCESSED_DATA_SUBDIR, "*_pi_events_processed.parquet"))
    proc_times = {_session_datetime(p) for p in proc}
    return sorted(os.path.basename(r) for r in raw if _session_datetime(r) not in proc_times)


def main(animals=("RK007", "RK008")):
    """RK008 is the un-swapped control: it should come out identical to RK007 if processing is right."""
    all_results = []
    for animal in animals:
        print(f"\n{'=' * 70}\n{animal}  (quality_control.port_swapped = {port_swapped.get(animal)})\n{'=' * 70}")
        res = check_animal_processed(animal)
        all_results.append(res)
        if res.empty:
            print("  No processed sessions found.")
            continue

        cols = ['session_type', 'session', 'n_trials_scored', 'score_port1', 'score_port2', 'verdict',
                'raw_context_port', 'raw_needed_swap', 'trial_label_agrees', 'trial_row_first_and_wrong']
        with pd.option_context('display.width', 200, 'display.max_rows', 500):
            print(res[cols].round(2).to_string(index=False))

        print("\n  Summary:")
        print("   verdicts:", res['verdict'].value_counts().to_dict())
        need = res['raw_needed_swap'].dropna()
        if len(need):
            print(f"   raw header says swap needed in {int(need.sum())}/{len(need)} sessions "
                  f"(quality_control says {port_swapped.get(animal)})")
            if need.astype(bool).any() != bool(port_swapped.get(animal)):
                print("   ⚠️ raw headers DISAGREE with quality_control.port_swapped")
        n_wrong = res['trial_row_first_and_wrong'].sum()
        print(f"   trial-start rows whose port label disagrees with the head entry: "
              f"{(1 - res['trial_label_agrees']).mul(res['n_trial_starts']).sum():.0f} "
              f"(of which would reach data_loader first in their bin: {n_wrong})")
        missing = report_unprocessed_raw(animal)
        if missing:
            print(f"   pre-surgery raw sessions with no processed file: {missing}")

        # Check 3: whatever pooled transitions exist for this animal in the active experiment
        data_dir = os.path.join(config.MODELING_PROJECT_ROOT, config.MODELING_DATA_SUBDIR)
        for kind, fname in [('train', f"pooled_transitions_{animal}.pkl"),
                            ('target', f"target_sessions_pooled_transitions_{animal}.pkl")]:
            pkl = os.path.join(data_dir, fname)
            if not os.path.exists(pkl):
                continue
            tr = check_pooled_transitions(pkl)
            # NOTE: this is NOT ~1.0 even for un-swapped animals (~0.93-0.96 median in exp_01; low in the
            # first few naive sessions). Judge RK007 against the RK008 control, not against 1.0.
            # A port swap would push it far lower (gambling rewards landing in "context" states).
            late = tr[tr['session_id'] > tr['session_id'].quantile(0.5)]
            print(f"   pooled {kind}: {len(tr)} sessions | gambling rewards while enabled: "
                  f"median {tr['gamble_rewards_when_enabled'].median():.3f} "
                  f"(later half {late['gamble_rewards_when_enabled'].median():.3f}) | "
                  f"max rewards_in_context {tr['max_rewards_in_context'].max():.0f}")

    out = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    if not out.empty:
        save_path = os.path.join(config.MODELING_PROJECT_ROOT, "outputs", "qc_port_swap_check.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out.to_csv(save_path, index=False)
        print(f"\n💾 Saved {save_path}")
    return out


if __name__ == "__main__":
    main()
