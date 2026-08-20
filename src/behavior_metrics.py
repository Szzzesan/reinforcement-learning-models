"""
Per-session behavioral performance metrics, persisted as a tidy table.

Refactor of the old `calculate_session_metrics` / `plot_metrics_panel` pair.
The metric definitions are unchanged; what changes is that the metrics are now
emitted as a cached tidy DataFrame (one row per animal x session x phase) that
downstream analyses -- in particular the state-space learning-curve model that
defines the train/test split -- can consume without recomputing.

Schema of the cached table
--------------------------
    animal_id      str    e.g. 'SZ036'
    session_idx    int    1-based, chronological across pre- AND post-surgery.
                          Identical to `session['id']` in 00_pool_animal_transitions.py.
    session_date   datetime64[ns]  parsed from the filename
    session_type   str    'pre-surgery' | 'post-surgery'
    phase          str    '0.4' (Low block) | '0.8' (High block)
    consumption    float  s, mean over trials in this phase
    realized_rr    float  rewards / engaged time
    pct_engaged    float  engaged time / total trial time
    reentry_index  float  context-port exits / trials in phase
    n_trials       int    trials assigned to this phase in this session
    n_rewards      int    rewards counted in this phase (numerator of realized_rr)

No binning of any kind is applied here -- five-session binning is plot-time only.
"""

import os
import glob

import numpy as np
import pandas as pd

import src.config

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# `min_dif` and `get_entry_exit` are part of the fiber-photometry-analysis codebase,
# not of reinforcement-learning-models. Nothing else in this module depends on
# that codebase.
# ---------------------------------------------------------------------------
def min_dif(a, b, tolerance=0, return_index=False, rev=False):
    """
    Calculates the minimum difference between elements of two arrays (b - a).
    Finds the smallest positive difference (next event in b after event in a).
    """
    if isinstance(a, pd.Series):
        a = a.values
    if isinstance(b, pd.Series):
        b = b.values

    # Standardize shape
    a = np.array(a)
    b = np.array(b)

    if rev:
        outer = -1 * np.subtract.outer(a, b)
        outer[outer <= tolerance] = np.nan
    else:
        # outer[i, j] = b[j] - a[i]
        outer = np.subtract.outer(b, a)
        outer[outer <= tolerance] = np.nan

    # Suppress "All-NaN slice" warnings for empty columns
    with np.errstate(all='ignore'):
        mins = np.nanmin(outer, axis=0)

    if return_index:
        with np.errstate(all='ignore'):
            index = np.nanargmin(outer, axis=0)
        return index, mins
    return mins


def get_entry_exit(df, trial):
    """
    Extracts entry and exit times for Background (Context) and Exponential (Investment) ports
    for a specific trial, handling various edge cases (middle of trial, early entries, etc.).
    """
    is_trial = df.trial == trial
    start = df.value == 1
    end = df.value == 0
    port1 = df.port == 1  # Investment
    port2 = df.port == 2  # Context

    # Trial boundaries
    try:
        trial_start = df[is_trial & start & (df.key == 'trial')].session_time.values[0]
        trial_end = df[is_trial & end & (df.key == 'trial')].session_time.values[0]
        # Trial middle is defined by the LED turning off in Port 2 (Context exit cue)
        trial_middle = df[is_trial & end & (df.key == 'LED') & port2].session_time.values[0]
    except IndexError:
        # If trial structure is incomplete, return empty
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # --- Background (Context) Port 2 ---
    bg_entries = df[is_trial & port2 & start & (df.key == 'head')].session_time.to_numpy()
    bg_exits = df[is_trial & port2 & end & (df.key == 'head')].session_time.to_numpy()

    # Handle BG boundary conditions
    if len(bg_entries) == 0 or (len(bg_exits) > 0 and bg_entries[0] > bg_exits[0]):
        bg_entries = np.concatenate([[trial_start], bg_entries])

    # If last entry has no exit and is close to end, trim or cap?
    # Logic from snippet: if trial_end - bg_entries[-1] < .1, remove entry.
    if len(bg_entries) > 0 and (trial_end - bg_entries[-1] < .1):
        bg_entries = bg_entries[:-1]

    # If missing last exit, assume trial_middle (end of context phase)
    if len(bg_exits) == 0 or (len(bg_entries) > 0 and bg_entries[-1] > bg_exits[-1]):
        bg_exits = np.concatenate([bg_exits, [trial_middle]])

    # --- Exponential (Investment) Port 1 ---
    # Standard entries (after trial middle)
    exp_entries = df[
        is_trial & port1 & start & (df.key == 'head') & (df.session_time > trial_middle)].session_time.to_numpy()
    exp_exits = df[
        is_trial & port1 & end & (df.key == 'head') & (df.session_time > trial_middle)].session_time.to_numpy()

    if not (len(exp_entries) == 0 and len(exp_exits) == 0):
        if len(exp_entries) == 0:
            exp_entries = np.concatenate([[trial_middle], exp_entries])
        if len(exp_exits) == 0:
            exp_exits = np.concatenate([exp_exits, [trial_end]])

        if len(exp_entries) > 0 and len(exp_exits) > 0:
            if exp_entries[0] > exp_exits[0]:
                exp_entries = np.concatenate([[trial_middle], exp_entries])
            if exp_entries[-1] > exp_exits[-1]:
                exp_exits = np.concatenate([exp_exits, [trial_end]])

    # Early entries (before trial middle - rare/error)
    early_exp_entries = df[
        is_trial & port1 & start & (df.key == 'head') & (df.session_time < trial_middle)].session_time.to_numpy()
    early_exp_exits = df[
        is_trial & port1 & end & (df.key == 'head') & (df.session_time < trial_middle)].session_time.to_numpy()

    # (Skipping detailed correction for early entries for brevity, using main logic)

    return bg_entries, bg_exits, exp_entries, exp_exits, early_exp_entries, early_exp_exits


# --- constants -------------------------------------------------------------

PHASES = ('0.4', '0.8')                       # '0.4' = Low block, '0.8' = High block
PHASE_LABELS = {'0.4': 'Low', '0.8': 'High'}

# Real travel time between the two ports. (An earlier comment in this code said
# 0.5 s; that was a superseded estimate. 0.3 s is the value we use.)
TRAVEL_TIME = 0.3

# Consumption is the latency from the last context reward to the next valid
# context-port exit; anything longer than this is not a consumption event.
MAX_CONSUMPTION_S = 10.0

METRIC_COLUMNS = ['consumption', 'realized_rr', 'pct_engaged', 'reentry_index']

TIDY_COLUMNS = [
    'animal_id', 'session_idx', 'session_date', 'session_type', 'phase',
    'consumption', 'realized_rr', 'pct_engaged', 'reentry_index',
    'n_trials', 'n_rewards',
]

CACHE_FILENAME = 'behavior_metrics_by_session.parquet'

SZ_animals = ['SZ036', 'SZ037', 'SZ038', 'SZ039', 'SZ042', 'SZ043']
RK_animals = ['RK007', 'RK008']
ALL_ANIMALS = SZ_animals + RK_animals


def get_cache_path():
    """Path of the cached tidy metrics table: data/behavior_metrics_by_session.parquet."""
    return os.path.join(src.config.MODELING_PROJECT_ROOT, 'data', CACHE_FILENAME)


# --- session discovery / ordering -----------------------------------------
# These mirror `get_session_files` and `extract_datetime_from_filename` in
# 00_pool_animal_transitions.py. That module cannot be imported (its name
# starts with digits), so the two definitions are kept byte-identical here.
# Once this module exists, 00_pool_animal_transitions.py can import them from
# here instead, and the duplication goes away.

def get_session_files(animal_id, subdir):
    """Finds and returns a list of processed .parquet file paths."""
    animal_dir = os.path.normpath(os.path.join(src.config.MAIN_DATA_ROOT, animal_id))
    processed_dir = os.path.join(animal_dir, subdir)

    file_pattern = os.path.join(processed_dir, "*_pi_events_processed.parquet")
    return glob.glob(file_pattern)  # We will sort them globally later


def extract_datetime_from_filename(filepath):
    """
    Extracts the 'YYYY-MM-DD_HH-MM-SS' portion from filenames like:
    'SZ036_2023-10-15_19-18-52_pi_events_processed.parquet'
    This ensures flawless chronological sorting.
    """
    basename = os.path.basename(filepath)
    try:
        parts = basename.split('_')
        # parts[1] is '2023-10-15', parts[2] is '19-18-52'
        return f"{parts[1]}_{parts[2]}"
    except IndexError:
        return basename  # Fallback to standard alphabetical sort if format differs


def parse_session_datetime(filepath):
    """
    Turns the sort key from `extract_datetime_from_filename` into a Timestamp.
    Returns NaT if the filename does not follow the expected convention, so that
    a malformed name degrades the `session_date` column without breaking the run.
    """
    key = extract_datetime_from_filename(filepath)
    return pd.to_datetime(key, format='%Y-%m-%d_%H-%M-%S', errors='coerce')


def list_animal_sessions(animal_id):
    """
    Returns every session for one animal, pre- and post-surgery merged and sorted
    chronologically, with a 1-based `session_idx` assigned exactly the way
    00_pool_animal_transitions.py assigns `session['id']`.

    Returns a list of dicts: {'path', 'session_type', 'session_idx', 'session_date'}.
    """
    pre_surgery_paths = get_session_files(animal_id, src.config.PRETRAINING_PROCESSED_DATA_SUBDIR)
    post_surgery_paths = get_session_files(animal_id, src.config.PROCESSED_DATA_SUBDIR)

    all_sessions = []
    all_sessions.extend([{'path': p, 'session_type': 'pre-surgery'} for p in pre_surgery_paths])
    all_sessions.extend([{'path': p, 'session_type': 'post-surgery'} for p in post_surgery_paths])

    all_sessions.sort(key=lambda s: extract_datetime_from_filename(s['path']))

    for i, session in enumerate(all_sessions, start=1):
        session['session_idx'] = i
        session['session_date'] = parse_session_datetime(session['path'])

    return all_sessions


# --- phase matching --------------------------------------------------------

def get_trials_in_phase(trial_phase_map, target_phase):
    """
    Trial IDs belonging to `target_phase`.

    The `phase` column is inconsistently typed across sessions (sometimes the
    float 0.4, sometimes the string '0.4'), so match both ways. Only conversion
    failures are swallowed -- a genuine error still propagates.

    Parameters
    ----------
    trial_phase_map : pd.Series   index = trial id, values = phase
    target_phase    : str         '0.4' or '0.8'
    """
    trials_in_phase = []

    for t_id, p_val in trial_phase_map.items():
        if pd.isna(t_id) or pd.isna(p_val):
            continue

        is_match = False
        if str(p_val) == target_phase:
            is_match = True
        else:
            try:
                is_match = abs(float(p_val) - float(target_phase)) < 0.001
            except (ValueError, TypeError):
                is_match = False

        if is_match:
            trials_in_phase.append(t_id)

    return trials_in_phase


# --- the metrics themselves ------------------------------------------------

def _empty_phase_metrics():
    return {
        'consumption': np.nan,
        'realized_rr': np.nan,
        'pct_engaged': np.nan,
        'reentry_index': np.nan,
        'n_trials': 0,
        'n_rewards': 0,
    }


def calculate_session_metrics(session_df):
    """
    Calculates the 4 key metrics for a single session, separately per phase.

    Returns
    -------
    dict : {'0.4': {...}, '0.8': {...}}, each inner dict holding
           consumption, realized_rr, pct_engaged, reentry_index, n_trials, n_rewards.

    Notes
    -----
    - Consumption is now computed *per phase* (restricted to that phase's trials)
      rather than once per session, so that it fits one row per (session, phase).
      The old session-wide number is recoverable as the n_trials-weighted mean of
      the two phase values.
    - `n_rewards` is exactly the numerator of `realized_rr`, i.e. rewards in
      trials that had a complete trial-start/trial-end pair. `n_trials` counts
      every trial assigned to the phase, matching the denominator of
      `reentry_index`.
    """
    metrics_by_phase = {phase: _empty_phase_metrics() for phase in PHASES}

    if 'phase' not in session_df.columns:
        print("Warning: 'phase' column missing in session dataframe.")
        return metrics_by_phase

    # Map each trial to its phase; take the first non-null phase seen in the trial.
    trial_phase_map = session_df.groupby('trial')['phase'].first()

    # --- events used by more than one metric ---
    # Context-port (port 2) LED on == last context reward delivered.
    is_bg_end = (session_df.key == 'LED') & (session_df.port == 2) & (session_df.value == 1)
    # Context-port exits.
    is_bg_exit = (session_df.port == 2) & (session_df.key == 'head') & (session_df.value == 0)

    if 'is_valid' in session_df.columns:
        is_valid_bg_exit = is_bg_exit & (session_df['is_valid'] == 1)
    else:
        is_valid_bg_exit = is_bg_exit

    # Valid exits stay session-wide: the exit that terminates a trial's
    # consumption bout may be logged just outside that trial's own rows.
    valid_exits = session_df[is_valid_bg_exit].session_time

    for target_phase in PHASES:
        trials_in_phase = get_trials_in_phase(trial_phase_map, target_phase)
        phase_metrics = metrics_by_phase[target_phase]

        num_ideal = len(trials_in_phase)
        phase_metrics['n_trials'] = num_ideal

        if num_ideal == 0:
            continue

        in_phase = session_df.trial.isin(trials_in_phase)

        # --- 1. Consumption ---------------------------------------------
        # Time from context end (LED on, port 2) to nearest valid exit (port 2).
        bg_end_times = session_df[is_bg_end & in_phase].session_time

        if len(bg_end_times) > 0 and len(valid_exits) > 0:
            dif = min_dif(bg_end_times, valid_exits)
            valid_dif = dif[(~np.isnan(dif)) & (dif > 0) & (dif < MAX_CONSUMPTION_S)]
            if len(valid_dif) > 0:
                phase_metrics['consumption'] = np.nanmean(valid_dif)

        # --- 2. Percent engaged & 3. Realized reward rate ----------------
        engaged_durations = []
        total_durations = []
        rewards_count = 0

        for trial in trials_in_phase:
            is_this_trial = session_df.trial == trial
            try:
                # Entry/exit times per port.
                bg_in, bg_out, exp_in, exp_out, _, _ = get_entry_exit(session_df, trial)

                # Engaged time = time with the head in either port.
                t_bg = np.sum(bg_out - bg_in)
                t_exp = np.sum(exp_out - exp_in)

                # Total trial time (trial start to trial end).
                t_start = session_df[is_this_trial & (session_df.key == 'trial')
                                     & (session_df.value == 1)].session_time.values[0]
                t_end = session_df[is_this_trial & (session_df.key == 'trial')
                                   & (session_df.value == 0)].session_time.values[0]

                total_durations.append(t_end - t_start)
                engaged_durations.append(t_bg + t_exp)

                rewards_count += int(((session_df.key == 'reward') & (session_df.value == 1)
                                      & is_this_trial).sum())
            except IndexError:
                # Incomplete trial (no trial-start or no trial-end row).
                continue

        # Travel-time penalty: TRAVEL_TIME s per port transition, two per trial.
        sum_engaged = sum(engaged_durations) + (TRAVEL_TIME * 2 * len(total_durations))
        sum_total = sum(total_durations)

        if sum_total > 0:
            phase_metrics['pct_engaged'] = sum_engaged / sum_total
        if sum_engaged > 0:
            phase_metrics['realized_rr'] = rewards_count / sum_engaged
        phase_metrics['n_rewards'] = rewards_count

        # --- 4. Re-entry index ------------------------------------------
        # Context-port exits per trial. Ideal = 1; higher means the animal left
        # the context port before collecting all four rewards. Counted on trial
        # membership, so it is robust to `phase` being NaN on the exit row.
        num_actual = int((is_bg_exit & in_phase).sum())
        phase_metrics['reentry_index'] = num_actual / num_ideal

    return metrics_by_phase


# --- table assembly --------------------------------------------------------

def build_animal_metrics(animal_id, verbose=True):
    """Tidy metrics table for one animal, both segments, chronologically ordered."""
    sessions = list_animal_sessions(animal_id)

    if not sessions:
        print(f"No .parquet files found for {animal_id}.")
        return pd.DataFrame(columns=TIDY_COLUMNS)

    if verbose:
        n_pre = sum(s['session_type'] == 'pre-surgery' for s in sessions)
        n_post = len(sessions) - n_pre
        print(f"--- Computing metrics for {animal_id}: "
              f"{n_pre} pre-surgery + {n_post} post-surgery sessions ---")

    rows = []
    for session in sessions:
        try:
            session_df = pd.read_parquet(session['path'])
        except Exception as e:
            print(f"  Skipping session {session['session_idx']} "
                  f"({os.path.basename(session['path'])}): {e}")
            continue

        if session_df.empty:
            print(f"  Skipping empty session file: {os.path.basename(session['path'])}")
            continue

        try:
            metrics_by_phase = calculate_session_metrics(session_df)
        except Exception as e:
            print(f"  Skipping session {session['session_idx']} "
                  f"({os.path.basename(session['path'])}): {e}")
            continue

        for phase in PHASES:
            rows.append({
                'animal_id': animal_id,
                'session_idx': session['session_idx'],
                'session_date': session['session_date'],
                'session_type': session['session_type'],
                'phase': phase,
                **metrics_by_phase[phase],
            })

    return pd.DataFrame(rows, columns=TIDY_COLUMNS)


def build_metrics_table(animals=None, verbose=True):
    """Tidy metrics table for every animal, concatenated."""
    animals = ALL_ANIMALS if animals is None else animals

    per_animal = [build_animal_metrics(animal_id, verbose=verbose) for animal_id in animals]
    per_animal = [tbl for tbl in per_animal if not tbl.empty]

    if not per_animal:
        return pd.DataFrame(columns=TIDY_COLUMNS)

    metrics_df = pd.concat(per_animal, ignore_index=True)
    metrics_df = metrics_df.sort_values(['animal_id', 'session_idx', 'phase'],
                                        ignore_index=True)

    metrics_df['session_idx'] = metrics_df['session_idx'].astype(int)
    metrics_df['n_trials'] = metrics_df['n_trials'].astype(int)
    metrics_df['n_rewards'] = metrics_df['n_rewards'].astype(int)
    metrics_df['session_date'] = pd.to_datetime(metrics_df['session_date'])

    return metrics_df


def save_metrics_table(metrics_df, cache_path=None):
    cache_path = get_cache_path() if cache_path is None else cache_path
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    metrics_df.to_parquet(cache_path, index=False)
    print(f"Saved {len(metrics_df)} rows "
          f"({metrics_df['animal_id'].nunique()} animals) to {cache_path}")
    return cache_path


def load_metrics_table(animals=None, force_rebuild=False, cache_path=None):
    """
    Loads the cached tidy metrics table, rebuilding it from the .parquet session
    files if it is missing or if `force_rebuild=True`.

    This is the entry point every downstream analysis should call.
    """
    cache_path = get_cache_path() if cache_path is None else cache_path

    if not force_rebuild and os.path.exists(cache_path):
        metrics_df = pd.read_parquet(cache_path)
        print(f"Loaded cached metrics from {cache_path} ({len(metrics_df)} rows).")
    else:
        metrics_df = build_metrics_table(animals=animals)
        if not metrics_df.empty:
            save_metrics_table(metrics_df, cache_path=cache_path)

    if animals is not None:
        metrics_df = metrics_df[metrics_df['animal_id'].isin(animals)].reset_index(drop=True)

    return metrics_df


# --- quality-control summary ----------------------------------------------

def summarize_metrics_table(metrics_df):
    """One row per animal x segment: session counts, trial counts, NaN rates."""
    grouped = metrics_df.groupby(['animal_id', 'session_type'], sort=False)

    summary = grouped.agg(
        n_sessions=('session_idx', 'nunique'),
        first_session_idx=('session_idx', 'min'),
        last_session_idx=('session_idx', 'max'),
        median_n_trials=('n_trials', 'median'),
        total_rewards=('n_rewards', 'sum'),
    )

    for metric in METRIC_COLUMNS:
        summary[f'pct_nan_{metric}'] = grouped[metric].apply(
            lambda s: 100.0 * s.isna().mean()).round(1)

    return summary.reset_index()


# --- plotting (binning lives here and only here) ---------------------------

COLOR_LOW = 'tab:blue'      # replace with the constants from your plotting module
COLOR_HIGH = 'tab:red'

_PANEL_SPEC = [
    ('consumption', 'Consumption (s)'),
    ('realized_rr', 'Reward Rate (rew/s)'),
    ('pct_engaged', '% Engaged'),
    ('reentry_index', 'Re-entry Index'),
]


def plot_metrics_panel(axes, animal_id, metrics_df=None, bin_size=5,
                       session_type='pre-surgery'):
    """
    Reproduces the original four-panel figure from the cached table.

    `bin_size` is a *display* choice only -- it is applied here and never enters
    the cached table or the state-space model.
    """
    if metrics_df is None:
        metrics_df = load_metrics_table()

    animal_df = metrics_df[metrics_df['animal_id'] == animal_id]
    if session_type is not None:
        animal_df = animal_df[animal_df['session_type'] == session_type]

    if animal_df.empty:
        print(f"No metric data for {animal_id} ({session_type}).")
        return

    # Bin on rank of session_idx so bins are contiguous within the segment.
    session_order = {idx: rank for rank, idx
                     in enumerate(sorted(animal_df['session_idx'].unique()))}
    animal_df = animal_df.assign(
        group=animal_df['session_idx'].map(session_order) // bin_size)

    err_kws = dict(marker='o', capsize=3, markersize=5, linestyle='-', alpha=0.8)

    for ax, (metric, ylabel) in zip(axes, _PANEL_SPEC):
        for phase, color in (('0.4', COLOR_LOW), ('0.8', COLOR_HIGH)):
            phase_df = animal_df[animal_df['phase'] == phase]
            binned = phase_df.groupby('group')[metric]
            ax.errorbar(binned.mean().index, binned.mean(), yerr=binned.sem(),
                        color=color, label=PHASE_LABELS[phase], **err_kws)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(f"{bin_size}-session Group")

    axes[1].legend(fontsize='x-small')


def main():
    metrics_df = load_metrics_table(force_rebuild=True)
    print()
    print(summarize_metrics_table(metrics_df).to_string(index=False))


if __name__ == "__main__":
    main()