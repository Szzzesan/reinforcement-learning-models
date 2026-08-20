import os
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from data_loader import load_pooled_transitions
from src.mouse_playback_agent import MousePlaybackAgent


def calculate_trials_to_stability(metric_array, session_array, context_array, window_size=5, threshold_ratio=0.15,
                                  consecutive_stable_needed=10, reference_percentile=0.95):
    """
    Calculates rolling std and independent convergence trials.
    Uses a percentile-based reference to ignore massive outliers.
    """
    df = pd.DataFrame({
        'metric': metric_array,
        'session': session_array,
        'context': context_array
    })

    # 1. Calculate rolling standard deviation strictly WITHIN each session AND context
    df['rolling_std'] = df.groupby(['session', 'context'])['metric'].transform(
        lambda x: x.rolling(window=window_size, min_periods=window_size).std()
    )

    convergence_trials = {}

    # 2. Evaluate stability separately for Low (0.0) and High (1.0)
    for ctx in [0.0, 1.0]:
        ctx_mask = df['context'] == ctx
        if not ctx_mask.any():
            convergence_trials[ctx] = None
            continue

        ctx_df = df[ctx_mask].copy()

        # --- THE FIX: Use the 95th percentile instead of the max ---
        reference_volatility = ctx_df['rolling_std'].quantile(reference_percentile)
        threshold = reference_volatility * threshold_ratio

        # Flag stable trials
        ctx_df['is_stable'] = ctx_df['rolling_std'] <= threshold

        # Find streaks of consecutive stable trials WITHIN this context
        ctx_df['consecutive_stable'] = ctx_df['is_stable'].rolling(window=consecutive_stable_needed).sum()

        # Find where the streak requirement is met
        stable_points = ctx_df.index[ctx_df['consecutive_stable'] == consecutive_stable_needed].tolist()

        if stable_points:
            streak_end_idx = stable_points[0]
            pos = ctx_df.index.get_loc(streak_end_idx)
            start_pos = pos - consecutive_stable_needed + 1
            convergence_trials[ctx] = ctx_df.index[start_pos]
        else:
            print(f"⚠️ Warning: Context {ctx} never reached the strict stability threshold.")
            convergence_trials[ctx] = len(df)  # Return max trials as penalty

    convergence_low = convergence_trials.get(0.0)
    convergence_high = convergence_trials.get(1.0)

    return convergence_low, convergence_high, df


def plot_trial_rolling_volatility(df, convergence_low=None, convergence_high=None, title="Evolution of Rolling Volatility"):
    """
    Plots the rolling standard deviation split by Low and High context, marking convergences.
    """
    plt.figure(figsize=(10, 5))

    # 1. Setup Colors
    palette = sns.color_palette('Set2', 2)
    color_low = palette[0]
    color_high = palette[1]

    # 2. Split Data
    df_low = df[df['context'] == 0.0]
    df_high = df[df['context'] == 1.0]

    # 3. Plot the rolling standard deviations
    plt.plot(df_low.index, df_low['rolling_std'], color=color_low, linewidth=2, label='Low Context (0)')
    plt.plot(df_high.index, df_high['rolling_std'], color=color_high, linewidth=2, label='High Context (1)')

    # 4. Find and plot session boundaries
    session_starts = df.drop_duplicates(subset=['session']).index.tolist()
    for idx in session_starts:
        if idx != 0:
            plt.axvline(x=idx - 0.5, color='gray', linestyle='--', alpha=0.7)

    # 5. Setup Legend Handles
    handles = [
        mlines.Line2D([], [], color=color_low, linewidth=2, label='Low Context'),
        mlines.Line2D([], [], color=color_high, linewidth=2, label='High Context'),
        mlines.Line2D([], [], color='gray', linestyle='--', alpha=0.7, label='Session Boundary')
    ]

    # 6. Mark the Convergence Trials
    if convergence_low is not None and convergence_low < len(df):
        y_val_low = df.loc[convergence_low, 'rolling_std']
        plt.scatter([convergence_low], [y_val_low], color=color_low, edgecolor='black', s=200, zorder=5, marker='*')
        handles.append(mlines.Line2D([], [], color='w', marker='*', markerfacecolor=color_low, markeredgecolor='black',
                                     markersize=14, label='Low Converged'))

    if convergence_high is not None and convergence_high < len(df):
        y_val_high = df.loc[convergence_high, 'rolling_std']
        plt.scatter([convergence_high], [y_val_high], color=color_high, edgecolor='black', s=200, zorder=5, marker='*')
        handles.append(mlines.Line2D([], [], color='w', marker='*', markerfacecolor=color_high, markeredgecolor='black',
                                     markersize=14, label='High Converged'))

    # Formatting
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Cumulative Trial Number", fontsize=12)
    plt.ylabel("Rolling Std Dev", fontsize=12)
    plt.legend(handles=handles, frameon=False, loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.xlim(0, 1000)
    plt.tight_layout()
    plt.show()


def extract_behavioral_metrics(transitions):
    """
    Extracts Stay Durations, Session IDs, and Contexts for every trial.
    """
    stay_durations = []
    session_ids = []
    contexts = []  # <--- NEW

    current_session = 0
    is_tracking = False

    for i in range(len(transitions)):
        obs_t, _, reward, obs_next, terminal = transitions[i]

        in_port_1 = (obs_t[0] == 1.0)
        is_enabled = (obs_t[5] == 0.0)

        # 1. Detect Entry
        if in_port_1 and is_enabled and not is_tracking:
            is_tracking = True

        # 2. Detect Exit
        if is_tracking and (obs_next[0] != 1.0 or terminal):
            stay_duration = obs_t[1]
            context = obs_t[3]  # <--- NEW: Context is index 3 in the state vector

            stay_durations.append(stay_duration)
            session_ids.append(current_session)
            contexts.append(context)  # <--- NEW

            is_tracking = False

        # 3. Detect Session Boundary
        if terminal:
            current_session += 1
            is_tracking = False

    return stay_durations, session_ids, contexts


def calculate_session_stats(metric_array, session_array, context_array):
    """
    Calculates the Mean, Standard Deviation (STD), and Standard Error of the Mean (SEM)
    of a metric for each session, split by context.
    """
    df = pd.DataFrame({
        'metric': metric_array,
        'session': session_array,
        'context': context_array
    })

    # Group by session and context, then aggregate the statistics
    session_stats = df.groupby(['session', 'context'])['metric'].agg(
        mean='mean',
        std='std',
        sem='sem',
        n_trials='count'  # Always good to know how many trials went into the math
    ).reset_index()

    return session_stats

def calculate_session_stats_unsplit(metric_array, session_array):
    """
    Calculates the Mean, Standard Deviation (STD), and Standard Error of the Mean (SEM)
    of a metric for each session, split by context.
    """
    df = pd.DataFrame({
        'metric': metric_array,
        'session': session_array
    })

    # Group by session and context, then aggregate the statistics
    session_stats = df.groupby(['session'])['metric'].agg(
        mean='mean',
        std='std',
        sem='sem',
        n_trials='count'  # Always good to know how many trials went into the math
    ).reset_index()

    return session_stats


def find_convergence_session(session_stats, window_size=5, threshold_ratio=0.15, consecutive_sessions=3):
    """
    Finds the session where the animal's rolling STD officially converges.

    Args:
        session_stats (pd.DataFrame): The dataframe from `calculate_session_stats`.
        window_size (int): The window for the rolling STD mean (default 5).
        threshold_ratio (float): The fraction of the learnable range to use as a buffer (default 15%).
        consecutive_sessions (int): How many sessions it must stay below threshold to count.

    Returns:
        conv_low (int): The session number where Low context converged.
        conv_high (int): The session number where High context converged.
    """
    convergence_sessions = {}

    for ctx in [0.0, 1.0]:
        # 1. Isolate and sort the data for this context
        ctx_data = session_stats[session_stats['context'] == ctx].sort_values('session').copy()

        if ctx_data.empty:
            convergence_sessions[ctx] = None
            continue

        # 2. Calculate the smoothed rolling STD (matches the plotting function exactly)
        ctx_data['rolling_std'] = ctx_data['std'].rolling(window=window_size, min_periods=1).mean()

        # 3. Find the Exploratory Peak
        peak_std = ctx_data['rolling_std'].max()

        # 4. Find the True Baseline (Average of the final 5 smoothed sessions)
        true_baseline = ctx_data['rolling_std'].tail(5).mean()

        # 5. Calculate the Threshold
        learnable_range = peak_std - true_baseline

        # Safeguard: If the curve went up instead of down (failed to learn at all)
        if learnable_range <= 0:
            threshold = true_baseline
        else:
            threshold = true_baseline + (threshold_ratio * learnable_range)

        # 6. Flag stable sessions
        ctx_data['is_stable'] = ctx_data['rolling_std'] <= threshold

        # 7. Find the streak
        ctx_data['streak'] = ctx_data['is_stable'].rolling(window=consecutive_sessions).sum()

        # Find all sessions where the streak requirement was met
        streak_ends = ctx_data[ctx_data['streak'] >= consecutive_sessions]

        if not streak_ends.empty:
            # Get the exact index of the FIRST time the streak finished
            first_streak_end_idx = streak_ends.index[0]

            # Find its positional row number to step backward to the START of the streak
            pos = ctx_data.index.get_loc(first_streak_end_idx)
            start_pos = max(0, pos - consecutive_sessions + 1)

            # Record the actual session number
            convergence_sessions[ctx] = ctx_data.iloc[start_pos]['session']
        else:
            print(
                f"⚠️ Warning: Context {ctx} never maintained stability for {consecutive_sessions} consecutive sessions.")
            convergence_sessions[ctx] = None

    return convergence_sessions.get(0.0), convergence_sessions.get(1.0)


def find_convergence_session_unsplit(session_stats, window_size=10, threshold_ratio=0.15, consecutive_sessions=3):
    """
    Finds the session where the animal's rolling STD officially converges.

    Args:
        session_stats (pd.DataFrame): The dataframe from `calculate_session_stats`.
        window_size (int): The window for the rolling STD mean (default 5).
        threshold_ratio (float): The fraction of the learnable range to use as a buffer (default 15%).
        consecutive_sessions (int): How many sessions it must stay below threshold to count.

    Returns:
        conv (int): The session number where leave time converged.
    """

    # 2. Calculate the smoothed rolling STD (matches the plotting function exactly)
    session_stats['rolling_std'] = session_stats['std'].rolling(window=window_size, min_periods=1).mean()

    # 3. Find the Exploratory Peak
    peak_std = session_stats['rolling_std'].max()

    # 4. Find the True Baseline (Average of the final 5 smoothed sessions)
    true_baseline = session_stats['rolling_std'].tail(5).mean()

    # 5. Calculate the Threshold
    learnable_range = peak_std - true_baseline

    # Safeguard: If the curve went up instead of down (failed to learn at all)
    if learnable_range <= 0:
        threshold = true_baseline
    else:
        threshold = true_baseline + (threshold_ratio * learnable_range)

    # 6. Flag stable sessions
    session_stats['is_stable'] = session_stats['rolling_std'] <= threshold

    # 7. Find the streak
    session_stats['streak'] = session_stats['is_stable'].rolling(window=consecutive_sessions).sum()

    # Find all sessions where the streak requirement was met
    streak_ends = session_stats[session_stats['streak'] >= consecutive_sessions]

    if not streak_ends.empty:
        # Get the exact index of the FIRST time the streak finished
        first_streak_end_idx = streak_ends.index[0]

        # Find its positional row number to step backward to the START of the streak
        pos = session_stats.index.get_loc(first_streak_end_idx)
        start_pos = max(0, pos - consecutive_sessions + 1)

        # Record the actual session number
        convergence_session = session_stats.iloc[start_pos]['session']
    else:
        print(
            f"⚠️ Warning: never maintained stability for {consecutive_sessions} consecutive sessions.")
        convergence_session = None

    return convergence_session


def plot_session_stats(session_stats, conv_low=None, conv_high=None, window_size=10, title="Evolution of Stay Duration by Session"):
    """
        Creates a 3-panel plot:
        1. Mean +/- SEM
        2. Raw Standard Deviation (STD)
        3. 5-Session Rolling Average of the STD
        """

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. Setup Colors
    palette = sns.color_palette('Set2', 2)
    color_low = palette[0]
    color_high = palette[1]

    # 2. Split and Sort Data (Sorting ensures rolling math is applied chronologically)
    low_stats = session_stats[session_stats['context'] == 0.0].sort_values('session')
    high_stats = session_stats[session_stats['context'] == 1.0].sort_values('session')

    # --- PANEL 1: Mean and SEM ---
    ax1.plot(low_stats['session'], low_stats['mean'], color=color_low, marker='o', linewidth=2, label='Low Context (0)')
    ax1.fill_between(low_stats['session'],
                     low_stats['mean'] - low_stats['sem'],
                     low_stats['mean'] + low_stats['sem'],
                     color=color_low, alpha=0.3)

    ax1.plot(high_stats['session'], high_stats['mean'], color=color_high, marker='o', linewidth=2,
             label='High Context (1)')
    ax1.fill_between(high_stats['session'],
                     high_stats['mean'] - high_stats['sem'],
                     high_stats['mean'] + high_stats['sem'],
                     color=color_high, alpha=0.3)

    ax1.set_ylabel("Mean $\pm$ SEM", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3)

    # --- PANEL 2: Raw Standard Deviation (Volatility) ---
    ax2.plot(low_stats['session'], low_stats['std'], color=color_low, marker='s', linestyle='--', linewidth=2)
    ax2.plot(high_stats['session'], high_stats['std'], color=color_high, marker='s', linestyle='--', linewidth=2)

    ax2.set_ylabel("Raw STD", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # --- PANEL 3: Rolling Average of STD ---
    # Calculate the rolling mean of the STD column
    low_stats['rolling_std'] = low_stats['std'].rolling(window=window_size, min_periods=1).mean()
    high_stats['rolling_std'] = high_stats['std'].rolling(window=window_size, min_periods=1).mean()

    # Update the plot calls to use the new columns
    ax3.plot(low_stats['session'], low_stats['rolling_std'], color=color_low, marker='D', linewidth=2)
    ax3.plot(high_stats['session'], high_stats['rolling_std'], color=color_high, marker='D', linewidth=2)

    # --- MARK CONVERGENCE STARS & LINES ---
    if conv_low is not None:
        y_val_low = low_stats.loc[low_stats['session'] == conv_low, 'rolling_std'].iloc[0]
        ax3.scatter([conv_low], [y_val_low], color=color_low, edgecolor='black', s=250, zorder=5, marker='*')
        # Drop a vertical line across all axes
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=conv_low, color=color_low, linestyle=':', alpha=0.8, linewidth=2)

    if conv_high is not None:
        y_val_high = high_stats.loc[high_stats['session'] == conv_high, 'rolling_std'].iloc[0]
        ax3.scatter([conv_high], [y_val_high], color=color_high, edgecolor='black', s=250, zorder=5, marker='*')
        # Drop a vertical line across all axes
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=conv_high, color=color_high, linestyle=':', alpha=0.8, linewidth=2)

    ax3.set_ylabel(f"{window_size}-Session\nRolling Mean of STD", fontsize=12)
    ax3.set_xlabel("Session Number", fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Force X-axis to display strictly integer session numbers
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()

def plot_session_stats_unsplit(session_stats, conv=None, window_size=10, title="Evolution of Stay Duration by Session"):
    """
        Creates a 3-panel plot for unsplit data:
        1. Mean +/- SEM
        2. Raw Standard Deviation (STD)
        3. 5-Session Rolling Average of the STD
        Marks the exact session of convergence with a star and a vertical line.
        """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. Setup Color (Using the first color from Set2 to keep the aesthetic)
    color_main = 'blue'

    # 2. Sort Data (Sorting ensures rolling math is applied chronologically)
    stats = session_stats.sort_values('session').copy()

    # --- PANEL 1: Mean and SEM ---
    ax1.plot(stats['session'], stats['mean'], color=color_main, marker='o', linewidth=2, label='Global Mean')
    ax1.fill_between(stats['session'],
                     stats['mean'] - stats['sem'],
                     stats['mean'] + stats['sem'],
                     color=color_main, alpha=0.3)

    ax1.set_ylabel("Mean $\pm$ SEM", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3)

    # --- PANEL 2: Raw Standard Deviation ---
    ax2.plot(stats['session'], stats['std'], color=color_main, marker='s', linestyle='--', linewidth=2)

    ax2.set_ylabel("Raw STD", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # --- PANEL 3: Rolling Average of STD ---
    # Store rolling std directly in the dataframe for easy lookups
    stats['rolling_std'] = stats['std'].rolling(window=window_size, min_periods=1).mean()

    ax3.plot(stats['session'], stats['rolling_std'], color=color_main, marker='D', linewidth=2)

    # --- MARK CONVERGENCE STAR & LINES ---
    if conv is not None:
        # Find the specific y-value for the rolling std exactly at the convergence session
        y_val = stats.loc[stats['session'] == conv, 'rolling_std'].iloc[0]
        ax3.scatter([conv], [y_val], color=color_main, edgecolor='black', s=250, zorder=5, marker='*')

        # Drop a vertical line across all axes
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=conv, color=color_main, linestyle=':', alpha=0.8, linewidth=2)

    ax3.set_ylabel(f"{window_size}-Session\nRolling Mean of STD", fontsize=12)
    ax3.set_xlabel("Session Number", fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Force X-axis to display strictly integer session numbers
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()

def run_agent_and_get_volatility(alpha, gamma, transitions):
    """
    Runs a fresh agent through all transitions and calculates the trial-by-trial
    Absolute Change in Value (|Delta V|) to mimic biological volatility.
    """
    # 1. Initialize a fresh agent with the current candidate alpha
    scales = [-1, 2.0, 0.0, -1, 0.0, -1]

    # Agent Setup
    agent = MousePlaybackAgent()
    agent_params = {
        "discount": gamma,
        "step_size": alpha,
        "num_tilings": 8,
        "iht_size": 32768,
        "gambling_max_time_s": 30.0,
        "context_rewards_max": 4,
        "scales": scales
    }
    agent.agent_init(agent_params)

    delta_vs = []
    session_ids = []
    contexts = []

    current_session = 0
    is_tracking = False

    # We need to remember the value from the previous trial to calculate Delta V
    prev_v_low = 0.0
    prev_v_high = 0.0

    for obs_t, action, reward, obs_next, terminal in transitions:

        in_port = (obs_t[0] == 1.0)
        is_enabled = (obs_t[5] == 0.0)
        context = obs_t[3]

        # Detect Entry (Start of Trial)
        if in_port and is_enabled and not is_tracking:
            is_tracking = True

            # Get the agent's current value prediction for this entry state
            current_v = agent.get_value(obs_t)  # Adjust this method call to match your agent

            # Calculate how much the value changed since the LAST trial of this context
            if context == 0.0:
                delta_v = abs(current_v - prev_v_low)
                prev_v_low = current_v
            else:
                delta_v = abs(current_v - prev_v_high)
                prev_v_high = current_v

            delta_vs.append(delta_v)
            session_ids.append(current_session)
            contexts.append(context)

        # Detect Exit (End of Trial)
        if is_tracking and (obs_next[0] != 1.0 or terminal):
            is_tracking = False

        # Agent learns from every step!
        agent.update(obs_t, action, reward, obs_next, terminal)  # Adjust to match your agent

        if terminal:
            current_session += 1
            is_tracking = False

    return delta_vs, session_ids, contexts


def get_session_weight_displacement(alpha, gamma, transitions):
    """
    Runs the agent and calculates the Net Weight Displacement (sum of absolute changes)
    between the start and end of every session.
    """
    # 1. Initialize a fresh agent with the current candidate alpha
    scales = [-1, 2.0, 0.0, -1, 0.0, -1]

    # Agent Setup
    agent = MousePlaybackAgent()
    agent_params = {
        "discount": gamma,
        "step_size": alpha,
        "num_tilings": 8,
        "iht_size": 32768,
        "gambling_max_time_s": 30.0,
        "context_rewards_max": 4,
        "scales": scales
    }
    agent.agent_init(agent_params)

    session_displacements = []

    need_start = True
    current_session = 0

    # 1. Take a snapshot of the initial weights before any learning happens
    # (np.copy is crucial here, otherwise it just references the changing array!)
    previous_w = np.copy(agent.w)

    for obs_t, action, reward, obs_next, terminal in transitions:

        if need_start:
            agent.agent_start(obs_t)
            need_start = False

        if terminal:
            # Agent processes the terminal step
            agent.agent_end(reward)
            need_start = True

            # --- END OF SESSION: Calculate Net Weight Displacement ---
            current_w = agent.w

            # Calculate the global shift across all 32,768 weights
            displacement = np.sum(np.abs(current_w - previous_w))
            session_displacements.append(displacement)

            # Take a new snapshot for the upcoming session
            previous_w = np.copy(current_w)
            current_session += 1

        else:
            # Agent processes a normal step
            agent.agent_step(reward, obs_next)

    # 2. Format directly into a DataFrame mimicking biological session stats
    # We map the displacement to the 'std' column so it plugs directly into
    # your find_convergence_session_unsplit() function!
    agent_session_stats = pd.DataFrame({
        'session': range(len(session_displacements)),
        'std': session_displacements
    })

    return agent_session_stats


def objective_alpha_unsplit(alpha, gamma, transitions, bio_conv):
    """
    The objective function for fitting Alpha using the global weight displacement method.
    Returns the squared error between Agent convergence and Biological convergence.
    """
    # 1. Get the Agent's global weight displacements per session
    agent_stats = get_session_weight_displacement(
        alpha, gamma, transitions
    )

    # 2. Find when the Agent globally converged
    # (Using the exact parameters you fine-tuned for your biological data)
    agent_conv = find_convergence_session_unsplit(
        agent_stats, window_size=10, threshold_ratio=0.20, consecutive_sessions=4
    )

    # 3. Penalty Handling
    # If the agent never converges (e.g., alpha is too high and it oscillates,
    # or too low and it never learns), apply a massive penalty.
    max_session = agent_stats['session'].max()
    if agent_conv is None:
        agent_conv = max_session * 2

    # 4. Calculate Loss (Squared Error)
    loss = (bio_conv - agent_conv) ** 2

    print(f"Testing Alpha: {alpha:.5f} | Agent Conv: {agent_conv} | Loss: {loss}")

    return loss


def plot_bio_vs_agent_convergence(bio_stats, agent_stats, bio_conv, agent_conv, window_size=10,
                                  title="Biological vs. Agent Learning Trajectories"):
    """
    Creates a 2-panel plot comparing animal behavioral volatility (top)
    with agent neural network weight shifts (bottom).
    Marks the mathematical convergence session on both curves.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 1. Setup Colors (Using two distinct colors from Set2 for contrast)
    palette = sns.color_palette('Set2')
    color_bio = palette[0]  # Teal-ish
    color_agent = palette[1]  # Orange-ish

    # 2. Sort Data strictly by session
    bio_stats = bio_stats.sort_values('session').copy()
    agent_stats = agent_stats.sort_values('session').copy()

    # 3. Calculate the rolling averages directly
    bio_stats['rolling_std'] = bio_stats['std'].rolling(window=window_size, min_periods=1).mean()
    agent_stats['rolling_std'] = agent_stats['std'].rolling(window=window_size, min_periods=1).mean()

    # --- PANEL 1: Biological Animal Data ---
    ax1.plot(bio_stats['session'], bio_stats['rolling_std'], color=color_bio, marker='D', linewidth=2,
             label='Animal Stay Duration (Rolling STD)')

    if bio_conv is not None:
        # Find the specific y-value for the star
        if bio_conv in bio_stats['session'].values:
            y_val_bio = bio_stats.loc[bio_stats['session'] == bio_conv, 'rolling_std'].iloc[0]
            ax1.scatter([bio_conv], [y_val_bio], color=color_bio, edgecolor='black', s=250, zorder=5, marker='*')

        # Drop a vertical line to highlight the day
        ax1.axvline(x=bio_conv, color=color_bio, linestyle=':', alpha=0.8, linewidth=2)

    ax1.set_ylabel(f"Bio Volatility\n({window_size}-Session Rolling)", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False, loc='upper right')

    # --- PANEL 2: Artificial Agent Data ---
    ax2.plot(agent_stats['session'], agent_stats['rolling_std'], color=color_agent, marker='D', linewidth=2,
             label='Agent Weight Displacement (Rolling)')

    if agent_conv is not None:
        # Handle safety edge case: if agent never converged and was heavily penalized beyond max sessions
        if agent_conv in agent_stats['session'].values:
            y_val_agent = agent_stats.loc[agent_stats['session'] == agent_conv, 'rolling_std'].iloc[0]
            ax2.scatter([agent_conv], [y_val_agent], color=color_agent, edgecolor='black', s=250, zorder=5, marker='*')

        # Drop a vertical line to highlight the day
        ax2.axvline(x=agent_conv, color=color_agent, linestyle=':', alpha=0.8, linewidth=2)

    ax2.set_ylabel(f"Agent Volatility\n({window_size}-Session Rolling)", fontsize=12)
    ax2.set_xlabel("Session Number", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False, loc='upper right')

    # Force X-axis to display strictly integer session numbers
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()


def main():
    alphas_to_test = np.round(np.arange(0.0001, 0.0011, 0.0001), 4)

    PROJECT_ROOT = Path(os.getcwd()).parent
    DATA_FOLDER = PROJECT_ROOT / "data"

    animal = "SZ036"
    transitions = load_pooled_transitions(DATA_FOLDER, animal)
    # 1. Provide your established biological baselines (e.g., from your previous plot)
    animal_durations, animal_sessions, contexts = extract_behavioral_metrics(transitions)
    ## --- looking for stability on a session-by-session basis ---
    stats_df = calculate_session_stats(animal_durations, animal_sessions, contexts)
    bio_conv_low, bio_conv_high = find_convergence_session(stats_df, threshold_ratio=0.2, consecutive_sessions=4)
    fixed_gamma = 0.7

    best_alpha = None
    lowest_loss = float('inf')
    results_log = {}

    print(f"🚀 Starting Grid Search for {len(alphas_to_test)} alpha values...")

    # 3. Simply loop through your specific list
    for current_alpha in alphas_to_test:

        # Calculate the loss for this specific alpha
        loss = objective_alpha(
            alpha=current_alpha,
            gamma=fixed_gamma,
            transitions=transitions,
            bio_conv_low=bio_conv_low,
            bio_conv_high=bio_conv_high
        )

        # Log the result so you can see the whole curve later if you want
        results_log[current_alpha] = loss

        # Check if this is the best one we've seen so far
        if loss < lowest_loss:
            lowest_loss = loss
            best_alpha = current_alpha

    print("\n✅ Grid Search Complete!")
    print(f"🏆 Best Alpha: {best_alpha} (Loss: {lowest_loss})")

    # Optional: Print all results to see how sensitive the model is
    print("\n📊 Full Results:")
    for a, l in results_log.items():
        print(f"Alpha: {a:.4f} | Loss: {l}")


if __name__ == '__main__':
    # main()
    # --- SINGLE ANIMAL TEST ---
    PROJECT_ROOT = Path(os.getcwd()).parent
    DATA_FOLDER = PROJECT_ROOT / "data"

    animal = "SZ036"
    SZ_animals = ['SZ036', 'SZ037', 'SZ038', 'SZ039', 'SZ042', 'SZ043']
    RK_animals = ['RK007', 'RK008']
    animal_list = SZ_animals + RK_animals
    for animal in animal_list:
        best_alpha = 0.1
        fixed_gamma = 0.7
        transitions = load_pooled_transitions(DATA_FOLDER, animal)
        # --- USAGE EXAMPLE ---
        animal_durations, animal_sessions, contexts = extract_behavioral_metrics(transitions)

        ## --- looking for stability on a session-by-session basis ---
        ## split by context
        # stats_df = calculate_session_stats(animal_durations, animal_sessions, contexts)
        # conv_low, conv_high = find_convergence_session(stats_df, threshold_ratio=0.2, consecutive_sessions=4)
        # plot_session_stats(stats_df, conv_low, conv_high, window_size=5, title=f"{animal}: Evolution of Stay Duration by Session")

        ## not split
        bio_stats_unsplit = calculate_session_stats_unsplit(animal_durations, animal_sessions)
        bio_conv = find_convergence_session_unsplit(bio_stats_unsplit, window_size=10)
        # plot_session_stats_unsplit(bio_stats_unsplit, bio_conv, window_size=10, title=f"{animal}: Evolution of Stay Duration by Session")
        # Plotting agent and biological learning curves together:
        optimal_agent_stats = get_session_weight_displacement(best_alpha, fixed_gamma, transitions)
        optimal_agent_conv = find_convergence_session_unsplit(optimal_agent_stats)
        #
        plot_bio_vs_agent_convergence(bio_stats_unsplit, optimal_agent_stats, bio_conv, optimal_agent_conv)

        ## --- looking for stability on a trial-by-trial basis ---
        # convergence_low, convergence_high, animal_df = calculate_trials_to_stability(
        #     metric_array=animal_durations,
        #     session_array=animal_sessions,
        #     context_array=contexts,
        #     window_size=5,
        #     threshold_ratio=0.2,
        #     consecutive_stable_needed=8,
        #     reference_percentile=0.99
        # )
        # print(
        #     f"Animal converged at cumulative trial: \n{convergence_low} in Low context; \n{convergence_high} in High context")
        # plot_rolling_volatility(animal_df,
        #                         convergence_low=convergence_low,
        #                         convergence_high=convergence_high,
        #                         title="Evolution of Behavioral Variability")
