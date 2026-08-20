import sys
import os
from pathlib import Path
import json
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import tiles3 as tc

from src.state_utils import build_investment_sim_state, build_travel_state, is_investment_state
import src.config as config
from mouse_playback_environment import MousePlaybackEnvironment


def extract_target_session_trajectories(target_data_file, pretrained_agent_file):
    """
    Evaluates a frozen agent on target sessions.remote
    """
    # 1. Load the pre-trained agent and FREEZE learning
    print(f"Loading agent from {pretrained_agent_file}...")
    with open(pretrained_agent_file, 'rb') as f:
        agent = pickle.load(f)
    agent.step_size = 0.0  # Crucial: Prevent weight updates during target evaluation!

    # 2. Load the target session data
    print(f"Loading target transitions from {target_data_file}...")
    with open(target_data_file, 'rb') as f:
        target_transitions_with_meta = pickle.load(f)
    target_transitions = [t[:5] for t in target_transitions_with_meta]

    # 3. Setup the playback environment
    env = MousePlaybackEnvironment()
    env.env_init({"transitions": target_transitions, "time_step_duration": 0.1})

    num_total_steps = len(env.transitions)

    trials = []
    in_gambling_trial = False
    current_trial_vs = []
    current_trial_times = []
    current_event_timer = []
    current_context = None
    last_gambling_obs = None  # To hold the state right before leaving

    obs = env.env_start()
    agent.agent_start(obs)

    # --- Data Extraction Loop ---
    for step_idx in range(num_total_steps):
        reward, next_obs, terminal = env.env_step(action=None)
        v_current = agent.get_value(obs)

        is_gambling = is_investment_state(obs)

        if is_gambling:
            if not in_gambling_trial:  # the first state in gambling port
                in_gambling_trial = True
                current_trial_vs = []
                current_trial_times = []
                current_event_timer = []
                current_context = obs[3]

            current_trial_vs.append(v_current)
            current_trial_times.append(obs[1])
            current_event_timer.append(obs[2])
            last_gambling_obs = obs  # Save this for the threshold calculation!
        else:
            if in_gambling_trial:  # just left the gambling port
                theoretical_leave_state = build_travel_state(last_gambling_obs)
                v_after_leaving = agent.get_value(theoretical_leave_state)

                # Save the completed trial
                if len(current_trial_vs) > 0:
                    trials.append({
                        'times': current_trial_times,
                        'event_timer': current_event_timer,
                        'values': current_trial_vs,
                        'context': current_context,
                        'v_after': v_after_leaving
                    })
                in_gambling_trial = False

        if terminal:
            agent.agent_end(reward)
            if env.current_step_index < num_total_steps:
                obs = next_obs
                agent.agent_start(obs)
            else:
                break
        else:
            agent.agent_step(reward, next_obs)
            obs = next_obs

    print(f"Extraction complete. Found {len(trials)} gambling trials.")
    return trials if len(trials) > 0 else None


def save_trajectory_data(trials, animal_id):
    """Saves trajectory data to the outputs folder."""
    project_root = Path(config.MODELING_PROJECT_ROOT)
    save_dir = project_root / Path(config.STEP3_EVALUATION_METRICS_SUBDIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / f"target_session_value_trajectory_{animal_id}.json"
    with open(file_path, 'w') as f:
        json.dump(trials, f, indent=4)
    print(f"Trajectory data saved to {file_path}")


def load_trajectory_data(animal_id):
    """Loads trajectory data from the outputs folder."""
    project_root = Path(config.MODELING_PROJECT_ROOT)
    file_path = project_root / Path(config.STEP3_EVALUATION_METRICS_SUBDIR) / f"target_session_value_trajectory_{animal_id}.json"

    with open(file_path, 'r') as f:
        return json.load(f)


def plot_target_session_trajectories(trials, animal_id):
    """
    Plots a 5x4 grid of 20 random
    gambling trials, showing the V(s) trajectory and the V(after_leaving) threshold.
    """

    # Randomly sample 20 trials (or all if less than 20)
    sampled_trials = random.sample(trials, min(20, len(trials)))

    # Calculate global Y limits for standardized subplots
    all_vs = [v for t in sampled_trials for v in t['values']] + [t['v_after'] for t in trials]
    y_min, y_max = min(all_vs) - 0.5, max(all_vs) + 0.5

    fig, axes = plt.subplots(5, 4, figsize=(13, 16))
    fig.suptitle(f"{animal_id}: Agent V(s) Trajectories vs. Leave Threshold", fontsize=16)
    axes = axes.flatten()

    for i, trial in enumerate(sampled_trials):
        ax = axes[i]

        # Color based on Context (Adjust values to match your data encoding)
        is_high_context = np.isclose(trial['context'], 1)
        color = sns.color_palette('Set2')[1] if is_high_context else sns.color_palette('Set2')[
            0]  # Orange for High, Green for Low
        ctx_label = "High" if is_high_context else "Low"

        # Plot V(s) trajectory
        ax.plot(trial['times'], trial['values'], color=color, linewidth=2.5, label='V(stay)')

        # Plot V(after_leaving) as a horizontal dashed line
        ax.axhline(y=trial['v_after'], color='grey', linestyle='--', linewidth=1.5, label='V(leave)')

        ax.set_ylim(-0.1, 0.5)
        ax.set_title(f"Trial {i + 1} ({ctx_label} Ctx)", fontsize=10)

        if i >= 16:  # Only label bottom row x-axis
            ax.set_xlabel("Time in Port (s)")
        else:
            ax.set_xticklabels([])
        if i % 4 == 0:  # Only label leftmost column y-axis
            ax.set_ylabel("State Value V(s)")
        else:
            ax.set_yticklabels([])

    # Add a single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
    plt.show()


def plot_target_session_trajectories_with_mc(trials, agent, animal_id, reward_prob_func,
                                             trial_indices=None, num_mc_traces=10, dt=0.1, max_extrap_s=30):
    """
    Plots a grid of random gambling trials with shared X and Y axes.
    Runs Monte Carlo rollouts to extrapolate V(s) trajectories if needed.
    """

    if trial_indices is not None:
        valid_indices = [i for i in trial_indices if i < len(trials)]
        sampled_trials = [trials[i] for i in valid_indices]
        display_numbers = [i + 1 for i in valid_indices]
    else:
        sample_size = min(20, len(trials))
        valid_indices = sorted(random.sample(range(len(trials)), sample_size))
        sampled_trials = [trials[i] for i in valid_indices]
        display_numbers = [i + 1 for i in valid_indices]

    num_plots = len(sampled_trials)
    if num_plots == 0:
        print("No valid trials to plot.")
        return

    cols = 4
    rows = math.ceil(num_plots / cols)

    # ---> MODIFICATION 1: Added sharey=True to lock all Y-axes to the same scale <---
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows), sharex=True, sharey=True)
    fig.suptitle(f"{animal_id}: Agent V(s) Trajectories with MC Extrapolations", fontsize=16)
    axes = np.atleast_1d(axes).flatten()

    for i, (trial, trial_num) in enumerate(zip(sampled_trials, display_numbers)):
        ax = axes[i]

        is_high_context = np.isclose(trial['context'], 1)
        color = sns.color_palette('Set2')[1] if is_high_context else sns.color_palette('Set2')[0]
        ctx_label = "High" if is_high_context else "Low"
        threshold = trial['v_after']

        # 1. Plot Actual V(s) trajectory
        ax.plot(trial['times'], trial['values'], color=color, linewidth=2.5, label='Actual V(stay)')

        # 2. Plot V(leave) Threshold
        ax.axhline(y=threshold, color='grey', linestyle='--', linewidth=1.5, label='V(leave)')

        # ---> MODIFICATION 3: Print the threshold value above the line <---
        ax.text(0.02, threshold, f" V = {threshold:.3f}", color='dimgray',
                va='bottom', ha='left', fontsize=8, fontweight='bold',
                transform=ax.get_yaxis_transform())

        # 3. Handle actual crossing and Monte Carlo predictions
        actual_time = trial['times'][-1]
        actual_crossed = any(v <= threshold for v in trial['values'])

        pred_time = None

        if not actual_crossed:
            last_time = trial['times'][-1]
            last_event_timer = trial['event_timer'][-1]
            context = trial['context']

            mc_leave_times = []  # Keep track of when universes cross

            for sim in range(num_mc_traces):
                sim_times = [last_time]
                sim_values = [trial['values'][-1]]

                current_time = last_time
                current_event_timer = last_event_timer

                while current_time < (last_time + max_extrap_s):
                    current_time += dt

                    prob_reward = min(1.0, max(0.0, reward_prob_func(current_time)))
                    if random.random() < prob_reward:
                        current_event_timer = 0.0
                    else:
                        current_event_timer += dt

                    sim_obs = build_investment_sim_state(current_time, current_event_timer, context)
                    v_sim = agent.get_value(sim_obs)

                    sim_times.append(current_time)
                    sim_values.append(v_sim)

                    if v_sim <= threshold:
                        mc_leave_times.append(current_time)
                        break
                else:
                    # If it hit max_extrap_s without crossing
                    mc_leave_times.append(current_time)

                mc_label = 'MC Extrapolation' if sim == 0 else ""
                ax.plot(sim_times, sim_values, color=color, linestyle='--',
                        linewidth=1, alpha=0.2, label=mc_label)

            # Average prediction from MC rollouts
            pred_time = np.mean(mc_leave_times)

        else:
            # If it crossed in reality, find the exact crossing time
            for t_val, v_val in zip(trial['times'], trial['values']):
                if v_val <= threshold:
                    pred_time = t_val
                    break

        # ---> MODIFICATION 2: Draw Actual vs Predicted Vertical Lines <---
        ax.axvline(actual_time, color='black', alpha=0.8, linestyle='-', label=f'Actual')
        if pred_time is not None:
            ax.axvline(pred_time, color='red', alpha=0.8, linestyle='--', label=f'Predicted')

        # Formatting
        ax.set_title(f"Trial {trial_num} ({ctx_label} Ctx)", fontsize=10)

        if i >= (rows - 1) * cols:
            ax.set_xlabel("Time in Port (s)")

        if i % cols == 0:
            ax.set_ylabel("State Value V(s)")
        else:
            # sharey=True hides the ticks automatically, but this ensures labels are removed
            ax.set_yticklabels([])

            # Clean up empty subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    # Handle unified legend cleanly without duplicates
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def predict_single_trial_leave_time(trial, agent, dt=0.1, max_extrap_s=20.0):
    """
    Predicts when the agent would leave based on your intersection logic.
    """
    times = trial['times']
    values = trial['values']
    event_timers = trial['event_timer']
    threshold = trial['v_after']

    # 1. Check for intersection during the actual trial
    for i, v in enumerate(values):
        if v <= threshold:
            return times[i]

    # 2. If no intersection, extrapolate assuming no further rewards
    # We reconstruct the state using the known indices:
    # [port, time_in_port, event_timer, context, rewards_in_context, gambling_disabled]

    current_time = times[-1]
    current_event_timer = event_timers[-1]

    # We assume context and port stay constant during extrapolation
    port_id = 1.0  # As defined in your function call
    context = trial['context']
    rew_in_ctx = 4.0
    disabled = 0.0

    for _ in range(int(max_extrap_s / dt)):
        current_time += dt
        current_event_timer += dt

        # Reconstruct state vector
        sim_obs = np.array([
            port_id,
            current_time,
            current_event_timer,
            context,
            rew_in_ctx,
            disabled
        ])

        v_sim = agent.get_value(sim_obs)

        if v_sim <= threshold:
            return current_time

    return current_time  # Return capped time if they never cross


def predict_leave_time_monte_carlo(trial, agent, get_reward_prob_func, num_simulations=100, dt=0.1, max_extrap_s=30.0):
    """
    Predicts leave time using Monte Carlo rollouts for the unobserved future.

    get_reward_prob_func: A function you define that takes (time_in_port)
                          and returns the probability of reward in the next dt.
    """
    times = trial['times']
    values = trial['values']
    event_timers = trial['event_timer']
    threshold = trial['v_after']

    # 1. Did it already cross in reality? (Keep this, it's actual data!)
    for i, v in enumerate(values):
        if v <= threshold:
            return times[i]

    # 2. If it didn't cross, we spawn N parallel universes
    actual_leave_time = times[-1]
    actual_last_timer = event_timers[-1]
    context = trial['context']
    port_id = 1.0

    simulated_leave_times = []

    for sim in range(num_simulations):
        current_time = actual_leave_time
        current_event_timer = actual_last_timer

        while current_time < (actual_leave_time + max_extrap_s):
            current_time += dt

            # --- THE STOCHASTIC ENVIRONMENT ---
            # Ask your environment's logic: does a reward happen right now?
            prob_reward = get_reward_prob_func(current_time)

            if random.random() < prob_reward:
                # REWARD DELIVERED! Reset the timer.
                current_event_timer = 0.0
            else:
                # NO REWARD. Increment the timer.
                current_event_timer += dt

            # Reconstruct the state and check the brain
            sim_obs = build_investment_sim_state(current_time, current_event_timer, context)
            v_sim = agent.get_value(sim_obs)

            if v_sim <= threshold:
                simulated_leave_times.append(current_time)
                break
        else:
            # If it never crossed within max_extrap, record the max time
            simulated_leave_times.append(current_time)

    # Return the average expected leave time across all parallel realities
    return np.mean(simulated_leave_times)


def plot_prediction_results_trial_series(results, animal_id=None):
    # 1. Configuration
    trials_per_axis = 40
    num_total = len(results)
    num_axes = int(np.ceil(num_total / trials_per_axis))

    # 2. Extract arrays for easier slicing
    actuals = [r['actual'] for r in results]
    predicts = [r['predicted'] for r in results]
    contexts = [r['context'] for r in results]

    # 3. Create the figure
    fig, axes = plt.subplots(num_axes, 1, figsize=(14, 4 * num_axes), sharey=True)

    # Ensure axes is an array even if there's only one subplot
    if num_axes == 1:
        axes = [axes]

    for i in range(num_axes):
        start_idx = i * trials_per_axis
        end_idx = min((i + 1) * trials_per_axis, num_total)

        # Create x-axis indices for this block
        x_range = np.arange(start_idx, end_idx)

        # Plot Actual vs Predicted
        axes[i].plot(x_range, actuals[start_idx:end_idx], 'o-', color='black',
                     alpha=0.4, label='Actual', markersize=4, linewidth=1)

        axes[i].plot(x_range, predicts[start_idx:end_idx], 's-', color='#D62728',
                     alpha=0.8, label='Predicted', markersize=4, linewidth=1.5)

        # Optional: Visualizing Context shifts (shading the background)
        # This assumes context is 0 or 1.
        for j in x_range:
            if contexts[j] == 1:  # High Context
                axes[i].axvspan(j - 0.5, j + 0.5, color='orange', alpha=0.1)

        # Formatting
        if animal_id is not None:
            plt.suptitle(f"Animal {animal_id}")
        axes[i].set_title(f"Trials {start_idx} to {end_idx - 1}")
        axes[i].set_ylabel("Leave Time (s)")
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.3)

        if i == 0:
            axes[i].legend(loc='upper right', frameon=True)

        if i == num_axes - 1:
            axes[i].set_xlabel("Trial Index")

    plt.tight_layout()
    plt.show()


def plot_prediction_results_scatters(results, jitter_amount=0.04, animal_id=None):
    """
    Plots Predicted vs. Actual leave times with Jitter.
    Includes regression lines with R-squared and equation annotations
    rotated to match the line angle.
    """
    actuals = np.array([r['actual'] for r in results])
    predicts = np.array([r['predicted'] for r in results])
    contexts = np.array([r['context'] for r in results])

    # 1. Generate uniform noise for the jitter
    jitter_x = np.random.uniform(-jitter_amount, jitter_amount, size=len(actuals))
    jitter_y = np.random.uniform(-jitter_amount, jitter_amount, size=len(predicts))

    jittered_actuals = actuals + jitter_x
    jittered_predicts = predicts + jitter_y

    all_vals = np.concatenate([actuals, predicts])
    global_max = np.percentile(all_vals, 99) + 1.0

    my_color = sns.color_palette('Set2')
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # Made slightly larger to fit text comfortably

    # --- High Context Plot & Regression ---
    mask_high = (contexts == 1)
    if np.any(mask_high):
        x_high_orig = actuals[mask_high]
        y_high_orig = predicts[mask_high]

        # Scatter
        ax.plot(jittered_actuals[mask_high], jittered_predicts[mask_high], 'o',
                color=my_color[1], label='High', alpha=0.4, markersize=4, markeredgewidth=0)

        # Math: Regression and R-squared
        m_high, b_high = np.polyfit(x_high_orig, y_high_orig, 1)
        r2_high = np.corrcoef(x_high_orig, y_high_orig)[0, 1] ** 2

        # Plot Line
        valid_x_h = x_high_orig[x_high_orig <= global_max]
        x_stop_h = valid_x_h.max() if len(valid_x_h) > 0 else x_high_orig.max()

        x_line_h = np.array([x_high_orig.min(), x_stop_h])
        ax.plot(x_line_h, m_high * x_line_h + b_high, color=my_color[1], linestyle='-', linewidth=2.5)

        # Dynamic Text Annotation
        angle_high = np.degrees(np.arctan(m_high))
        text_x_h = x_line_h[0] + 0.75 * (x_line_h[1] - x_line_h[0])
        text_y_h = m_high * text_x_h + b_high + 0.3

        ax.text(text_x_h, text_y_h, f" y = {m_high:.2f}x {b_high:+.2f} | R² = {r2_high:.2f} ",
                color=my_color[1], fontsize=9, fontweight='bold',
                rotation=angle_high, rotation_mode='anchor',
                ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # --- Low Context Plot & Regression ---
    mask_low = (contexts == 0)
    if np.any(mask_low):
        x_low_orig = actuals[mask_low]
        y_low_orig = predicts[mask_low]

        # Scatter
        ax.plot(jittered_actuals[mask_low], jittered_predicts[mask_low], 'o',
                color=my_color[0], label='Low', alpha=0.5, markersize=4, markeredgewidth=0)

        # Math: Regression and R-squared
        m_low, b_low = np.polyfit(x_low_orig, y_low_orig, 1)
        r2_low = np.corrcoef(x_low_orig, y_low_orig)[0, 1] ** 2

        # ---> FIND LARGEST X BELOW MAX_VAL <---
        valid_x_l = x_low_orig[x_low_orig <= global_max]
        x_stop_l = valid_x_l.max() if len(valid_x_l) > 0 else x_low_orig.max()

        x_line_l = np.array([x_low_orig.min(), x_stop_l])
        ax.plot(x_line_l, m_low * x_line_l + b_low, color=my_color[0], linestyle='-', linewidth=2.5)

        # Dynamic Text Annotation
        angle_low = np.degrees(np.arctan(m_low))
        text_x_l = x_line_l[0] + 0.25 * (x_line_l[1] - x_line_l[0])
        text_y_l = m_low * text_x_l + b_low + 0.3

        ax.text(text_x_l, text_y_l, f" y = {m_low:.2f}x {b_low:+.2f} | R² = {r2_low:.2f} ",
                color=my_color[0], fontsize=9, fontweight='bold',
                rotation=angle_low, rotation_mode='anchor',
                ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # --- Axis Formatting ---
    # Plot perfect prediction baseline safely within the bounds
    ax.plot([-1, global_max + 1], [-1, global_max + 1], 'k:', alpha=0.3, label='y = x (Perfect)')

    ax.set_xlim([-0.2, global_max])
    ax.set_ylim([-0.2, global_max])
    ax.set_aspect('equal', adjustable='box')

    ax.legend(title='Context Block', loc='upper left')
    ax.set_xlabel('Actual Leave Time (s)')
    ax.set_ylabel('Predicted Leave Time (s)')
    if animal_id is not None:
        ax.set_title(f"Animal {animal_id}")

    ax.grid(True, linestyle='--', alpha=0.2)

    plt.tight_layout()
    plt.show()


# reward probability as a function of t (current time in port)
def exp_decreasing(t, cumulative=8., starting=1.):
    a = starting
    b = a / cumulative
    density = a / np.exp(b * t)
    prob = density / 10
    return prob


def evaluate_frozen_trajectories_for_animal(animal_id):
    """
    Pipeline-ready wrapper to execute the full frozen evaluation for one animal.
    """
    project_root = Path(config.MODELING_PROJECT_ROOT)
    DATA_FOLDER = project_root / Path(config.MODELING_DATA_SUBDIR)
    AGENT_FOLDER = project_root / Path(config.STEP2_PRETRAINED_AGENTS_SUBDIR)

    target_data_file = DATA_FOLDER / f"target_sessions_pooled_transitions_{animal_id}.pkl"
    pretrained_agent_file = AGENT_FOLDER / f"pretrained_agent_{animal_id}.pkl"

    if not pretrained_agent_file.exists():
        print(f"❌ Cannot evaluate: Pretrained agent for {animal_id} not found.")
        return

    # 1. Extract and Save
    # trials = extract_target_session_trajectories(target_data_file, pretrained_agent_file)
    # if trials:
    #     save_trajectory_data(trials, animal_id)

    # 2. Load Agent and Freeze it
    with open(pretrained_agent_file, 'rb') as f:
        agent = pickle.load(f)
    agent.step_size = 0.0

    # 3. Load Trials and Plot
    trials = load_trajectory_data(animal_id)

    # plot_target_session_trajectories_with_mc(
    #     trials, agent, animal_id,
    #     reward_prob_func=exp_decreasing,
    #     trial_indices=None,
    #     num_mc_traces=100
    # )

    # 4. Predict Leave Times
    results = []
    for trial in trials:
        pred_time = predict_leave_time_monte_carlo(trial, agent, exp_decreasing, num_simulations=100)
        actual_time = trial['times'][-1]

        results.append({
            'actual': actual_time,
            'predicted': pred_time,
            'context': trial['context']
        })

    plot_prediction_results_trial_series(results, animal_id=animal_id)
    plot_prediction_results_scatters(results, jitter_amount=0.04, animal_id=animal_id)


def compile_all_animal_predictions(animal_ids, num_mc_sims=100):
    """
    Loops through all animals, loads their frozen agents and trajectory data,
    runs the Monte Carlo predictions, and combines them into one master list.
    """
    project_root = Path(config.MODELING_PROJECT_ROOT)
    agent_dir = project_root / Path(config.STEP2_PRETRAINED_AGENTS_SUBDIR)
    metrics_dir = project_root / Path(config.STEP3_EVALUATION_METRICS_SUBDIR)

    master_results = []

    print(f"🌍 Starting master prediction compilation for {len(animal_ids)} animals...")

    for animal_id in animal_ids:
        print(f"\n⚙️ Processing {animal_id}...")

        # 1. Load the frozen agent
        agent_file = agent_dir / f"pretrained_agent_{animal_id}.pkl"
        if not agent_file.exists():
            print(f"   ❌ Skipping {animal_id}: Pretrained agent not found.")
            continue

        with open(agent_file, 'rb') as f:
            agent = pickle.load(f)

        agent.step_size = 0.0  # FREEZE THE AGENT!

        # 2. Load the trajectories
        traj_file = metrics_dir / f"target_session_value_trajectory_{animal_id}.json"
        if not traj_file.exists():
            print(f"   ❌ Skipping {animal_id}: Trajectory JSON not found. Did you extract them first?")
            continue

        with open(traj_file, 'r') as f:
            trials = json.load(f)

        # 3. Generate predictions
        print(f"   Running {num_mc_sims} Monte Carlo simulations for {len(trials)} trials...")
        for trial in trials:
            pred_time = predict_leave_time_monte_carlo(
                trial,
                agent,
                exp_decreasing,
                num_simulations=num_mc_sims
            )
            actual_time = trial['times'][-1]

            # Append to the master list (Added 'animal_id' for data tracking!)
            master_results.append({
                'animal_id': animal_id,
                'actual': actual_time,
                'predicted': pred_time,
                'context': trial['context']
            })

        print(f"   ✅ {animal_id} complete.")

    print(f"\n🎉 Master compilation finished! Total trials processed: {len(master_results)}")
    return master_results


def save_master_predictions(results, filename="master_leave_time_predictions.json"):
    """Caches the master list to the disk so we don't have to recalculate."""
    project_root = Path(config.MODELING_PROJECT_ROOT)
    save_file = project_root / "outputs" / "3_evaluation_metrics" / filename

    with open(save_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"💾 Master predictions saved to {save_file}")


def load_master_predictions(filename="master_leave_time_predictions.json"):
    """Loads the cached master list instantly."""
    project_root = Path(config.MODELING_PROJECT_ROOT)
    load_file = project_root / "outputs" / "3_evaluation_metrics" / filename

    if load_file.exists():
        with open(load_file, 'r') as f:
            return json.load(f)
    else:
        return None


if __name__ == "__main__":
    # Use this line when evaluating single animal
    # evaluate_frozen_trajectories_for_animal("SZ037")

    # Use this when plotting scatters for all animal results
    SZ_animals = ['SZ036', 'SZ037', 'SZ038', 'SZ039', 'SZ042', 'SZ043']
    RK_animals = ['RK007', 'RK008']
    all_animals = SZ_animals + RK_animals

    # Set this to True to ignore the saved JSON and force the MC simulations to run again
    FORCE_RECOMPILE = True

    print("=" * 50)
    print("🚀 STEP 1: Ensure all trajectories are extracted")
    print("=" * 50)
    for animal in all_animals:
        # This will load the agent, run the target sessions, and save the trajectory JSON.
        # It's fast, so it's safe to run it in a loop.
        evaluate_frozen_trajectories_for_animal(animal)

    print("\n" + "=" * 50)
    print("🧠 STEP 2: Compile Monte Carlo Predictions")
    print("=" * 50)

    if FORCE_RECOMPILE:
        print("⚠️ FORCE_RECOMPILE is True. Bypassing cache...")
        master_results = None
    else:
        master_results = load_master_predictions()

    if master_results is None:
        # Calculate it (this takes a minute)
        master_results = compile_all_animal_predictions(all_animals, num_mc_sims=100)
        # Cache it for next time!
        if master_results:
            save_master_predictions(master_results)
    else:
        print(f"⚡ Successfully loaded {len(master_results)} cached predictions from disk!")

    print("\n" + "=" * 50)
    print("📊 STEP 3: Plot Results")
    print("=" * 50)
    if master_results:
        # You can adjust the jitter amount here without waiting for MC rollouts!
        plot_prediction_results_scatters(master_results, jitter_amount=0.05)
    # # 1. Define your master list
    # SZ_animals = ['SZ036', 'SZ037', 'SZ038', 'SZ039', 'SZ042', 'SZ043']
    # RK_animals = ['RK007', 'RK008']
    # all_animals = SZ_animals + RK_animals
    #
    # # 2. Check if we already compiled the data
    # master_results = load_master_predictions()
    #
    # if master_results is None:
    #     # If it doesn't exist, calculate it (this takes a minute)
    #     master_results = compile_all_animal_predictions(all_animals, num_mc_sims=100)
    #
    #     # Cache it for next time!
    #     if master_results:
    #         save_master_predictions(master_results)
    # else:
    #     print(f"⚡ Successfully loaded {len(master_results)} cached predictions from disk!")
    #
    # # 3. Plot the final master figure!
    # if master_results:
    #     # You can adjust the jitter amount here without waiting for MC rollouts!
    #     plot_prediction_results_scatters(master_results, jitter_amount=0.05)
    print("hello")
