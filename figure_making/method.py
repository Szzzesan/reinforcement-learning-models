import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation
import seaborn as sns
import random
import math
import sys
import os
from pathlib import Path
project_root = Path(os.getcwd()).parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.append(src_path)
from src.mouse_playback_agent import MousePlaybackAgent
from src.data_loader import load_pooled_transitions
import src.config as config


def plot_nvg_illustration(transitions, agent_info, alpha=0.001, gamma=0.7, trial_index=-10):
    """
    Creates a 2-row figure for a thesis illustration.
    Row 1 is left empty for a cartoon.
    Row 2 (left) plots V_stay vs V_leave for a single converged trial, illustrating the NVG gaps.
    """
    # 1. Identify valid trials
    trial_map = []
    curr_start = -1

    for i in range(len(transitions)):
        obs, _, _, obs_next, term = transitions[i]
        in_port = (obs[0] == 1.0 and obs[5] == 0.0)

        if in_port and curr_start == -1:
            curr_start = i

        if curr_start != -1 and (obs_next[0] != 1.0 or term):
            trial_map.append((curr_start, i))
            curr_start = -1

    if not trial_map:
        print("No valid trials found.")
        return

    # Select ONE trial. By default, we pick one near the end (-10) so the agent has learned
    selected_trial = trial_map[trial_index]

    times = []
    v_stay_log = []
    v_leave_log = []
    opportunity_cost = 0

    # 2. Run Agent up to the selected trial
    info = agent_info.copy()
    info['step_size'] = alpha
    info['discount'] = gamma

    agent = MousePlaybackAgent()
    agent.agent_init(info)
    agent.agent_start(transitions[0][0])

    for i in range(len(transitions)):
        obs, _, reward, obs_next, term = transitions[i]

        # If we are inside our target trial, record the values
        if selected_trial[0] <= i <= selected_trial[1]:
            v_stay = agent.get_value(obs)

            # Construct the "leave" observation
            obs_l = obs.copy()
            obs_l[0] = 2.0
            obs_l[1] = 0.0
            v_leave = agent.get_value(obs_l) + opportunity_cost

            v_stay_log.append(v_stay)
            v_leave_log.append(v_leave)
            times.append(obs[1])

        # Step the agent
        if term:
            agent.agent_end(reward)
            if i + 1 < len(transitions):
                agent.agent_start(transitions[i + 1][0])
        else:
            agent.agent_step(reward, obs_next)

        # Stop running the agent once our target trial is over to save time
        if i == selected_trial[1]:
            break

    # 3. Setup the Thesis Figure Layout
    fig = plt.figure(figsize=(12, 8))  # 12 inches wide, 8 inches tall
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    # --- TOP ROW (Empty for Cartoon) ---
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.axis('off')  # Hide axes entirely
    # Add panel letter 'a'
    ax_top.text(
        0.0, 1.0, 'a',
        transform=(ax_top.transAxes + ScaledTranslation(-20 / 72, +7 / 72, fig.dpi_scale_trans)),
        fontsize=16,
        va='bottom',
        fontfamily='sans-serif',
        weight='bold'
    )

    # --- BOTTOM ROW LEFT (Data Plot) ---
    ax_bot_left = fig.add_subplot(gs[1, 0])

    # Add panel letter 'b'
    ax_bot_left.text(
        0.0, 1.0, 'b',
        transform=(ax_bot_left.transAxes + ScaledTranslation(-20 / 72, +7 / 72, fig.dpi_scale_trans)),
        fontsize=16,
        va='bottom',
        fontfamily='sans-serif',
        weight='bold'
    )

    # Plot the curves
    ax_bot_left.plot(times[8:] - times[8], v_stay_log[8:], color='blue', linewidth=2.5, label='Value of Staying ($V_{stay}$)')
    ax_bot_left.plot(times[8:] - times[8], v_leave_log[8:], color='orange', linestyle='--', linewidth=2.5,
                     label='Value of Leaving ($V_{leave}$)')

    # Mark the exit time
    ax_bot_left.axvline(times[-1] - times[8], color='black', alpha=0.5, linestyle=':', label='Animal Exits Port')

    # --- ADD ILLUSTRATIVE ANNOTATIONS ---
    t_start, t_end = 0, times[-1] - times[8]
    vs_start, vl_start = v_stay_log[8], v_leave_log[8]
    vs_end, vl_end = v_stay_log[-1], v_leave_log[-1]

    # Draw arrow for Initial Gap
    ax_bot_left.annotate('', xy=(t_start, vs_start), xytext=(t_start, vl_start),
                         arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax_bot_left.text(t_start + 0.2, (vs_start + vl_start) / 2, r'$\Delta V_{init}$',
                     ha='left', va='center', fontsize=16, fontweight='bold')

    # Draw arrow for Final Gap
    ax_bot_left.annotate('', xy=(t_end, vs_end), xytext=(t_end, vl_end),
                         arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    # Offset the text slightly to the left so it doesn't overlap the exit line
    ax_bot_left.text(t_end - 0.2, (vs_end + vl_end) / 2, r'$\Delta V_{end}$',
                     ha='right', va='center', fontsize=16, fontweight='bold')

    # Format Bottom Left Axis
    sns.despine(ax=ax_bot_left)
    ax_bot_left.set_xlabel("Time in Port (s)")
    ax_bot_left.set_ylabel("Value")
    ax_bot_left.set_title("Normalized Value Gap Calculation")
    ax_bot_left.legend(frameon=True, fontsize='small', loc='upper right')
    ax_bot_left.grid(False)
    ax_bot_left.set_ylim(0, 0.55)

    # --- BOTTOM ROW RIGHT (Empty) ---
    ax_bot_right = fig.add_subplot(gs[1, 1])
    ax_bot_right.axis('off')  # Leave this blank to keep the plot constrained to the left half

    plt.tight_layout()

    # --- Save Figure ---
    # (Adjust 'config' references if you aren't passing the config module directly)
    save_dir = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "fig_2-3_method_modeling.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure successfully saved to: {save_path}")

    plt.show()


def preview_nvg_trials(transitions, agent_info, alpha=0.001, gamma=0.7, num_samples=6, sample_from_last_n=100):
    """
    Randomly selects and plots a grid of converged trials to help you
    visually hunt for the perfect thesis illustration example.
    """
    # 1. Identify valid trials
    trial_map = []
    curr_start = -1

    for i in range(len(transitions)):
        obs, _, _, obs_next, term = transitions[i]
        in_port = (obs[0] == 1.0 and obs[5] == 0.0)

        if in_port and curr_start == -1:
            curr_start = i

        if curr_start != -1 and (obs_next[0] != 1.0 or term):
            trial_map.append((curr_start, i))
            curr_start = -1

    if not trial_map:
        print("No valid trials found.")
        return

    # 2. Randomly sample from the latest trials (so the agent has already learned)
    total_trials = len(trial_map)
    start_sampling_idx = max(0, total_trials - sample_from_last_n)

    # Get the actual index numbers so we can label them for you
    pool_of_indices = list(range(start_sampling_idx, total_trials))
    selected_indices = sorted(random.sample(pool_of_indices, min(num_samples, len(pool_of_indices))))

    # Dictionary to hold the data: {trial_idx: {'times': [], 'v_stay': [], 'v_leave': []}}
    plot_data = {idx: {'times': [], 'v_stay': [], 'v_leave': []} for idx in selected_indices}

    # 3. Run the Agent
    print(f"Running agent to simulate {total_trials} trials...")
    info = agent_info.copy()
    info['step_size'] = alpha
    info['discount'] = gamma

    agent = MousePlaybackAgent()  # Swap to ET_MousePlaybackAgent if you are using traces!
    agent.agent_init(info)
    agent.agent_start(transitions[0][0])

    opportunity_cost = 0

    for i in range(len(transitions)):
        obs, _, reward, obs_next, term = transitions[i]

        # Check if current step is inside ANY of our selected trials
        active_idx = None
        for idx in selected_indices:
            start_step, end_step = trial_map[idx]
            if start_step <= i <= end_step:
                active_idx = idx
                break

        # If it is, log the values
        if active_idx is not None:
            v_stay = agent.get_value(obs)

            obs_l = obs.copy()
            obs_l[0] = 2.0
            obs_l[1] = 0.0
            v_leave = agent.get_value(obs_l) + opportunity_cost

            plot_data[active_idx]['times'].append(obs[1])
            plot_data[active_idx]['v_stay'].append(v_stay)
            plot_data[active_idx]['v_leave'].append(v_leave)

        # Step agent
        if term:
            agent.agent_end(reward)
            if i + 1 < len(transitions):
                agent.agent_start(transitions[i + 1][0])
        else:
            agent.agent_step(reward, obs_next)

        # Optimization: Stop running if we've passed the last selected trial
        if i == trial_map[selected_indices[-1]][1]:
            break

    # 4. Plot the Grid
    print(f"Plotting {num_samples} candidate trials...")
    cols = 3
    rows = math.ceil(num_samples / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), constrained_layout=True)
    axes = axes.flatten() if num_samples > 1 else [axes]

    for ax_idx, trial_idx in enumerate(selected_indices):
        ax = axes[ax_idx]
        data = plot_data[trial_idx]

        ax.plot(data['times'], data['v_stay'], color='blue', linewidth=2, label='V_stay')
        ax.plot(data['times'], data['v_leave'], color='orange', linestyle='--', linewidth=2, label='V_leave')
        ax.axvline(data['times'][-1], color='black', alpha=0.5, linestyle=':')

        # PROMINENT TITLE: This is the number you need!
        ax.set_title(f"Trial Index: {trial_idx}", fontweight='bold', fontsize=12)
        ax.set_xlabel("Time in Port (s)")
        ax.set_ylabel("Value")
        # ax.grid(False)

        if ax_idx == 0:
            ax.legend(frameon=True)

    # Hide any unused subplots if num_samples isn't a perfect multiple of cols
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.show()

if __name__ == '__main__':
    PROJECT_ROOT = Path(os.getcwd()).parent
    DATA_FOLDER = PROJECT_ROOT / "data"

    animal = "SZ036"
    transitions = load_pooled_transitions(DATA_FOLDER, animal)

    scales_to_use = [
        -1,  # 0: port
        0.5,  # 1: time_in_port
        0.0,  # 2: event_timer (SCALE SET TO 0)
        -1,  # 3: context
        0.0,  # 4: rewards_in_context
        -1  # 5: gambling_disabled
    ]
    agent_info = {
        "discount": 0.95,  # Or chosen value
        "step_size": 0.025,  # Start with a smaller step size for stability over large datasets
        "num_tilings": 16,
        "iht_size": 32768,
        "gambling_max_time_s": 30.0,  # Match data_loader params if used
        "context_rewards_max": 4,  # Match data_loader params if used
        "scales": scales_to_use
    }  # agent parameters

    plot_nvg_illustration(transitions, agent_info, alpha=0.0005, gamma=0.85, trial_index=2303)
    # preview_nvg_trials(transitions, agent_info, alpha=0.0005, gamma=0.85, num_samples=15, sample_from_last_n=1000)

