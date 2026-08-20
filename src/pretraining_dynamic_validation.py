import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.mouse_playback_agent import MousePlaybackAgent
import config


def validate_learning_trajectory(animal_id, best_params, data_folder="data"):
    # 1. Setup & Load
    project_root = Path.cwd().parent
    data_path = project_root / data_folder / f"pooled_transitions_{animal_id}.pkl"

    with open(data_path, 'rb') as f:
        transitions = pickle.load(f)

    # 2. Initialize NAIVE Agent
    # We use the Context Scale -1 config as discussed
    optimal_scales = [-1, 1 / 2.0, 0.0, -1, 0.0, -1]

    info = {
        "discount": best_params['gamma'],
        "step_size": best_params['alpha'],
        "num_tilings": 16,
        "iht_size": 32768,
        "gambling_max_time_s": 30.0,
        "context_rewards_max": 4,
        "scales": optimal_scales
    }

    agent = MousePlaybackAgent()
    agent.agent_init(info)

    # Validation Parameters
    opportunity_cost = 0.1
    max_lookahead = 40
    simulation_dt = 0.1

    results = []

    print(f"📈 Validating Learning Trajectory for {animal_id}...")

    i = 0
    trial_count = 0
    session_counter = 0  # <--- NEW: Track Session Index

    agent.agent_start(transitions[0][0])

    with tqdm(total=len(transitions), unit="step") as pbar:
        while i < len(transitions):
            obs, _, reward, obs_next, term = transitions[i]

            # Context is usually at index 3 in your feature vector
            current_context = obs[3]

            # CHECK: Start of Investment Trial? (Port 1, Enabled, Time 0)
            in_port_1 = (obs[0] == 1.0)
            is_enabled = (obs[5] == 0.0)

            if in_port_1 and is_enabled and obs[1] == 0.0:
                # --- A. PAUSE & PREDICT ---
                pred_leave_time = max_lookahead
                sim_obs = obs.copy()

                for t in np.arange(0, max_lookahead, simulation_dt):
                    sim_obs[1] = t
                    v_stay = agent.get_value(sim_obs)

                    obs_leave = sim_obs.copy();
                    obs_leave[0] = 2.0;
                    obs_leave[1] = 0.0
                    v_leave = agent.get_value(obs_leave) + opportunity_cost

                    if v_leave > v_stay:
                        pred_leave_time = t
                        break

                # --- B. FIND ACTUAL OUTCOME ---
                j = i
                actual_leave_time = None
                while j < len(transitions):
                    curr_obs, _, _, next_obs, curr_term = transitions[j]
                    if next_obs[0] != 1.0 or curr_term:
                        actual_leave_time = curr_obs[1]
                        break
                    j += 1

                if actual_leave_time is not None:
                    results.append({
                        "trial_idx": trial_count,
                        "session_idx": session_counter,  # <--- Saved Here
                        "context": current_context,  # <--- Saved Here
                        "actual": actual_leave_time,
                        "predicted": pred_leave_time
                    })
                    trial_count += 1

            # --- C. RESUME & UPDATE SESSION ---
            if term:
                agent.agent_end(reward)
                session_counter += 1  # <--- Increment Session Index on Terminal
                if i + 1 < len(transitions):
                    agent.agent_start(transitions[i + 1][0])
            else:
                agent.agent_step(reward, obs_next)

            i += 1
            pbar.update(1)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # # 4. Visualization
    # # Rolling Average to smooth out the noise and see the "Trend"
    # window = 10
    # df['actual_smooth'] = df['actual'].rolling(window=window).mean()
    # df['pred_smooth'] = df['predicted'].rolling(window=window).mean()
    #
    # plt.figure(figsize=(14, 6))
    #
    # # Scatter of raw trials
    # plt.scatter(df['trial_idx'], df['actual'], alpha=0.2, color='gray', s=10, label='Actual (Raw)')
    # plt.scatter(df['trial_idx'], df['predicted'], alpha=0.4, color='orange', s=10, label='Predicted (Raw)')
    #
    # # Trend Lines (The Learning Curve)
    # plt.plot(df['trial_idx'], df['actual_smooth'], color='black', linewidth=2, label=f'Actual ({window}-Trial Avg)')
    # plt.plot(df['trial_idx'], df['pred_smooth'], color='red', linewidth=2, label=f'Predicted ({window}-Trial Avg)')
    #
    # plt.xlabel("Trial Number")
    # plt.ylabel("Leave Time (s)")
    # plt.ylim(-1, 20)
    # plt.xlim(2600, 3100)
    # plt.title(f"Learning Trajectory: Actual vs. Predicted ({animal_id})")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()

    # fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    # low_df = df[df['context']==0]
    # high_df = df[df['context']==1]
    # for i, df in enumerate([low_df, high_df]):
    #     # Scatter of raw trials
    #     # ax[i].scatter(df['trial_idx'], df['actual'], alpha=0.2, color='gray', s=10, label='Actual (Raw)')
    #     # ax[i].scatter(df['trial_idx'], df['predicted'], alpha=0.4, color='orange', s=10, label='Predicted (Raw)')
    #
    #     # Trend Lines (The Learning Curve)
    #     ax[i].plot(df['trial_idx'], df['actual'], color='black', linewidth=2, label=f'Actual ({window}-Trial Avg)')
    #     ax[i].plot(df['trial_idx'], df['predicted'], color='red', linewidth=2, label=f'Predicted ({window}-Trial Avg)')
    #     ax[i].set_ylabel("Leave Time (s)")
    #     ax[i].set_ylim(-1, 35)
    #
    # ax[1].set_xlabel("Trial Number")
    # ax[0].set_title(f"Low Context")
    # ax[1].set_title(f"High Context")
    # ax[0].legend()
    # fig.suptitle(f"Learning Trajectory: Actual vs. Predicted ({animal_id})")
    # plt.show()

    return df


def plot_behavior_evolution(trajectory_df, bin_size=5):
    """
    Plots the evolution of animal vs. agent behavior over sessions.

    Args:
        trajectory_df (pd.DataFrame): Must contain columns:
            - 'actual': Actual leave time
            - 'predicted': Agent predicted leave time
            - 'context': 0 (Low) or 1 (High)
            - 'session_idx': Session number (integer)
        bin_size (int): Number of sessions to group together (default 5).
    """
    # 1. PREPARE DATA
    # Create a 'Session Block' column (e.g., Sessions 0-4 -> Block 0)
    df = trajectory_df.copy()
    df['session_bin'] = (df['session_idx'] // bin_size).astype(int)

    # Map context to string labels for clearer legends
    context_map = {0: 'Low', 1: 'High'}
    if 'context_label' not in df.columns:
        df['context_label'] = df['context'].map(context_map)

    # Set Colors
    colors = sns.color_palette('Set2')
    low_color = colors[0]
    high_color = colors[1]
    custom_palette = {'Low': low_color, 'High': high_color}
    hue_order = ['Low', 'High']

    # 2. SETUP PLOT
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)  # Minimize gap between rows

    # --- ROW 1: ACTUAL BEHAVIOR (Split Violins) ---
    ax_top = axes[0]

    sns.violinplot(
        data=df,
        x='session_bin',
        y='actual',
        hue='context_label',
        hue_order=hue_order,
        split=True,  # The "Split" look you asked for
        inner='quartile',  # Show quartiles inside
        palette=custom_palette,
        ax=ax_top,
        linewidth=0.5,
        alpha=0.5,
        cut=0  # Don't extend violins past data range
    )

    # Customizing the Median Lines
    # Iterate over the lines drawn by violinplot.
    # They are drawn in groups of 3 per violin: [25th, Median, 75th]
    for i, line in enumerate(ax_top.lines):
        # Every 2nd line in a group of 3 is the median (index 1, 4, 7...)
        if (i + 1) % 3 == 2:  # Indices 1, 4, 7 correspond to the Medians
            line.set_linewidth(1.5)  # Make it thicker
            line.set_linestyle('--')  # Make it solid (optional, looks more like a 'bar')
            line.set_color('black')  # Optional: Make it pop against the color block
        else:
            # Optional: Style the quartiles (25th/75th) to be subtle
            line.set_linewidth(0.8)
            line.set_linestyle(':')
            line.set_color('black')
            line.set_alpha(0.6)

    ax_top.set_ylabel("Actual Leave Time (s)")
    # ax_top.set_title(f"Evolution of Actual Animal Behavior (Binned by {bin_size} Sessions)", fontsize=14)
    ax_top.legend(loc='upper left', title=None, frameon=True)
    ax_top.grid(axis='y', linestyle='--', alpha=0.3)

    # --- ROW 2: PREDICTED BEHAVIOR (Box + Swarm) ---
    ax_bot = axes[1]

    # Boxplot with your exact settings
    sns.boxplot(
        data=df,
        x='session_bin',
        y='predicted',
        hue='context_label',
        notch=True,
        gap=0.1,
        hue_order=hue_order,
        palette=custom_palette,
        boxprops=dict(alpha=0.4),  # Translucent box
        medianprops={'linewidth': 1, 'color': 'black'},
        legend=False,
        showfliers=False,
        ax=ax_bot
    )

    # Swarmplot with your exact settings
    sns.swarmplot(
        data=df,
        x='session_bin',
        y='predicted',
        hue='context_label',
        hue_order=hue_order,
        size=1,
        palette=custom_palette,
        dodge=True,
        legend=False,
        ax=ax_bot,
        linewidth=0.5,
        edgecolor='face'
    )

    # Apply alpha to collections (Your requested loop)
    fill_alpha = 0.5
    for collection in ax_bot.collections:
        # We only apply this to PathCollections (which are the swarm dots)
        if isinstance(collection, matplotlib.collections.PathCollection):
            face_colors = collection.get_facecolors()
            if len(face_colors) > 0:
                face_colors[:, 3] = fill_alpha
                collection.set_facecolors(face_colors)

    ax_bot.set_ylabel("Predicted Leave Time (s)")
    ax_bot.set_xlabel("Five-session group")
    ax_bot.grid(axis='y', linestyle='--', alpha=0.3)

    # --- FORMATTING AXES ---
    # Match Y-Limits
    y_max = max(df['actual'].max(), df['predicted'].max()) * 1.1
    ax_top.set_ylim(0, 40)
    ax_bot.set_ylim(0, 15)

    # X-Axis Labels
    unique_bins = sorted(df['session_bin'].unique())
    ax_bot.set_xticks(range(len(unique_bins)))

    sns.despine(fig=fig, offset=10, trim=True)
    plt.show()


def load_pretrained_agent(animal_id, model_folder="pretrained_agents",
                          state_config='uniform_tile_contexted_withIRI_TD-lambda'):
    """
    Loads a pretrained agent from a pickle file.

    Args:
        animal_id (str): The ID of the animal (e.g., 'SZ036').
        model_folder (str): The folder name where models are saved.

    Returns:
        agent: The loaded MousePlaybackAgent object.
    """
    # Construct path using pathlib for cross-platform compatibility
    # Assuming the folder is in the project root
    file_path = os.path.join(config.MODELING_PROJECT_ROOT, model_folder, state_config,
                             f"pretrained_agent_{animal_id}.pkl")

    print(f"Loading agent from: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            agent = pickle.load(f)
        print(f"✅ Successfully loaded agent for {animal_id}")
        return agent
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading agent: {e}")
        return None


def plot_value_functions(animal_id, max_time_s=20.0, dt=0.1):
    """
    Plots the learned Value Function V(s) for the Gambling Port in Low vs High context.

    Args:
        animal_id (str): ID of the animal (e.g., 'SZ036').
        max_time_s (float): How far into the trial to simulate (x-axis limit).
        dt (float): Time step for the simulation curve.
    """
    # 1. Load Agent
    agent = load_pretrained_agent(animal_id)
    if agent is None:
        print(f"❌ Could not load agent for {animal_id}")
        return

    # 2. Setup Simulation Arrays
    time_steps = np.arange(0, max_time_s, dt)
    v_low = []
    v_high = []

    # Standard State Vector Structure:
    # [Port, Time, Event_Timer, Context, Rewards_in_Context, Disabled]
    # We assume: Port=1 (Gambling), Disabled=0 (Enabled)

    # 3. Compute Values for LOW Context (Context = 0)
    for t in time_steps:
        # Construct state vector
        # Note: The scale for Event_Timer (idx 2) is always 0, so it doesn't matter what we pass to it
        obs = np.array([1.0, t, 0, 0.0, 0.0, 0.0])
        v_low.append(agent.get_value(obs))

    # 4. Compute Values for HIGH Context (Context = 1)
    for t in time_steps:
        obs = np.array([1.0, t, 0, 1.0, 0.0, 0.0])
        v_high.append(agent.get_value(obs))

    # 5. Plotting
    palette = sns.color_palette("Set2", 2)
    plt.figure(figsize=(8, 5))

    plt.plot(time_steps, v_low, label='Low Context (0)', color=palette[0], linewidth=2.5)
    plt.plot(time_steps, v_high, label='High Context (1)', color=palette[1], linewidth=2.5)

    # Add "Opportunity Cost" line (V_leave)
    # This helps you see where the agent *would* decide to leave
    opportunity_cost = 0.0  # Adjust if your agent uses a different cost

    # To calculate V_leave, we check Port 2 at t=0
    obs_leave_low = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    obs_leave_high = np.array([2.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    v_leave_val_low = agent.get_value(obs_leave_low) + opportunity_cost
    v_leave_val_high = agent.get_value(obs_leave_high) + opportunity_cost
    plt.axhline(y=v_leave_val_low, color=palette[0], linestyle=':', alpha=0.8, label=f'Leave Threshold Low (V_travel)')
    plt.axhline(y=v_leave_val_high, color=palette[1], linestyle=':', alpha=0.8,
                label=f'Leave Threshold High (V_travel)')

    plt.title(f"Learned Value Functions: {animal_id}", fontsize=14)
    plt.xlabel("Time in Gambling Port (s)", fontsize=12)
    plt.ylabel("Estimated Value V(s)", fontsize=12)
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)

    # Formatting
    plt.tight_layout()
    plt.show()


## --- Examine Values after Integrating Both Event_timer and Time_in_port ---
def plot_value_vs_time_in_port(agent, event_timer_values=[1.0, 1.8, 3.0],
                               port=1, rewards_in_context=4, gambling_disabled=0,
                               max_time=10.0, step_size=0.1):
    """
    Plots V(s) vs. time_in_port, splitting by Context 0 (Low) and Context 1 (High).
    Different event_timer values are represented by shades of the context color.
    """
    time_in_port_sweep = np.arange(0, max_time + step_size, step_size)

    # Extract base colors from Set2
    set2 = sns.color_palette('Set2')
    color_low = set2[0]  # Context 0
    color_high = set2[1]  # Context 1

    # Generate shades (reverse=True makes it go from darkest/pure color to lightest)
    # We add +2 to n_colors to prevent the final shade from being purely white/invisible
    shades_low = sns.light_palette(color_low, n_colors=len(event_timer_values) + 2, reverse=True)
    shades_high = sns.light_palette(color_high, n_colors=len(event_timer_values) + 2, reverse=True)

    plt.figure(figsize=(10, 6))

    # 1. Sweep for Context 0 (Low Reward Rate)
    for i, et in enumerate(event_timer_values):
        values = []
        for tip in time_in_port_sweep:
            # Context = 0
            state = np.array([port, tip, et, 0, rewards_in_context, gambling_disabled])
            values.append(agent.get_value(state))
        plt.plot(time_in_port_sweep, values, color=shades_low[i], linewidth=2.5,
                 label=f'Ctx 0 (Low), event_timer = {et}s')

    # 2. Sweep for Context 1 (High Reward Rate)
    for i, et in enumerate(event_timer_values):
        values = []
        for tip in time_in_port_sweep:
            # Context = 1
            state = np.array([port, tip, et, 1, rewards_in_context, gambling_disabled])
            values.append(agent.get_value(state))
        plt.plot(time_in_port_sweep, values, color=shades_high[i], linewidth=2.5,
                 label=f'Ctx 1 (High), event_timer = {et}s')

    plt.title("Learned Value vs. Time in Port", fontsize=14)
    plt.xlabel("time_in_port (seconds)", fontsize=12)
    plt.ylabel("State Value V(s)", fontsize=12)

    # Place legend outside the plot so it doesn't cover the curves
    plt.legend(title="Context & Held Variables:", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_value_vs_event_timer(agent, time_in_port_values=[1.0, 2.0, 4.0, 8.0],
                              port=1, rewards_in_context=4, gambling_disabled=0,
                              max_time=3.0, step_size=0.05):
    """
    Plots V(s) vs. event_timer, splitting by Context 0 (Low) and Context 1 (High).
    Different time_in_port values are represented by shades of the context color.
    """
    event_timer_sweep = np.arange(0, max_time + step_size, step_size)

    # Extract base colors from Set2
    set2 = sns.color_palette('Set2')
    color_low = set2[0]
    color_high = set2[1]

    # Generate shades
    shades_low = sns.light_palette(color_low, n_colors=len(time_in_port_values) + 2, reverse=True)
    shades_high = sns.light_palette(color_high, n_colors=len(time_in_port_values) + 2, reverse=True)

    plt.figure(figsize=(10, 6))

    # 1. Sweep for Context 0 (Low Reward Rate)
    for i, tip in enumerate(time_in_port_values):
        values = []
        for et in event_timer_sweep:
            # Context = 0
            state = np.array([port, tip, et, 0, rewards_in_context, gambling_disabled])
            values.append(agent.get_value(state))
        plt.plot(event_timer_sweep, values, color=shades_low[i], linewidth=2.5,
                 label=f'Ctx 0 (Low), time_in_port = {tip}s')

    # 2. Sweep for Context 1 (High Reward Rate)
    for i, tip in enumerate(time_in_port_values):
        values = []
        for et in event_timer_sweep:
            # Context = 1
            state = np.array([port, tip, et, 1, rewards_in_context, gambling_disabled])
            values.append(agent.get_value(state))
        plt.plot(event_timer_sweep, values, color=shades_high[i], linewidth=2.5,
                 label=f'Ctx 1 (High), time_in_port = {tip}s')

    plt.title("Learned Value vs. Time Since Last Event (Anticipation)", fontsize=14)
    plt.xlabel("event_timer (seconds)", fontsize=12)
    plt.ylabel("State Value V(s)", fontsize=12)

    # Place legend outside the plot
    plt.legend(title="Context & Held Variables:", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_value_vs_travel_time(agent, max_travel_time=1.0, step_size=0.1):
    """
    Plots V(s) vs. travel time (port = 2).
    In the data loader, during travel, both time_in_port and event_timer
    increment together, and gambling_disabled is set to 1.
    """
    travel_times = np.arange(0, max_travel_time + step_size, step_size)

    values_low = []
    values_high = []

    for t in travel_times:
        # State: [port, time_in_port, event_timer, context, rewards, gambling_disabled]
        # Port is 2 (traveling). Gambling is 1 (disabled after leaving gambling port).
        state_low = np.array([2.0, t, t, 0.0, 4.0, 1.0])
        state_high = np.array([2.0, t, t, 1.0, 4.0, 1.0])

        values_low.append(agent.get_value(state_low))
        values_high.append(agent.get_value(state_high))

    # Plotting
    set2 = sns.color_palette('Set2')
    plt.figure(figsize=(8, 5))

    plt.plot(travel_times, values_low, label='Context 0 (Low)', color=set2[0], linewidth=2.5)
    plt.plot(travel_times, values_high, label='Context 1 (High)', color=set2[1], linewidth=2.5)

    plt.title("Learned Value During Travel (Gambling -> Context Port)", fontsize=14)
    plt.xlabel("Travel Time (seconds)", fontsize=12)
    plt.ylabel("State Value V(s)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    agent = load_pretrained_agent('SZ036', model_folder="pretrained_agents",
                                  state_config='uniform_tile_contexted_withIRI_TD-lambda')
    plot_value_vs_travel_time(agent)
    plot_value_vs_time_in_port(agent)
    plot_value_vs_event_timer(agent)

    # plot_value_functions("SZ043")

    # # --- EXECUTE ---
    # # --- 1. SET UP PATHS ---
    # project_root = Path(os.getcwd()).parent
    # params_path = project_root / "model_fitting_midsteps" / "uniform_tile_uncontexted" / "best_params_round1.pkl"
    #
    # # --- 2. LOAD BEST PARAMETERS ---
    # print(f"Reading parameters from {params_path}...")
    # try:
    #     with open(params_path, 'rb') as f:
    #         # Assuming the pickle contains the dictionary {animal_id: pd.Series}
    #         best_params_loaded = pickle.load(f)
    #     print(f"Successfully loaded parameters for {len(best_params_loaded)} animals.")
    # except FileNotFoundError:
    #     print(f"❌ Error: Could not find the file at {params_path}")
    #     best_params_loaded = None
    # except Exception as e:
    #     print(f"❌ An error occurred while reading the pickle: {e}")
    #     best_params_loaded = None
    #
    # # Need to fetch the specific params for this animal first
    # params = best_params_loaded["SZ036"]
    # df_traj = validate_learning_trajectory("SZ036", params)
    # plot_behavior_evolution(df_traj, bin_size=5)

    print('hello')
