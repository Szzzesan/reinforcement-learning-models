import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import src.config as config
from src.data_loader import load_pooled_transitions


# def load_pretrained_agent(animal_id, model_folder="pretrained_agents",
#                           state_config='uniform_tile_contexted_withIRI_TD-lambda'):
#     """
#     Loads a pretrained agent from a pickle file.
#
#     Args:
#         animal_id (str): The ID of the animal (e.g., 'SZ036').
#         model_folder (str): The folder name where models are saved.
#
#     Returns:
#         agent: The loaded MousePlaybackAgent object.
#     """
#     # Construct path using pathlib for cross-platform compatibility
#     # Assuming the folder is in the project root
#     file_path = os.path.join(config.MODELING_PROJECT_ROOT, model_folder, state_config,
#                              f"pretrained_agent_{animal_id}.pkl")
#
#     print(f"Loading agent from: {file_path}")
#
#     try:
#         with open(file_path, 'rb') as f:
#             agent = pickle.load(f)
#         print(f"✅ Successfully loaded agent for {animal_id}")
#         return agent
#     except FileNotFoundError:
#         print(f"❌ Error: File not found at {file_path}")
#         return None
#     except Exception as e:
#         print(f"❌ Error loading agent: {e}")
#         return None

def load_pretrained_agent(animal_id):
    """
    Loads a pretrained agent from the Phase 1 consolidated output folder.
    """
    project_root = Path(config.MODELING_PROJECT_ROOT)

    file_path = project_root / Path(config.STEP2_PRETRAINED_AGENTS_SUBDIR) / f"pretrained_agent_{animal_id}.pkl"

    print(f"Loading agent from: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            agent = pickle.load(f)
        print(f"✅ Successfully loaded agent for {animal_id}")
        return agent
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return None


def run_target_session_analysis(agent, transitions, animal_id, override_alpha=None):
    """
    Runs the pretrained agent through target sessions, logging TD errors
    and maintaining specific trial/phase logic.

    Args:
        agent: The loaded MousePlaybackAgent.
        transitions: List of (obs, action, reward, next_obs, term) tuples.
        animal_id: ID for logging purposes.

    Returns:
        pd.DataFrame: A log of every timestep including TD errors and trial info.
    """
    if override_alpha is not None:
        print(f"❄️ Overriding Pretrained Alpha: Changing from {agent.step_size} to {override_alpha}")
        agent.step_size = override_alpha
    else:
        print(f"🔥 Using Pretrained Alpha: {agent.step_size}")

    results = []

    # --- Counters & Flags (Matching your Logic) ---
    session_idx = 0
    trial_idx = 0
    context_port_phase_complete = False
    gambling_port_phase_complete = False

    # --- Initialization ---
    # Get the very first observation from the transition list
    first_obs = transitions[0][0]

    # Initialize trackers
    last_port = first_obs[0]
    last_context = first_obs[3]
    last_observation = first_obs

    # Handle the very first step logic
    if last_port == 0:
        trial_idx = 1

    # Start the agent
    agent.agent_start(first_obs)

    num_steps = len(transitions)
    print(f"🚀 Starting analysis for {animal_id} ({num_steps} steps)...")

    # --- Main Loop ---
    for i in range(num_steps):
        # Unpack the transition data
        # transitions[i] = (observation, action, reward, next_observation, terminal)
        # We treat 'observation' as 'current_observation' in your loop logic
        current_observation, _, reward, next_observation, terminal = transitions[i]

        # 1. Calculate TD Error (Before Update)
        # -------------------------------------
        v_current = agent.get_value(current_observation)

        if terminal:
            v_next = 0.0
        else:
            v_next = agent.get_value(next_observation)

        td_error = reward + agent.discount * v_next - v_current

        # 2. Extract Metadata
        # -------------------
        current_port = current_observation[0]
        current_context = current_observation[3]

        is_context_switch = (current_context != last_context)

        # 3. Trial & Phase Logic (Strictly copied)
        # ----------------------------------------

        # Event: Exit Context Port (0)
        if current_port != 0 and last_port == 0:
            last_gambling_disabled = last_observation[5]
            if last_gambling_disabled == 0:  # Enabled
                context_port_phase_complete = True
            else:
                context_port_phase_complete = False
                gambling_port_phase_complete = False

        # Event: Exit Gambling Port (1)
        if current_port != 1 and last_port == 1:
            if context_port_phase_complete:
                gambling_port_phase_complete = True
            else:
                gambling_port_phase_complete = False

        # Event: Enter Context Port (0)
        if current_port == 0 and last_port != 0:
            if gambling_port_phase_complete:
                trial_idx += 1
                context_port_phase_complete = False
                gambling_port_phase_complete = False

            # Handle session start case
            if trial_idx == 0:
                trial_idx = 1

        # 4. Store Data
        # -------------
        record = {
            'step': i,
            'session': session_idx,
            'trial': trial_idx,
            'td_error': td_error,
            'v_stay': v_current,  # Helpful for debugging
            'reward': reward,
            'port': current_port,
            'time_in_port': current_observation[1],
            'event_timer': current_observation[2],
            'context': current_context,
            'rewards_in_context': current_observation[4],
            'gambling_disabled': current_observation[5],
            'context_switch_flag': is_context_switch
        }
        results.append(record)

        # 5. Update Trackers
        # ------------------
        last_port = current_port
        last_context = current_context
        last_observation = current_observation

        # 6. Agent Update & Session Boundary
        # ----------------------------------
        if terminal:
            agent.agent_end(reward)

            # Check if there are more steps remaining
            if i + 1 < num_steps:
                session_idx += 1

                # Prepare for next session
                # The 'next_observation' from a terminal step is usually the start
                # of the *next* episode in pooled arrays.
                obs_next_session = next_observation
                agent.agent_start(obs_next_session)

                # Reset Trackers for new session
                last_port = obs_next_session[0]
                last_context = obs_next_session[3]
                last_observation = obs_next_session

                context_port_phase_complete = False
                gambling_port_phase_complete = False

                # Logic for trial_idx at start of session
                # If we start in port 2 (ITI?), wait for entry.
                # If we start in port 0, trial 1 starts.
                # (Assuming Port 2 is ITI/Travel)
                if last_port == 2:
                    trial_idx = 1  # Per your snippet logic
                else:
                    trial_idx = 0
        else:
            agent.agent_step(reward, next_observation)

    print(f"✅ Analysis complete. Processed {len(results)} steps across {session_idx + 1} sessions.")
    return pd.DataFrame(results)


def extract_investment_rewards(results_df):
    """
        Extracts reward events from the Gambling Port (Port 1) when enabled.

        Args:
            results_df (pd.DataFrame): The output from run_target_session_analysis.

        Returns:
            pd.DataFrame: A subset of rows corresponding to valid gambling rewards.
        """
    # 1. Define Filter Criteria
    # Port 1 = Gambling Port
    # Reward > 0 = A reward was actually delivered
    # gambling_disabled == 0 = The port was ENABLED (valid trial)

    mask = (
            (results_df['port'] == 1) &
            (results_df['reward'] > 0) &
            (results_df['gambling_disabled'] == 0)
    )

    # 2. Apply Filter
    rewards_df = results_df[mask].copy()

    # 3. Select & Rename Columns for Clarity
    # We want: reward, time, context, event_timer, td_error, session
    # Note: 'event_timer' wasn't explicitly in the previous record dict,
    # but if it was in your observation vector at index 2, we can recover it.
    # If your record dict only had specific keys, we stick to those.

    # Check if 'event_timer' exists (it wasn't in the previous save function explicitly).
    # If not, we can't extract it unless we re-run the analysis to include it.
    # Assuming standard columns from previous step:
    cols_to_keep = [
        'session',
        'trial',
        'time_in_port',
        'event_timer',
        'context',
        'reward',
        'td_error'
    ]

    rewards_df = rewards_df[cols_to_keep]

    print(f"✅ Extracted {len(rewards_df)} gambling rewards.")
    return rewards_df


# --- Saving ---
def save_analysis_results(df, animal_id, filename_prefix="td_errors_target_sessions"):
    """"
    Saves the analysis DataFrame to a Parquet file in the consolidated outputs directory.
    """
    project_root = Path(config.MODELING_PROJECT_ROOT)

    save_dir = project_root / Path(config.STEP3_EVALUATION_METRICS_SUBDIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{filename_prefix}_{animal_id}.parquet"
    save_path = save_dir / filename

    try:
        df.to_parquet(save_path)
        print(f"💾 Results saved successfully to: {save_path}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")


# --- Visualization functions ---
def get_trial_data(df, session_id, trial_id, dt=0.1):
    """
    Extracts all data for a specific trial and computes 'time_in_trial' (s).

    Args:
        df (pd.DataFrame): The main results_df.
        session_id (int): The session index to pull from.
        trial_id (int): The trial index to pull from.
        dt (float): The duration of a single time step in seconds.

    Returns:
        pd.DataFrame: A new DataFrame containing only the data for the
                      specified trial, plus a 'time_in_trial' column.
    """
    # Filter for the specific session and trial
    trial_df = df[(df['session'] == session_id) & (df['trial'] == trial_id)].copy()

    if trial_df.empty:
        print(f"Warning: No data found for session {session_id}, trial {trial_id}.")
        return None

    # Create the 'time_in_trial' column
    # We subtract the step number of the first row and multiply by dt
    first_step = trial_df['step'].iloc[0]
    trial_df['time_in_trial'] = (trial_df['step'] - first_step) * dt

    return trial_df


def draw_vertical_lines(ax, x_npy, ymin=0, ymax=1, color='r', alpha=1, linestyle='-', linewidth=1):
    for x_value in x_npy:
        ax.axvline(x_value, ymin=ymin, ymax=ymax, color=color, linestyle=linestyle, linewidth=linewidth)


def plot_trial_trace(trial_df, animal_id, dt=0.1):
    """
    Creates a detailed plot of a single trial's TD-error trace
    """

    if trial_df is None or trial_df.empty:
        print("Cannot plot empty DataFrame.")
        return

    fig, ax = plt.subplots(figsize=(8, 2))

    # 1. Plot the TD-Error Trace
    ax.plot(trial_df['time_in_trial'], trial_df['td_error'],
            label='Model TD-Error', color='black')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)  # Zero line

    # 2. Add shaded regions for Port Locations
    # Port IDs: 0=Context, 1=Gambling, 2=Travel
    # for port_id, color, label in [(0, 'skyblue', 'Context Port'),
    #                               (1, 'lightcoral', 'Gambling Port')]:
    #     port_times = trial_df[trial_df['port'] == port_id]['time_in_trial']
    #     if not port_times.empty:
    #         # Find continuous blocks
    #         blocks = np.split(port_times, np.where(np.diff(port_times) > (1.1 * dt))[0] + 1)
    #         for i, block in enumerate(blocks):
    #             if not block.empty:
    #                 ax.axvspan(block.iloc[0] - dt, block.iloc[-1],
    #                            color=color, alpha=0.6,
    #                            label=label if i == 0 else "_") # Only label first block

    # 3. Add vertical lines for Rewards
    reward_times = trial_df[trial_df['reward'] > 0]['time_in_trial'].to_numpy() - 0.8 * dt
    if not reward_times.size == 0:
        draw_vertical_lines(ax, reward_times, color='b', linestyle='-')

    # 4. Add vertical lines for Port Entry and Exits
    # find where port changes to 0 or 1 from 2
    port_entries = trial_df[(trial_df['port'].isin([0, 1])) &
                            (trial_df['port'].shift(1) == 2)]['time_in_trial'].to_numpy() - dt
    if not port_entries.size == 0:
        port_entries = np.concatenate(([0.0], port_entries))
        draw_vertical_lines(ax, port_entries, color='g', linestyle='--')

    port_exits = trial_df[(trial_df['port'] == 2) &
                          (trial_df['port'].shift(1).isin([0, 1]))]['time_in_trial'].to_numpy() - dt
    if not port_exits.size == 0:
        draw_vertical_lines(ax, port_exits, color='r', linestyle='--')

    # --- Formatting ---
    ax.set_xticks(np.arange(0, 15.5, 2.5))
    # ax.set_xlim(-1, 16)
    ax.set_xlabel('Time in Trial (s)')
    ax.set_ylabel('Model TD-Error (a.u.)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    session_id = trial_df['session'].iloc[0]
    trial_id = trial_df['trial'].iloc[0]
    ax.set_title(f'{animal_id}: Session {session_id}, Trial {trial_id} - TD-Error Trace')

    # # Create legend
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles, labels=labels, loc='upper right')
    plt.grid(False)
    plt.tight_layout()
    plt.show()


# --- Batch Process Animals ---
def batch_run_target_session(animal_list):
    transitions_data_path = os.path.join(config.MODELING_PROJECT_ROOT, config.MODELING_DATA_SUBDIR)
    for animal_id in animal_list:
        # load the target session transitions
        target_transitions = load_pooled_transitions(transitions_data_path, animal_id, type='target')
        # load the expert agent
        agent = load_pretrained_agent(animal_id, model_folder="pretrained_agents",
                                      state_config='circular_uniform_tiles')
        # run the expert agent through target session transitions, save their td_error
        results_df = run_target_session_analysis(agent, target_transitions, animal_id)
        td_error_vs_reward_features = extract_investment_rewards(results_df)
        save_analysis_results(results_df, animal_id, filename_prefix="result_log_target_sessions",
                              results_folder="results/circular_uniform_tiles")
        save_analysis_results(td_error_vs_reward_features, animal_id, filename_prefix="tde_reward_features",
                              results_folder="results/circular_uniform_tiles")
        # visualize random trials to validate the results
        # trials = [1, 2]
        # for trial in trials:
        #     my_trial_data = get_trial_data(results_df, session_id=1, trial_id=trial)
        #     plot_trial_trace(my_trial_data, animal_id=animal_id)


# These two functions just replace the batch_run_target_session() function
def evaluate_agent_on_target_sessions(animal_id, override_alpha=None):
    """
    Pipeline-ready function to evaluate a single animal's target sessions.
    """
    project_root = Path(config.MODELING_PROJECT_ROOT)

    data_dir = project_root / Path(config.MODELING_DATA_SUBDIR)

    # Load target sessions
    target_transitions_with_meta = load_pooled_transitions(data_dir, animal_id, type='target')
    if target_transitions_with_meta is None:
        print(f"❌ Target transitions not found for {animal_id}")
        return
    target_transitions = [t[:5] for t in target_transitions_with_meta]

    # Load pretrained expert agent
    agent = load_pretrained_agent(animal_id)
    if agent is None:
        return

    # Run Analysis
    results_df = run_target_session_analysis(agent, target_transitions, animal_id, override_alpha=override_alpha)
    td_error_vs_reward_features = extract_investment_rewards(results_df)

    # Save Outputs
    save_analysis_results(results_df, animal_id, "result_log_target_sessions")
    save_analysis_results(td_error_vs_reward_features, animal_id, "tde_reward_features")


def evaluate_all_animals(override_alpha=None):
    """Batch processes all animals found in the pretrained agents directory."""
    project_root = Path(config.MODELING_PROJECT_ROOT)

    agents_dir = project_root / Path(config.STEP2_PRETRAINED_AGENTS_SUBDIR)

    if not agents_dir.exists():
        print(f"❌ No trained agents found at {agents_dir}")
        return

    for agent_file in agents_dir.glob("pretrained_agent_*.pkl"):
        # Extract "SZ037" from "pretrained_agent_SZ037.pkl"
        animal_id = agent_file.stem.split("_")[-1]
        evaluate_agent_on_target_sessions(animal_id, override_alpha=override_alpha)

def main():
    animal_id = 'SZ036'

    # load the target session transitions
    data_path = os.path.join(config.MODELING_PROJECT_ROOT, config.MODELING_DATA_SUBDIR)
    target_transitions = load_pooled_transitions(data_path, animal_id, type='target')

    # load the expert agent
    agent = load_pretrained_agent(animal_id, state_config='circular_uniform_tiles')

    # run the expert agent through target session transitions, save their td_error
    results_df = run_target_session_analysis(agent, target_transitions, animal_id)

    # visualize random trials to validate the results
    for trial_id in [31, 32, 33]:
        my_trial_data = get_trial_data(results_df, session_id=9, trial_id=trial_id)
        plot_trial_trace(my_trial_data, animal_id=animal_id)


if __name__ == "__main__":
    evaluate_all_animals(override_alpha=0) #override_alpha can be None is we want to use the agent's pretraining alpha

    # SZ_animals = ['SZ036', 'SZ037', 'SZ038', 'SZ039', 'SZ042', 'SZ043']
    # RK_animals = ['RK007', 'RK008']
    # animal_list = SZ_animals + RK_animals

    # animal_list = ["SZ036"]
    # batch_run_target_session(animal_list)

    # main()

    # animal_id = 'SZ036'
    #
    # # load the target session transitions
    # data_path = os.path.join(config.MODELING_PROJECT_ROOT, config.MODELING_DATA_SUBDIR)
    # target_transitions = load_pooled_transitions(data_path, animal_id, type='target')
    #
    # # load the expert agent
    # agent = load_pretrained_agent(animal_id)
    #
    # # run the expert agent through target session transitions, save their td_error
    # results_df = run_target_session_analysis(agent, target_transitions, animal_id)
    # td_error_vs_reward_features = extract_investment_rewards(results_df)
    # save_analysis_results(results_df, animal_id, filename_prefix="result_log_target_sessions", results_folder="results")
    # save_analysis_results(td_error_vs_reward_features, animal_id, filename_prefix="tde_reward_features", results_folder="results")
    # # visualize random trials to validate the results
    # for trial_id in [31, 32, 33]:
    #     my_trial_data = get_trial_data(results_df, session_id=9, trial_id=trial_id)
    #     plot_trial_trace(my_trial_data)
    print('hello')
