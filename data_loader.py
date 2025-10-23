import os
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_behavior_data(animal_id, session_id=None, session_long_name=None):
    dir = os.path.join(config.MAIN_DATA_ROOT, animal_id, config.PROCESSED_DATA_SUBDIR)
    if not os.path.exists(dir):
        print(f"Warning: Directory not found at {dir}")
        return None
    if session_long_name is not None:
        target_filename = f"{animal_id}_{session_long_name}_pi_events_processed.parquet"
    elif session_id is not None:
        matching_files = [
            f for f in os.listdir(dir)
            if f.startswith(f"{animal_id}") and f.endswith("_pi_events_processed.parquet")
        ]
        matching_files.sort()
        if not 0 <= session_id < len(matching_files):
            print(f"Error: session_id '{session_id}' is out of bounds.")
            print(f"Found {len(matching_files)} sessions for '{animal_id}' with data product 'pi_events_processed'.")
            # For debugging, see what files were found:
            # print("Found files:", matching_files)
            return None

        target_filename = matching_files[session_id]

    file_path = os.path.join(dir, target_filename)
    print(f"Loading: {file_path}")  # Helpful for confirming the right file is being loaded
    print(f"session: {target_filename[:22]}")

    return pd.read_parquet(file_path)

def load_pretraining_data(animal_id, session_id=None, session_long_name=None):
    dir = os.path.join(config.MAIN_DATA_ROOT, animal_id, config.PRETRAINING_PROCESSED_DATA_SUBDIR)
    if not os.path.exists(dir):
        print(f"Warning: Directory not found at {dir}")
        return None
    if session_long_name is not None:
        target_filename = f"{animal_id}_{session_long_name}_pi_events_processed.parquet"
    elif session_id is not None:
        matching_files = [
            f for f in os.listdir(dir)
            if f.startswith(f"{animal_id}") and f.endswith("_pi_events_processed.parquet")
        ]
        matching_files.sort()
        if not 0 <= session_id < len(matching_files):
            print(f"Error: session_id '{session_id}' is out of bounds.")
            print(f"Found {len(matching_files)} sessions for '{animal_id}' with data product 'pi_events_processed'.")
            # For debugging, see what files were found:
            # print("Found files:", matching_files)
            return None

        target_filename = matching_files[session_id]

    file_path = os.path.join(dir, target_filename)
    print(f"Loading: {file_path}")  # Helpful for confirming the right file is being loaded
    print(f"session: {target_filename[:22]}")

    return pd.read_parquet(file_path)


def convert_behavior_data_to_state_transitions(df, env_params):
    """
    This function parses raw behavioral data file and return a
    list of transition tuples for one complete session.

    Returns:
        list: [(obs_t, action_t, reward_t+1, next_obs_t+1, terminal_flag), ...]
    *** obs_t is a 6-element array:
    [
        current_port,
        time_in_port, <-- time elapsed from port entry
        event_timer, <-- time elapsed from last significant event (reward or entry/exit)
        current_context, <-- low or high block
        rewards_in_context, <-- how many rewards agent has collected from context
        gambling_disabled_feature <-- no reward from gambling port if the agent exited the context port prematurely
    ] ***
    """
    # --- Extract necessary parameters ---
    dt = env_params.get("time_step_duration", 0.1)
    session_duration = env_params.get("session_duration_min", 18) * 60
    context_rewards_max = env_params.get("context_rewards_max", 4)
    block_duration = env_params.get("block_duration_min", 3) * 60
    port_map = {2: 0, 1: 1}  # Data: Context=2, Gambling=1 -> Env: Context=0, Gambling=1

    # --- Preprocess and Prepare Time Series ---
    df = df[(df['key'] == "trial") | (df['key'] == "head") | ((df['key'] == "reward") & (df['value'] == 1))]
    df = df.sort_values(by='task_time').reset_index(drop=True)
    df['time_delta'] = pd.to_timedelta(df['task_time'], unit='s')
    df = df.set_index('time_delta')

    start_time = df.index.min()
    end_time = pd.Timedelta(seconds=session_duration)
    # Ensure index covers full duration, even if data ends early
    if df.index.max() < end_time:
        end_time = df.index.max() + pd.Timedelta(seconds=dt)  # Extend slightly beyond last event

    full_time_index = pd.timedelta_range(start=start_time, end=end_time, freq=f"{int(dt * 1000)}ms",
                                         closed='left')  # Use closed='left'

    event_df = df.copy()  # Keep original events

    # Resample and forward-fill state information
    state_cols = ['phase', 'port', 'time_in_port', 'trial']
    df_resampled = df[state_cols].reindex(full_time_index, method='ffill')  # Use method='ffill' directly
    df_resampled['task_time'] = df_resampled.index.total_seconds()

    # --- Initialize State Tracking Variables ---
    transitions = []
    rewards_in_context = 0
    gambling_disabled = True
    last_event_time_in_port = 0.0
    is_traveling = True
    travel_timer = 0.0
    current_port_id = -1  # Start with invalid port
    last_port_event_type = None # 'entry' or 'exit'

    print("Processing time steps...")
    num_steps = len(df_resampled)
    for i in range(num_steps):  # Iterate up to the second-to-last possible step start
        t = df_resampled.index[i].total_seconds()
        t_next = t + dt
        # 1. Determine obs_t based on state carried over from the end of step i-1
        if i == 0:
            # --- Initial State Logic ---
            # session starts exactly with context port entry
            port_id_t = 0  # Context Port
            time_in_port_t = 0.0
            event_timer_t = 0.0
            row_t = df_resampled.iloc[i]  # Get initial phase/context
            current_context = 0 if row_t['phase'] == '0.4' else 1
            rewards_in_context = 0
            gambling_disabled = False
            is_traveling_at_start_of_t = False  # Start in a port
            last_event_time_in_port = 0.0
            travel_timer = 0.0
            last_port_event_type = 'entry'  # Assume session start is an entry
            # --- End Initial State Logic ---
        else:
            # --- Get state from the end of the previous transition ---
            prev_obs_t_plus_1 = transitions[-1][3]
            port_id_t = int(prev_obs_t_plus_1[0])
            time_in_port_t = prev_obs_t_plus_1[1]
            event_timer_t = prev_obs_t_plus_1[2]
            current_context = int(prev_obs_t_plus_1[3])
            rewards_in_context = int(prev_obs_t_plus_1[4])
            gambling_disabled = bool(prev_obs_t_plus_1[5])
            # --- End Get state ---

            # Need to know if we *were* traveling at the start of this step
            is_traveling_at_start_of_t = (port_id_t == 2)
            # last_event_time_in_port, travel_timer, last_port_event_type are persistent loop variables

        # --- Assemble obs_t ---
        obs_t = np.array([
            float(port_id_t),
            round(time_in_port_t, 3),
            round(event_timer_t, 3),
            float(current_context),
            float(rewards_in_context),
            float(gambling_disabled)
        ])
        # --- End Assemble obs_t ---

        # 2. Examine Interval [t, t+dt) for events
        events_in_interval = event_df[(event_df['task_time'] >= t) & (event_df['task_time'] < t_next)]
        entry_event = events_in_interval[
            ((events_in_interval['key'] == 'head') | (events_in_interval['key'] == 'trial')) & (
                        events_in_interval['value'] == 1.0)
            ]
        exit_event = events_in_interval[
            (events_in_interval['key'] == 'head') & (events_in_interval['value'] == 0.0)
            ]
        reward_event = events_in_interval[
            (events_in_interval['key'] == 'reward') & (events_in_interval['value'] == 1.0)
        ]
        trial_start_event = events_in_interval[
            (events_in_interval['key'] == 'trial') & (events_in_interval['value'] == 1.0)
        ]
        # Filter potential double entry
        entry_event_valid = False
        if not entry_event.empty:
            if last_port_event_type != 'entry':  # Only process entry if last event wasn't also an entry
                entry_event_valid = True
                entry_port_data = port_map.get(entry_event['port'].iloc[0], -1)
                last_port_event_type = 'entry'
            # else: print(f"Ignoring double entry at t={t:.2f}") # Optional: for debugging

        exit_event_valid = False
        if not exit_event.empty:
            exit_event_valid = True
            leaving_port_data = port_map.get(exit_event['port'].iloc[0], -1)
            last_port_event_type = 'exit'
        # --------

        # 3. Determine Action action_t
        action_t = 0  # Default action
        if is_traveling_at_start_of_t and entry_event_valid:
            action_t = 1  # 'Enter'
            entry_port_data = port_map.get(entry_event['port'].iloc[0], -1)  # Port entered
            last_port_event_type = 'entry'
        elif not is_traveling_at_start_of_t and exit_event_valid:
            action_t = 1  # 'Exit'
            leaving_port_data = port_map.get(exit_event['port'].iloc[0], -1)  # Port left
            last_port_event_type = 'exit'
        # If no valid entry/exit, action remains 0 ('stay' or 'continue traveling')
        # --------

        # 4. Determine Reward reward_t_plus_1
        reward_t_plus_1 = 1.0 if not reward_event.empty else 0.0
        # --------

        # 5. Calculate State Variables for obs_t_plus_1
        # --- Initialize next state variables based on current state obs_t ---
        next_port_id = port_id_t
        next_time_in_port = time_in_port_t
        next_event_timer = event_timer_t
        next_context = current_context
        next_rewards_in_context = rewards_in_context
        next_gambling_disabled = gambling_disabled
        next_is_traveling = is_traveling_at_start_of_t  # Will be updated based on action
        current_travel_timer = travel_timer  # Use local copy for updates

        # --- Apply effects of Trial Start event occurring in [t, t+dt) ---
        if not trial_start_event.empty:
            next_rewards_in_context = 0
            next_gambling_disabled = False
            last_event_time_in_port = t  # Trial start resets the timer base

        # --- Apply effects of Action action_t occurring in [t, t+dt) ---
        if action_t == 1:  # If an entry or exit happened
            if is_traveling_at_start_of_t:  # Must be an ENTRY
                next_port_id = entry_port_data
                next_time_in_port = 0.0
                next_event_timer = 0.0
                last_event_time_in_port = t  # Entry is a significant event
                next_is_traveling = False
                current_travel_timer = 0.0
            else:  # Must be an EXIT
                next_port_id = 2  # Start traveling
                next_time_in_port = 0.0
                next_event_timer = 0.0
                last_event_time_in_port = t  # Exit is a significant event
                next_is_traveling = True
                current_travel_timer = 0.0
                # Update gambling disabled if exiting context port early
                if port_id_t == 0 and rewards_in_context < context_rewards_max:
                    next_gambling_disabled = True
        else:  # If action_t == 0 (stay or continue traveling)
            if is_traveling_at_start_of_t:  # Continue traveling
                current_travel_timer += dt
                next_time_in_port = current_travel_timer
                next_event_timer = current_travel_timer
                # next_is_traveling remains True
            else:  # Stay in port
                next_time_in_port = time_in_port_t + dt
                next_event_timer = event_timer_t + dt
                # next_is_traveling remains False

        # --- Apply effects of Reward reward_t_plus_1 occurring in [t, t+dt) ---
        if reward_t_plus_1 > 0:
            # Only process reward effects if the agent wasn't traveling when it happened
            if not is_traveling_at_start_of_t:
                last_event_time_in_port = t  # Time the reward interval started
                next_event_timer = 0.0  # Reset event timer for the state *after* reward
                if port_id_t == 0:  # If reward was in context port
                    next_rewards_in_context += 1
                    if next_rewards_in_context >= context_rewards_max:
                        next_gambling_disabled = False  # Enable gambling

        # --- Determine Context for Next State (at time t+dt) ---
        if i + 1 < num_steps:
            row_t_next = df_resampled.iloc[i + 1]
            next_context = 0 if row_t_next['phase'] == '0.4' else 1
        else:
            next_context = current_context  # Persist if last step
        # (Add block switch logic here if needed, applied to next_context)
        # --------

        # 6. Assemble obs_t_plus_1
        obs_t_plus_1 = np.array([
            float(next_port_id),
            round(next_time_in_port, 3),
            round(next_event_timer, 3),
            float(next_context),
            float(next_rewards_in_context),
            float(next_gambling_disabled)
        ])
        # --------

        # --- Determine Terminal Flag ---
        terminal_flag = (i == num_steps - 1) or (t_next >= session_duration)
        # --------

        # 7. Append Transition
        transitions.append((obs_t, action_t, reward_t_plus_1, obs_t_plus_1, terminal_flag))
        # --------

        # 8. Update persistent loop variables for NEXT iteration's step 1
        travel_timer = current_travel_timer  # Carry over updated travel timer
        # --------

    print(f"Processed {len(transitions)} time steps.")
    return transitions


if __name__ == '__main__':
    pi_events = load_behavior_data("SZ036", session_id=5)
    env_params = {
        "time_step_duration": 0.1,
        "travel_time": 0.4,
        "session_duration_min": 18,
        "context_rewards_max": 4,
        "block_duration_min": 3
    }
    transitions = convert_behavior_data_to_state_transitions(pi_events, env_params)
    print("hello")
