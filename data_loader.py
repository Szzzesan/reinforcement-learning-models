import os
import config
import numpy as np
import pandas as pd


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
    gambling_disabled = False
    last_reward_time_in_port = -np.inf
    is_traveling = False
    travel_timer = 0.0
    current_port_id = -1  # Start with invalid port

    print("Processing time steps...")
    num_steps = len(df_resampled)
    for i in range(num_steps):  # Iterate up to the second-to-last possible step start
        t = df_resampled.index[i].total_seconds()
        t_next = t + dt

        # Get state info for the *start* of the current dt interval
        row_t = df_resampled.iloc[i]

        # --- Check for events *within* the interval [t, t_next) ---
        events_in_interval = event_df[
            (event_df['task_time'] >= t) & (event_df['task_time'] < t_next)
            ]

        # --- Determine State at time t ---
        current_context = 0 if row_t['phase'] == '0.4' else 1

        # Port ID depends on whether we *just entered* traveling state or *are* traveling
        # We need to look at events *in this interval* to decide the state *at time t*

        entry_event = events_in_interval[
            ((events_in_interval['key'] == 'head') | (events_in_interval['key'] == 'trial')) & (
                        events_in_interval['value'] == 1.0)
            ]
        exit_event = events_in_interval[
            (events_in_interval['key'] == 'head') & (events_in_interval['value'] == 0.0)
            ]

        # State Determination Logic:
        if not entry_event.empty:  # Entered a port *during* this interval
            is_traveling = False
            travel_timer = 0.0  # Reset travel timer upon entry
            current_port_id = port_map.get(entry_event['port'].iloc[0], -1)
            time_in_port_t = 0.0  # Just entered
            event_timer_t = 0.0
            last_reward_time_in_port = t  # Start event timer from entry
            if current_port_id == 0:
                rewards_in_context = 0  # Reset context rewards on entry
        elif is_traveling:  # Was already traveling before this interval started
            current_port_id = 2  # Traveling state
            time_in_port_t = travel_timer  # Use the timer value from *previous* step end
            event_timer_t = travel_timer  # Event timer also counts travel time? Or reset? Let's keep it counting travel time.
            travel_timer += dt  # Increment for *this* interval
        else:  # Was in a port before this interval started
            current_port_id = port_map.get(row_t['port'],
                                           0 if i == 0 else current_port_id)  # Persist last known port if ffill failed
            time_in_port_t = row_t['time_in_port'] if pd.notna(row_t['time_in_port']) else (
                time_in_port_t + dt if i > 0 else 0.0)  # Increment if no event
            event_timer_t = max(0.0, t - last_reward_time_in_port)

        obs_t = np.array([
            float(current_port_id),
            round(time_in_port_t, 3),
            round(event_timer_t, 3),
            float(current_context),
            float(rewards_in_context),
            float(gambling_disabled)
        ])

        # --- Determine Action at time t ---
        # todo: right now action only records leaving, but doesn't record entering (which is a decision to leave the traveling port)
        action_t = 0  # Default: Stay (or continue traveling implicitly)
        if not exit_event.empty:  # Exited a port *during* this interval
            action_t = 1  # Leave action occurred at time t
            is_traveling = True  # Will be traveling in the next state
            leaving_port = current_port_id  # The port just exited
            travel_timer = 0.0  # Start timer *after* this step
            # Update gambling_disabled if leaving context port early
            if leaving_port == 0 and rewards_in_context < context_rewards_max:
                gambling_disabled = True

        # --- Determine Reward received between t and t_next ---
        reward_t_plus_1 = 0.0
        reward_events = events_in_interval[
            (events_in_interval['key'] == 'reward') & (events_in_interval['value'] == 1.0)
            ]
        if not reward_events.empty:
            reward_t_plus_1 = 1.0
            # Only update reward counters/timers if *not* currently traveling
            if not is_traveling:
                last_reward_time_in_port = t  # Time of the *start* of the interval where reward occurred
                if current_port_id == 0:
                    rewards_in_context += 1
                    if rewards_in_context >= context_rewards_max:
                        gambling_disabled = False

        # --- Determine Next State (obs_t_plus_1) ---
        # Look ahead one step in the resampled data
        if i + 1 < num_steps:
            row_t_next = df_resampled.iloc[i + 1]
            next_context_from_data = 0 if row_t_next['phase'] == '0.4' else 1
        else:  # Handle the very last step
            row_t_next = row_t  # Assume state persists if no more data
            next_context_from_data = current_context

        # Determine next state based on *current* calculations
        next_port_id_calc = current_port_id
        next_time_in_port_calc = time_in_port_t + dt
        next_event_timer_calc = event_timer_t + dt
        next_rewards_in_context_calc = rewards_in_context
        next_gambling_disabled_calc = gambling_disabled
        next_context_calc = current_context

        # Check for upcoming block switch *after* this interval
        if t_next > 0 and t_next % block_duration < dt:
            next_context_calc = 1 - current_context

        if action_t == 1:  # If we just decided to leave
            next_port_id_calc = 2  # Next state is traveling
            next_time_in_port_calc = 0.0  # Travel timer starts now
            next_event_timer_calc = 0.0
            # rewards_in_context remains as is until entry event
        elif is_traveling:  # If we were traveling *during* this interval
            # Check if *next* interval contains an entry event
            next_entry_events = event_df[
                (event_df['task_time'] >= t_next) &
                (event_df['task_time'] < t_next + dt) &
                (event_df['key'] == 'head') &
                (event_df['value'] == 1.0)
                ]
            if not next_entry_events.empty:  # Arrival happens in the next step
                next_port_id_calc = port_map.get(next_entry_events['port'].iloc[0], -1)
                next_time_in_port_calc = 0.0
                next_event_timer_calc = 0.0
                if next_port_id_calc == 0:
                    next_rewards_in_context_calc = 0  # Reset context rewards on entry
            else:  # Still traveling
                next_port_id_calc = 2
                next_time_in_port_calc = travel_timer  # Use the updated timer
                next_event_timer_calc = travel_timer  # Continue counting

        # If reward occurred, reset next event timer
        if reward_t_plus_1 > 0 and not is_traveling:  # Only if not traveling
            next_event_timer_calc = 0.0

        obs_t_plus_1 = np.array([
            float(next_port_id_calc),
            round(next_time_in_port_calc, 3),
            round(next_event_timer_calc, 3),
            float(next_context_calc),
            float(next_rewards_in_context_calc),
            float(next_gambling_disabled_calc)
        ])

        # --- Determine Terminal Flag ---
        # Terminal if this is the last step in the resampled index or session ends
        terminal_flag = (i == num_steps - 1) or (t_next >= session_duration)

        # --- Append Transition ---
        transitions.append((obs_t, action_t, reward_t_plus_1, obs_t_plus_1, terminal_flag))

        # Update last known port for next iteration's entry detection
        if current_port_id != 2:  # Don't update last_port_id while traveling
            last_port_id = current_port_id

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
