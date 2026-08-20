# src/rl_config.py

# --- Environment Parameters ---
# Used primarily by your data_loader to parse the raw behavior
ENV_PARAMS = {
    "time_step_duration": 0.1,
    "travel_time": 0.4,
    "session_duration_min": 18,
    "context_rewards_max": 4,
    "block_duration_min": 3,
    "gambling_cumulative": 8.0,
    "gambling_starting": 1.0,
}

# --- State Representation Scales ---
# The definitive scales for the tile coder.
# Dimensions: [port, time_in_port, event_timer, context, rewards_in_context, gambling_disabled]
STATE_SCALES = [
    -1,   # 0: port (discrete/ignored by tile coder)
    2.0,  # 1: time_in_port
    2.0,  # 2: event_timer
    -1,   # 3: context (discrete)
    -1,   # 4: rewards_in_context (discrete)
    -1    # 5: gambling_disabled (discrete)
]

# --- Agent Initialization Template ---
# Used across your fitting, training, and plotting scripts
AGENT_INFO_TEMPLATE = {
    "discount": 0.95,       # Default fallback; usually overwritten by the results of the grid search
    "step_size": 0.0005,     # Default fallback
    "num_tilings": 8,
    "iht_size": 32768,      # Your integer hash size to prevent memory bloat
    "gambling_max_time_s": 30.0,
    "context_rewards_max": ENV_PARAMS["context_rewards_max"],
    "scales": STATE_SCALES
}