import numpy as np
import pandas as pd
from rl_glue import RLGlue
from foraging_environment import ForagingEnvironment
from q_tile_coding_agent import QLearningTileCodingAgent  # The agent we just created

import matplotlib.pyplot as plt


def run_rlglue_session(alpha, gamma, epsilon=0.1, seed=42):
    """
    Runs a single session using the standard RLGlue interface.
    """

    # 1. Instantiate Environment and Agent
    env = ForagingEnvironment()
    agent = QLearningTileCodingAgent()

    # 2. Setup parameters
    env_info = {"seed": seed}
    agent_info = {
        "step_size": alpha,
        "discount": gamma,
        "epsilon": epsilon,
        "num_tilings": 8,
        "iht_size": 32768
    }

    # 3. Explicitly initialize (since rl_init internals are commented out in your rl_glue.py)
    env.env_init(env_info)
    agent.agent_init(agent_info)

    # 4. Create RLGlue instance
    rl_glue = RLGlue(env, agent)

    # 5. Start the episode
    rl_glue.rl_start()
    print(f"Started RLGlue simulation with alpha={alpha}, gamma={gamma}...")

    # Data storage arrays
    log_time = []
    log_port = []
    log_context = []
    log_is_traveling = []
    log_action = []
    log_reward = []

    is_terminal = False

    # 6. The Standard RL-Glue Loop
    while not is_terminal:
        # Take one step in the environment and agent
        reward, state, action, is_terminal = rl_glue.rl_step()

        # Log the state of the environment AFTER the step
        log_time.append(env.total_time_elapsed)
        log_port.append(env.port_id)
        log_context.append(env.current_context)
        log_is_traveling.append(env.is_traveling)
        log_action.append(action)
        log_reward.append(reward)

    print(f"Simulation complete. Total steps: {rl_glue.rl_num_steps()}")

    # 7. Extract RPEs and structure the Data
    # RLGlue allows us to pass messages directly to the agent!
    rpe_log = rl_glue.rl_agent_message("get_rpe_log")

    # Create a DataFrame for easy analysis and plotting
    session_data = pd.DataFrame({
        "time_elapsed": log_time,
        "port_id": log_port,
        "context": log_context,
        "action": log_action,  # The action the agent will take NEXT (or None if terminal)
        "reward": log_reward,
        "rpe": rpe_log
    })

    return session_data


def plot_agent_environment_interaction(alpha=0.05, gamma=0.90, epsilon=0.1, seed=42):
    print(f"\n--- Running Agent Interaction Visualization (alpha={alpha}, gamma={gamma}) ---")

    # 1. Initialize Env, Agent, and RLGlue
    env = ForagingEnvironment()
    agent = QLearningTileCodingAgent()

    env.env_init({"seed": seed})
    agent.agent_init({
        "step_size": alpha,
        "discount": gamma,
        "epsilon": epsilon,
        "num_tilings": 8,
        "iht_size": 32768
    })

    rl_glue = RLGlue(env, agent)
    rl_glue.rl_start()

    # 2. Data tracking lists
    states = []
    rewards_received = []
    trial_numbers = []
    actions = []

    is_terminal = False

    # 3. The Interaction Loop
    while not is_terminal:
        # Capture current state BEFORE the step (using env's internal method for logging)
        current_obs = env._get_observation()
        states.append(current_obs)
        trial_numbers.append(env.env_message("get_trial_number"))

        # Take a step
        reward, state, action, is_terminal = rl_glue.rl_step()

        rewards_received.append(reward)
        actions.append(action)

    # 4. Convert to numpy arrays for easy plotting
    states = np.array(states)
    rewards_received = np.array(rewards_received)
    trial_numbers = np.array(trial_numbers)

    # RLGlue returns None for action on the terminal step, so we append a dummy 0 for alignment
    actions_padded = np.array([a if a is not None else 0 for a in actions])

    # 5. Plotting (Similar to your smoke test, but tracking Action as well)
    fig, axes = plt.subplots(9, 1, figsize=(16, 16), sharex=True)
    time_points = np.arange(states.shape[0]) * env.dt

    axes[0].plot(time_points, states[:, 0], '.-')
    axes[0].set_ylabel("Port ID\n(0=C, 1=G, 2=T)")
    axes[1].plot(time_points, states[:, 1], '.-')
    axes[1].set_ylabel("time_in_port (s)")
    axes[2].plot(time_points, states[:, 2], '.-')
    axes[2].set_ylabel("event_timer (s)")
    axes[3].plot(time_points, states[:, 3], '.-')
    axes[3].set_ylabel("Context\n(0=Low, 1=High)")
    axes[4].plot(time_points, states[:, 4], '.-')
    axes[4].set_ylabel("Context Rewards")
    axes[5].plot(time_points, states[:, 5], '.-')
    axes[5].set_ylabel("Gambling Disabled")
    axes[6].plot(time_points, rewards_received, '-')
    axes[6].set_ylabel("Reward Received")
    axes[6].set_ylim([-0.1, 1.1])
    axes[7].plot(time_points, trial_numbers, '-')
    axes[7].set_ylabel("Trial Number")

    # New Subplot: Agent Actions
    axes[8].plot(time_points, actions_padded, '.', color='red', alpha=0.5)
    axes[8].set_ylabel("Action\n(0=Stay, 1=Leave)")
    axes[8].set_xlabel("Total Time (s)")
    axes[8].set_yticks([0, 1])

    # Draw vertical lines for block durations
    block_duration = env.block_duration
    session_duration = env.session_duration
    for switch_time in np.arange(block_duration, session_duration, block_duration):
        axes[3].axvline(x=switch_time, color='k', linestyle='--', alpha=0.7)
        axes[7].axvline(x=switch_time, color='k', linestyle='--', alpha=0.7)

    for i in range(9):
        axes[i].set_xlim(800, 1000)

    fig.suptitle("Q-Learning Agent Interaction Trajectories", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save instead of show to prevent blocking
    # plt.savefig("agent_smoke_test_plot.png")
    # print("Plot saved to 'agent_smoke_test_plot.png'")
    plt.show()

    return states, actions_padded


def train_agent_multiple_sessions(num_sessions=10, alpha=0.05, gamma=0.90):
    # 1. Initialize the Environment and ONE Agent
    env = ForagingEnvironment()
    agent = QLearningTileCodingAgent()

    # 2. Set up Agent Parameters (including the calculated decay rate)
    agent_info = {
        "step_size": alpha,
        "discount": gamma,
        "epsilon_start": 0.9,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.99998,  # 0.9999732 decays to 0.05 over 108,000 steps
        "num_tilings": 8,
        "iht_size": 32768
    }
    agent.agent_init(agent_info)

    all_sessions_data = []

    print(f"Starting multi-session training for {num_sessions} sessions...")

    # 3. Loop over the sessions
    for session_idx in range(num_sessions):
        # Initialize a fresh environment for this session
        env_info = {"seed": 42 + session_idx}  # Different seed for env randomness each day
        env.env_init(env_info)

        # Link them with RLGlue
        rl_glue = RLGlue(env, agent)
        rl_glue.rl_start()

        # Data tracking lists for THIS session
        log_session = []
        log_trial = []
        log_time = []
        log_port = []
        log_time_in_port = []
        log_context = []
        log_event_timer = []
        log_disabled = []
        log_action = []
        log_reward = []

        is_terminal = False

        # Run the 18-minute session
        while not is_terminal:
            # We capture the state BEFORE taking the step so the action
            # aligns perfectly with the state that triggered it
            current_action = rl_glue.last_action

            # Extract state variables directly from the environment
            current_time = env.total_time_elapsed
            current_port = env.port_id
            current_time_in_port = env.time_in_port
            current_context = env.current_context
            current_event_timer = env.event_timer
            current_disabled = env.gambling_port_disabled
            current_trial = env.env_message("get_trial_number")

            # Take the step
            reward, state, next_action, is_terminal = rl_glue.rl_step()

            # Log everything
            log_session.append(session_idx + 1)
            log_trial.append(current_trial)
            log_time.append(current_time)
            log_port.append(current_port)
            log_time_in_port.append(current_time_in_port)
            log_context.append(current_context)
            log_event_timer.append(current_event_timer)
            log_disabled.append(current_disabled)
            log_action.append(current_action)
            log_reward.append(reward)

        print(f"Session {session_idx + 1}/{num_sessions} complete. Final Epsilon: {agent.epsilon:.4f}")

        # 5. Extract the TD-errors (RPEs) for just this session
        all_rpes = agent.agent_message("get_rpe_log")
        # Slice the last N RPEs where N is the number of steps we just took
        session_rpes = all_rpes[-len(log_time):]

        # Create the comprehensive DataFrame
        session_df = pd.DataFrame({
            "session": log_session,
            "trial": log_trial,
            "time_elapsed": log_time,
            "port_id": log_port,
            "time_in_port": log_time_in_port,
            "context": log_context,
            "event_timer": log_event_timer,
            "gambling_disabled": log_disabled,
            "action": log_action,
            "reward": log_reward,
            "td_error": session_rpes
        })

        all_sessions_data.append(session_df)

        # Note: We DO NOT call agent_cleanup() so the agent retains its weights!

    # Combine all sessions into one large DataFrame
    full_training_data = pd.concat(all_sessions_data, ignore_index=True)
    return full_training_data, agent


# --- Execution Example ---
if __name__ == "__main__":
    # Plug in the best parameters you obtained from fitting the mouse data
    best_alpha = 0.001
    best_gamma = 0.78
    # train_agent_multiple_sessions(num_sessions=10, alpha=best_alpha, gamma=best_gamma)
    plot_agent_environment_interaction(alpha=best_alpha, gamma=best_gamma, epsilon=0.9)
    # behavioral_df = run_rlglue_session(alpha=best_alpha, gamma=best_gamma)

    # print("\nFirst 10 steps of the session:")
    # print(behavioral_df.head(10))