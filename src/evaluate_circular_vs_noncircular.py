import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import your modules
from data_loader import load_behavior_data, load_pretraining_data, convert_behavior_data_to_state_transitions
from mouse_playback_environment import MousePlaybackEnvironment
from mouse_playback_agent import MousePlaybackAgent


def train_single_session(df, env_params, is_circular, gamma=0.95, max_epochs=50):
    """
    Trains an agent on a single session's dataframe.
    """
    print(f"\n--- Generating Transitions (Circular={is_circular}) ---")
    transitions = convert_behavior_data_to_state_transitions(df, env_params, is_circular=is_circular)

    # 1. Environment Setup
    env = MousePlaybackEnvironment()
    env.env_init({"transitions": transitions, "time_step_duration": env_params["time_step_duration"]})

    # 2. Agent Setup
    # Using the optimal scales and the expanded IHT size we discussed
    scales = [-1, 2.0, 0.0, -1, -1, -1]
    agent_params = {
        "discount": gamma,
        "lambda": 0.95,
        "step_size": 0.004,
        "num_tilings": 8,
        "iht_size": 1048576,  # 2^20 to prevent collisions
        "scales": scales
    }
    agent = MousePlaybackAgent()
    agent.agent_init(agent_params)

    # 3. Training Loop
    num_steps = len(transitions)
    last_w = agent.w.copy()

    print(f"Training Agent (Circular={is_circular})...")
    for epoch in range(max_epochs):
        current_observation = env.env_start()
        agent.agent_start(current_observation)

        for step_idx in range(num_steps):
            reward, next_observation, terminal = env.env_step(action=None)
            if terminal:
                agent.agent_end(reward)
                if env.current_step_index < num_steps:
                    current_observation = next_observation
                    agent.agent_start(current_observation)
                else:
                    break
            else:
                agent.agent_step(reward, next_observation)

        # Convergence Check
        w_change = np.sqrt(np.sum((agent.w - last_w) ** 2))
        if w_change < 0.02 and epoch > 0:
            print(f"  -> Converged at Epoch {epoch + 1} (L2 Norm: {w_change:.6f})")
            break

        last_w = agent.w.copy()
        env.current_step_index = 0  # Reset environment for next epoch

    return agent


def plot_gambling_values(agent_circ, agent_noncirc, max_time=15.0, step_size=0.1):
    """
    Plots V(s) vs time_in_port in the Gambling Port, separated into two side-by-side subplots
    for Circular and Non-Circular setups to prevent overlapping lines from being hidden.
    """
    time_sweep = np.arange(0, max_time + step_size, step_size)
    set2 = sns.color_palette('Set2')

    # Create a figure with 1 row and 2 columns of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ==========================================
    # 1. Circular Setup (Farsighted) on Axis 1
    # ==========================================
    v_low_circ, v_high_circ = [], []
    for t in time_sweep:
        s_low = np.array([1.0, t, 0.0, 0.0, 4.0, 0.0])
        s_high = np.array([1.0, t, 0.0, 1.0, 4.0, 0.0])

        v_low_circ.append(agent_circ.get_value(s_low))
        v_high_circ.append(agent_circ.get_value(s_high))

    ax1.plot(time_sweep, v_low_circ, label='Context 0 (Low)', color=set2[0], linewidth=2.5)
    ax1.plot(time_sweep, v_high_circ, label='Context 1 (High)', color=set2[1], linewidth=2.5)
    ax1.set_title("Circular Model (Continuous Session)", fontsize=14)
    ax1.set_xlabel("Time in Port (seconds)", fontsize=12)
    ax1.set_ylabel("Expected Value V(s)", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ==========================================
    # 2. Non-Circular Setup (Myopic) on Axis 2
    # ==========================================
    v_low_noncirc, v_high_noncirc = [], []
    for t in time_sweep:
        s_low = np.array([1.0, t, 0.0, 0.0, 4.0, 0.0])
        s_high = np.array([1.0, t, 0.0, 1.0, 4.0, 0.0])

        v_low_noncirc.append(agent_noncirc.get_value(s_low))
        v_high_noncirc.append(agent_noncirc.get_value(s_high))

    # We use dashed lines here to differentiate the models stylistically
    ax2.plot(time_sweep, v_low_noncirc, label='Context 0 (Low)', color=set2[0], linestyle='--', linewidth=3.5)
    ax2.plot(time_sweep, v_high_noncirc, label='Context 1 (High)', color=set2[1], linestyle=':', linewidth=2.5)
    ax2.set_title("Non-Circular Model (Trial-by-Trial)", fontsize=14)
    ax2.set_xlabel("Time in Port (seconds)", fontsize=12)
    # Keeping y-axis label off ax2 for cleaner look, but you can uncomment below if desired
    # ax2.set_ylabel("Expected Value V(s)", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Master title and layout adjustments
    plt.suptitle("Learned Value of the Gambling Port: Strategy Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_travel_values(agent_circ, agent_noncirc, max_travel_time=0.8, step_size=0.1):
    """
    Plots V(s) vs travel time (Port 2) for both setups.
    """
    travel_sweep = np.arange(0, max_travel_time + step_size, step_size)
    set2 = sns.color_palette('Set2')

    plt.figure(figsize=(12, 6))

    for agent, ls, label_prefix in zip([agent_circ, agent_noncirc], ['-', '--'], ['Circular', 'Non-Circ']):
        v_low, v_high = [], []
        for t in travel_sweep:
            # State: [port=2, time_in_port=t, event_timer=t, context, rewards=0, gambling_disabled=1]
            s_low = np.array([2.0, t, t, 0.0, 4.0, 1.0])
            s_high = np.array([2.0, t, t, 1.0, 4.0, 1.0])

            v_low.append(agent.get_value(s_low))
            v_high.append(agent.get_value(s_high))

        plt.plot(travel_sweep, v_low, label=f'{label_prefix} - Context 0 (Low)', color=set2[0], linestyle=ls,
                 linewidth=2.5)
        plt.plot(travel_sweep, v_high, label=f'{label_prefix} - Context 1 (High)', color=set2[1], linestyle=ls,
                 linewidth=2.5)

    plt.title("Travel State Value (Gambling -> Context): Circular vs Non-Circular", fontsize=14)
    plt.xlabel("Travel Time (seconds)", fontsize=12)
    plt.ylabel("Expected Value V(s)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_value_difference(agent_circ, agent_noncirc, max_time=15.0, step_size=0.1):
    """
    Plots the difference (V_circ - V_noncirc) to reveal the hidden
    baseline shift caused by the future Context port.
    """
    time_sweep = np.arange(0, max_time + step_size, step_size)
    set2 = sns.color_palette('Set2')

    plt.figure(figsize=(8, 5))

    diff_low, diff_high = [], []
    for t in time_sweep:
        s_low = np.array([1.0, t, 0.0, 0.0, 4.0, 0.0])
        s_high = np.array([1.0, t, 0.0, 1.0, 4.0, 0.0])

        # Calculate the pure difference (subtracting out the local patch value)
        diff_low.append(agent_circ.get_value(s_low) - agent_noncirc.get_value(s_low))
        diff_high.append(agent_circ.get_value(s_high) - agent_noncirc.get_value(s_high))

    plt.plot(time_sweep, diff_low, label='Context 0 (Low) Difference', color=set2[0], linewidth=2.5)
    plt.plot(time_sweep, diff_high, label='Context 1 (High) Difference', color=set2[1], linewidth=2.5)

    plt.title("The Hidden Future: (Circular V) - (Non-Circular V)", fontsize=14)
    plt.xlabel("Time in Port (seconds)", fontsize=12)
    plt.ylabel("Difference in Value ($\Delta V$)", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)  # Reference line for zero difference
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Environment parameters
    env_params = {
        "time_step_duration": 0.1,
        "session_duration_min": 18,
        "context_rewards_max": 4,
        "block_duration_min": 3
    }

    # 1. Load data for one session
    print("Loading raw dataframe...")
    df = load_pretraining_data("SZ036", session_id=5)  # Adjust ID as needed

    if df is not None:
        # Note: We use gamma=0.95 so the Circular agent can successfully look across the travel gap
        agent_circular = train_single_session(df, env_params, is_circular=True, gamma=0.78)
        agent_noncircular = train_single_session(df, env_params, is_circular=False, gamma=0.78)

        # 2. Plotting
        print("\nPlotting Results...")
        plot_travel_values(agent_circular, agent_noncircular)
        plot_gambling_values(agent_circular, agent_noncircular)
        plot_value_difference(agent_circular, agent_noncircular, max_time=15.0, step_size=0.1)