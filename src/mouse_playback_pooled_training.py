import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import itertools
from mouse_playback_environment import MousePlaybackEnvironment
from mouse_playback_agent import MousePlaybackAgent

def configure_agent_and_env(pooled_data_file, use_redundant_features):
    print(f"Loading pooled transitions from {pooled_data_file}...")
    try:
        with open(pooled_data_file, 'rb') as f:
            pooled_transitions = pickle.load(f)
        print(f"Loaded {len(pooled_transitions)} transitions.")
    except FileNotFoundError:
        print(f"Error: {pooled_data_file} not found.")
        pooled_transitions = None
    except Exception as e:
        print(f"An error occurred loading the pickle file: {e}")
        pooled_transitions = None

    # Set up the environment and the agent
    # Define the full 6-feature configurations
    full_scales = [
        -1,  # 0: port
        1 / 2.0,  # 1: time_in_port
        1 / 1.0,  # 2: event_timer
        -1,  # 3: context
        -1,  # 4: rewards_in_context
        -1  # 5: gambling_disabled
    ]

    # Create the final config
    if use_redundant_features:
        scales_to_use = full_scales
        print("Agent will use 6 features (including event_timer).")
    else:
        # To "remove" a feature, we set its scale to 0.0
        # and its tiling dimension to 1 (so it's not subdivided).
        scales_to_use = [
            -1,  # 0: port
            1 / 2.0,  # 1: time_in_port
            0.0,  # 2: event_timer (SCALE SET TO 0)
            -1,  # 3: context
            0.0,  # 4: rewards_in_context
            -1  # 5: gambling_disabled
        ]
        print("Agent will use 4 features (event_timer and context_rewards scales are 0).")

    if pooled_transitions:
        print("Initializing agent and environment...")

        env = MousePlaybackEnvironment()
        env_params_for_playback = {"transitions": pooled_transitions, "time_step_duration": 0.1}
        env.env_init(env_params_for_playback)

        agent = MousePlaybackAgent()
        agent_params = {
            "discount": 0.95,  # Or chosen value
            "step_size": 0.1,  # Start with a smaller step size for stability over large datasets
            "num_tilings": 16,
            "iht_size": 4096,
            "gambling_max_time_s": 30.0,  # Match data_loader params if used
            "context_rewards_max": 4,  # Match data_loader params if used
            "scales": scales_to_use
        }  # agent parameters
        agent.agent_init(agent_params)
        return agent, env

def pooled_training(agent, env, max_epochs=50):
    max_epochs = max_epochs  # Adjust as needed
    convergence_threshold = 0.02  # Adjust as needed
    last_w = agent.w.copy()
    weight_change_across_epochs = []

    num_total_steps = len(env.transitions)  # Total transitions in the pooled list

    for epoch in range(max_epochs):
        print(f"Starting Epoch {epoch + 1}/{max_epochs}")

        # --- Start the very first episode of the epoch ---
        current_observation = env.env_start()  # Resets env index to 0
        agent.agent_start(current_observation)

        # for step_idx in tqdm(range(num_total_steps)):
        for step_idx in range(num_total_steps):
            # Action is ignored by the playback environment
            reward, next_observation, terminal = env.env_step(action=None)

            if terminal:
                agent.agent_end(reward)
                # Check if this is the end of the entire pooled list
                if env.current_step_index < num_total_steps:
                    # --- Start the next episode within the epoch ---
                    # The environment has already advanced its index, so next_observation
                    # is the start state of the next session in the pool.
                    current_observation = next_observation
                    agent.agent_start(current_observation)
                else:
                    # This was the very last transition in the pool
                    break  # End the epoch
            else:
                # Standard step within an episode
                agent.agent_step(reward, next_observation)
                current_observation = next_observation  # Update for the next iteration if not terminal

        # --- Check for convergence after each epoch ---
        w_change = np.sqrt(np.sum((agent.w - last_w) ** 2))
        weight_change_across_epochs.append(w_change)
        print(f"Epoch {epoch + 1} finished. Weight change (L2 norm): {w_change:.8f}")
        if w_change < convergence_threshold and epoch > 0:
            print("Convergence detected.")
            break
        last_w = agent.w.copy()
        # Reset environment index for the next epoch pass (handled by env.env_start() at loop top)

    return agent


def generate_probe_states_dict(examine=False):
    # --- 1. Define the dimensions of interest ---
    times_in_port = np.arange(0.0, 25.0, 0.1)  # [0.5, 1.5, ..., 14.5]
    times_in_port_context = np.arange(0.0, 12.0, 0.1)
    contexts = [0.0, 1.0]  # [Low, High]

    probe_states_dict = {}

    # --- 2. Generate GAMBLING Port (port=0) probe states ---
    # State vector: [port, time_in_port, event_timer, context, rewards_in_context, gambling_disabled]

    # Use itertools.product to get all combinations
    gambling_combinations = itertools.product(times_in_port, contexts)

    for tip, context in gambling_combinations:
        # In Gambling port, rewards_in_context is 4
        # and event_timer matches time_in_port
        state_vector = [
            1.0,  # port = 1 (Gambling)
            round(tip, 3),  # time_in_port
            0.0,  # event_timer (collapsed to 1d because we don't care about it for now)
            context,  # context (0=Low, 1=High)
            0.0,  # rewards_in_context (collapsed to 1d)
            0.0
        ]

        # Create a descriptive name for plotting
        context_name = "Low" if context == 0.0 else "High"
        state_name = f"V(Gamb_{context_name}_tip={tip}s)"

        probe_states_dict[state_name] = state_vector

    # --- 3. Generate CONTEXT Port (port=0) probe states ---
    # Use itertools.product for all combinations
    context_combinations = itertools.product(times_in_port_context, contexts)

    for tip, context in context_combinations:
        # In Context port, event_timer matches time_in_port
        state_vector = [
            0.0,  # port = 1 (Context)
            round(tip, 3),  # time_in_port
            0.0,  # event_timer (collapsed to 1d)
            context,  # context (0=Low, 1=High)
            0.0,  # rewards_in_context (collapsed to 1d)
            1.0
        ]

        # Create a descriptive name for plotting
        context_name = "Low" if context == 0.0 else "High"
        state_name = f"V(Cont_{context_name}_tip={tip}s)"

        probe_states_dict[state_name] = state_vector

    # --- 4. Check the results ---
    if examine:
        print(f"Generated a total of {len(probe_states_dict)} probe states.")
        print("\n--- Example Gambling States ---")
        g_keys = [k for k in probe_states_dict if k.startswith("V(Gamb")][::50]  # Show a few
        for k in g_keys:
            print(f"{k}: {probe_states_dict[k]}")

        print("\n--- Example Context States ---")
        c_keys = [k for k in probe_states_dict if k.startswith("V(Cont")][::250]  # Show a few
        for k in c_keys:
            print(f"{k}: {probe_states_dict[k]}")

    return probe_states_dict

def calculate_learned_values(agent, probe_states_dict):
    print("Calculating learned values for all probe states...")

    # 1. Create a dictionary to store the final learned values
    final_probe_values = {}

    # 2. Iterate through the probe_states_dict you defined earlier
    for state_name, state_vector in probe_states_dict.items():
        # 3. Get the learned value from the agent for each state
        value = agent.get_value(state_vector)
        final_probe_values[state_name] = value

    print("Calculation complete.")

    # 4. Print a few examples to check
    print("\n--- Example Learned Values ---")
    example_keys = list(final_probe_values.keys())[::100]  # Show some samples
    for k in example_keys:
        print(f"{k}: {final_probe_values[k]:.4f}")

    # 5. Convert to a DataFrame for plotting and analysis
    print("\nConverting to DataFrame for analysis...")
    value_list = []
    for state_name, value in final_probe_values.items():
        # Parse the state_name to get sortable columns for plotting
        port_str = "Gambling" if "Gamb" in state_name else "Context"
        context_str = "High" if "High" in state_name else "Low"
        time_str = state_name.split("=")[-1].replace("s", "").replace(")", "")

        value_list.append({
            "state_name": state_name,
            "port": port_str,
            "context": context_str,
            "time_in_port": float(time_str),
            "learned_value": value
        })

    final_value_df = pd.DataFrame(value_list)

    print("DataFrame created. You can now plot V(s) vs. time_in_port.")
    print(final_value_df.head())
    return final_value_df

def plot_learned_values(final_value_df, animal_id):
    # Plot learned values in both ports
    gambling_data = final_value_df[final_value_df['port'] == 'Gambling']
    custom_palette = {
        'Low': sns.color_palette('Set2')[0],
        'High': sns.color_palette('Set2')[1]
    }
    plt.figure(figsize=(12, 7))
    ax1 = sns.lineplot(
        data=gambling_data,
        x='time_in_port',
        y='learned_value',
        hue='context',
        style='context',
        palette=custom_palette,
        markers=True,
        lw=2
    )
    ax1.set_title(f'{animal_id}: Learned Value vs. Time in Gambling Port', fontsize=16)
    ax1.set_xlabel('Time in Port (seconds)', fontsize=12)
    ax1.set_ylabel('Learned State Value V(s)', fontsize=12)
    ax1.legend(title='Context', fontsize=11)

    # Save the figure
    # gambling_plot_filename = 'gambling_port_learned_value.png'
    # plt.savefig(gambling_plot_filename)
    # print(f"Successfully saved Gambling Port plot to {gambling_plot_filename}")
    plt.show()  # Display the plot in the notebook

    # ------------------------------------------------------------
    context_data = final_value_df[final_value_df['port'] == 'Context']

    plt.figure(figsize=(12, 7))
    ax2 = sns.lineplot(
        data=context_data,
        x='time_in_port',
        y='learned_value',
        hue='context',
        style='context',
        palette=custom_palette,  # <-- YOUR CUSTOM COLORS ARE HERE
        markers=True,
        lw=2
    )
    ax2.set_title(f'{animal_id}: Learned Value vs. Time in Context Port', fontsize=16)
    ax2.set_xlabel('Time in Port (seconds)', fontsize=12)
    ax2.set_ylabel('Learned State Value V(s)', fontsize=12)
    ax2.legend(title='Context', fontsize=11)

    # Save the figure
    # context_plot_filename = 'context_port_learned_value_custom_color.png'
    # plt.savefig(context_plot_filename)
    # print(f"Successfully saved Context Port plot to {context_plot_filename}")
    plt.show()  # Display the plot in the notebook

def save_learned_values(final_value_df, animal_id):
    filename = f"learned_values_vs_time_in_port_{animal_id}.parquet"
    final_value_df.to_parquet(filename)

def main():
    SZ_animals = ['SZ038', 'SZ039', 'SZ042', 'SZ043']
    RK_animals = ['RK007', 'RK008']
    animals = SZ_animals + RK_animals
    for animal_id in animals:
        pooled_data_file = f"pooled_transitions_{animal_id}.pkl"
        agent, env = configure_agent_and_env(pooled_data_file, use_redundant_features=False)
        agent = pooled_training(agent, env)
        probe_states_dict = generate_probe_states_dict(examine=False)
        value_df = calculate_learned_values(agent, probe_states_dict)
        save_learned_values(value_df, animal_id)
        plot_learned_values(value_df, animal_id)
        # # --- Save the converged weights ---
        # converged_weights = agent.w.copy()
        # np.save(f"expert_weights_{animal_id}_multi_epoch.npy", converged_weights)
        # print("Saved converged expert weights.")

if __name__ == "__main__":
    main()