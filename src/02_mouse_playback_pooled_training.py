import os
import pickle
from tqdm import tqdm
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.mouse_playback_environment import MousePlaybackEnvironment
from src.mouse_playback_agent import MousePlaybackAgent

from src.rl_config import AGENT_INFO_TEMPLATE
import src.config


def configure_agent_and_env(pooled_data_file, alpha, gamma):
    """
    Configures the environment and agent using specific hyperparameters.
    """
    print(f"Loading pooled transitions from {pooled_data_file}...")
    try:
        with open(pooled_data_file, 'rb') as f:
            pooled_transitions_with_meta = pickle.load(f)
            pooled_transitions = [t[:5] for t in pooled_transitions_with_meta]
    except Exception as e:
        print(f"Error loading {pooled_data_file}: {e}")
        return None, None

    # Environment Setup
    env = MousePlaybackEnvironment()
    env_params = {"transitions": pooled_transitions, "time_step_duration": 0.1}
    env.env_init(env_params)

    # Agent Setup
    agent_params = AGENT_INFO_TEMPLATE.copy()
    agent_params['step_size'] = alpha
    agent_params['discount'] = gamma

    # Agent Setup
    agent = MousePlaybackAgent()
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
        # This bar tracks the steps within ONE epoch for ONE animal
        with tqdm(total=num_total_steps, desc=f"   Epoch {epoch + 1}/{max_epochs}", unit="step", leave=False) as pbar:
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
                pbar.update(1)

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


def batch_train_and_save(best_params):
    project_root = Path(config.MODELING_PROJECT_ROOT)
    save_path = project_root / Path(config.STEP2_PRETRAINED_AGENTS_SUBDIR)
    data_path = project_root / Path(config.MODELING_DATA_SUBDIR)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Batch training {len(best_params)} agents...")

    # Overall Progress Bar
    animals_pbar = tqdm(best_params.items(), desc="🧬 Overall Cohort", unit="animal")

    for animal_id, params in animals_pbar:
        animals_pbar.set_postfix(current=animal_id)

        data_file = data_path / f"pooled_transitions_{animal_id}.pkl"

        # --- NEW CLEAN INVOCATION ---
        # No re-initialization needed; agent is born with the right params
        agent, env = configure_agent_and_env(
            data_file,
            alpha=params['alpha'],
            gamma=params['gamma']
        )

        if agent is None:
            continue

        # Train with inner progress bar
        trained_agent = pooled_training(agent, env, max_epochs=1)

        # Save results
        output_file = save_path / f"pretrained_agent_{animal_id}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(trained_agent, f)

    print("\n🎉 All agents trained and saved.")


def train_agent_for_animal(animal_id, max_epochs=1):
    """
    Pipeline-ready function to train and save an agent for a specific animal.
    """
    project_root = src.config.MODELING_PROJECT_ROOT

    params_filepath = os.path.join(project_root, src.config.STEP1_PARAMETER_FITTING_SUBDIR, "best_params.pkl")

    try:
        with open(params_filepath, 'rb') as f:
            global_params_dict = pickle.load(f)

        if animal_id not in global_params_dict:
            print(f"❌ Error: {animal_id} not found in {params_filepath}.")
            return

        best_params_series = global_params_dict[animal_id]

    except Exception as e:
        print(f"❌ Error loading parameters: {e}")
        return

    # Pandas lets us access Series values directly by their string index!
    alpha = best_params_series['alpha']
    gamma = best_params_series['gamma']

    data_file = os.path.join(project_root, src.config.MODELING_DATA_SUBDIR, f"pooled_transitions_{animal_id}.pkl")
    save_path = os.path.join(project_root, src.config.STEP2_PRETRAINED_AGENTS_SUBDIR)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Training agent for {animal_id}...")
    print(f"Using Params -> Alpha: {alpha:.4f}, Gamma: {gamma:.2f}")

    agent, env = configure_agent_and_env(data_file, alpha=alpha, gamma=gamma)

    if agent is None:
        print(f"❌ Skipping {animal_id} due to missing data.")
        return

    trained_agent = pooled_training(agent, env, max_epochs=max_epochs)

    output_file = os.path.join(save_path, f"pretrained_agent_{animal_id}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(trained_agent, f)

    print(f"🎉 Agent saved to {output_file}")


def batch_train_all(max_epochs: object = 1) -> None:
    """
    Helper function to train all animals found in the best_params.pkl file at once.
    """
    project_root = src.config.MODELING_PROJECT_ROOT

    params_filepath = os.path.join(project_root, src.config.STEP1_PARAMETER_FITTING_SUBDIR, "best_params.pkl")

    try:
        with open(params_filepath, 'rb') as f:
            global_params_dict = pickle.load(f)

        print(f"🚀 Initiating batch training for {len(global_params_dict)} animals...")
        for animal_id in global_params_dict.keys():
            train_agent_for_animal(animal_id, max_epochs=max_epochs)

    except Exception as e:
        print(f"❌ Error executing batch train: {e}")

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


## --- Examine Values after Integrating Both Event_timer and Time_in_port ---
def plot_value_vs_time_in_port(agent, event_timer_values=[1.0, 1.8, 3.0],
                               port=1, rewards_in_context=0, gambling_disabled=0,
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
                              port=1, rewards_in_context=0, gambling_disabled=0,
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


def plot_value_vs_travel_time(agent, max_travel_time=2.0, step_size=0.1):
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
        state_low = np.array([2.0, t, t, 0.0, 0.0, 1.0])
        state_high = np.array([2.0, t, t, 1.0, 0.0, 1.0])

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


def main():
    # --- 1. SET UP PATHS ---
    project_root = Path(config.MODELING_PROJECT_ROOT)
    params_path = project_root / Path(config.STEP1_PARAMETER_FITTING_SUBDIR) / "best_params.pkl"

    # --- 2. LOAD BEST PARAMETERS ---
    print(f"Reading parameters from {params_path}...")
    try:
        with open(params_path, 'rb') as f:
            # Assuming the pickle contains the dictionary {animal_id: pd.Series}
            best_params_loaded = pickle.load(f)
        print(f"Successfully loaded parameters for {len(best_params_loaded)} animals.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file at {params_path}")
        best_params_loaded = None
    except Exception as e:
        print(f"❌ An error occurred while reading the pickle: {e}")
        best_params_loaded = None

    # --- 3. RUN BATCH TRAINING ---
    if best_params_loaded:
        # Ensure all helper functions (configure_agent_and_env, pooled_training_with_pbar)
        # and batch_train_and_save are defined in your workspace before running this.
        batch_train_and_save(
            best_params=best_params_loaded
        )

    # SZ_animals = ['SZ038', 'SZ039', 'SZ042', 'SZ043']
    # RK_animals = ['RK007', 'RK008']
    # animals = SZ_animals + RK_animals
    # animals=['SZ036']
    # for animal_id in animals:
    #     pooled_data_file = f"../data/pooled_transitions_{animal_id}.pkl"
    #     agent, env = configure_agent_and_env(pooled_data_file, gamma=0.78)
    #     agent = pooled_training(agent, env, max_epochs=1)
    #     plot_value_vs_travel_time(agent, max_travel_time=0.8, step_size=0.05)
        # plot_value_vs_event_timer(agent, time_in_port_values=[1.0, 2.0, 4.0, 8.0],
        #                           port=1, rewards_in_context=0, gambling_disabled=0,
        #                           max_time=5.0, step_size=0.1)
        # plot_value_vs_time_in_port(agent, event_timer_values=[1.0, 1.8, 3.0],
        #                            port=1, rewards_in_context=0, gambling_disabled=0,
        #                            max_time=10.0, step_size=0.1)
        # probe_states_dict = generate_probe_states_dict(examine=False)
        # value_df = calculate_learned_values(agent, probe_states_dict)
        # save_learned_values(value_df, animal_id)
        # plot_learned_values(value_df, animal_id)
        # # --- Save the converged weights ---
        # converged_weights = agent.w.copy()
        # np.save(f"expert_weights_{animal_id}_multi_epoch.npy", converged_weights)
        # print("Saved converged expert weights.")

if __name__ == "__main__":
    main()
    # batch_train_all(max_epochs=1)