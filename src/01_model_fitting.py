import pickle
import os
import json
import numpy as np
import pandas as pd
# from scipy._lib.array_api_compat import torch
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from src.mouse_playback_agent import MousePlaybackAgent
from src.data_loader import load_pooled_transitions
import src.config as config

from src.rl_config import AGENT_INFO_TEMPLATE
from src.state_utils import build_travel_state, build_investment_sim_state, is_investment_state

random.seed(43)
np.random.seed(43)

def validate_leave_time_error_visual(transitions, agent_info, test_alphas=[0.001, 0.02, 0.1], num_trials=10):
    """
    Plots the same trials under different Alphas to verify the MSE
    of the Monte Carlo predicted leave times.
    """
    # 1. Identify valid GAMBLING trials (Needs to be port 1, enabled, and context rewards = 4)
    trial_map = []
    curr_start = -1

    for i in range(len(transitions)):
        obs, _, _, obs_next, term, session_info = transitions[i]

        # Ensures we are specifically visualizing gambling trials
        is_gambling = is_investment_state(obs)

        if is_gambling and curr_start == -1:
            curr_start = i

        if curr_start != -1 and (obs_next[0] != 1.0 or term):
            trial_map.append((curr_start, i))
            curr_start = -1

    if not trial_map:
        print("No valid gambling trials found.")
        return

    selected_trials = sorted(random.sample(trial_map, min(num_trials, len(trial_map))))

    # Storage structure: [ {alpha1: data, alpha2: data}, ... ]
    plot_data_store = [{} for _ in range(len(selected_trials))]

    # 2. Setup Plot
    fig, axes = plt.subplots(len(selected_trials), len(test_alphas),
                             figsize=(12, 12 * 0.6 * len(selected_trials) / len(test_alphas)), constrained_layout=True)

    if len(selected_trials) == 1: axes = np.array([axes])
    if len(test_alphas) == 1: axes = axes.reshape(-1, 1)
    if len(selected_trials) == 1 and len(test_alphas) > 1:
        axes = axes.reshape(1, -1)
    elif len(selected_trials) > 1 and len(test_alphas) == 1:
        axes = axes.reshape(-1, 1)

    opportunity_cost = 0

    # 3. Run Agent
    for col_idx, alpha in enumerate(test_alphas):
        info = agent_info.copy()
        info['step_size'] = alpha
        agent = MousePlaybackAgent()
        agent.agent_init(info)
        # from pathlib import Path
        # import src.config as config
        # project_root = Path(config.MODELING_PROJECT_ROOT)
        # agent_file = project_root / "outputs" / "2_trained_agents" / f"pretrained_agent_SZ036.pkl"
        #
        # with open(agent_file, 'rb') as f:
        #     agent = pickle.load(f)

        agent.step_size = alpha

        agent.agent_start(transitions[0][0])

        for i in range(len(transitions)):
            obs, _, reward, obs_next, term, session_info = transitions[i]

            # Check if we are inside a selected trial
            active_trial_idx = -1
            for idx, (start, end) in enumerate(selected_trials):
                if start <= i <= end:
                    active_trial_idx = idx
                    break

            if active_trial_idx != -1:
                v_stay = agent.get_value(obs)
                # obs_l = obs.copy()
                # obs_l[0] = 2 # port id: traveling
                # obs_l[1] = 0.0 # time in port
                # obs_l[2] = 0.0 # event timer
                # obs_l[5] = 1 # investment port diabled
                obs_l = build_travel_state(obs)
                v_leave = agent.get_value(obs_l) + opportunity_cost

                if alpha not in plot_data_store[active_trial_idx]:
                    # Added 'timers' and 'context' to store the extra MC inputs
                    plot_data_store[active_trial_idx][alpha] = {'stay': [], 'leave': [], 'times': [], 'timers': [],
                                                                'context': obs[3]}

                plot_data_store[active_trial_idx][alpha]['stay'].append(v_stay)
                plot_data_store[active_trial_idx][alpha]['leave'].append(v_leave)
                plot_data_store[active_trial_idx][alpha]['times'].append(obs[1])
                plot_data_store[active_trial_idx][alpha]['timers'].append(obs[2])

                # Plot at end of trial
                if i == selected_trials[active_trial_idx][1]:
                    ax = axes[active_trial_idx][col_idx]
                    data = plot_data_store[active_trial_idx][alpha]

                    # Get the value threshold (the value immediately upon leaving the port)
                    # v_after = agent.get_value(obs_next)
                    v_after = data['leave'][-1]

                    # --- NEW: MSE Calculation via MC Predictor ---
                    trial_data = {
                        'times': data['times'],
                        'values': data['stay'],
                        'event_timer': data['timers'],
                        'context': data['context'],
                        'v_after': v_after
                    }

                    actual_time = data['times'][-1]
                    # Bumping num_sims a bit for smoother visualization
                    pred_time = quick_mc_predict(trial_data, agent, num_sims=50)

                    mse_loss = (pred_time - actual_time) ** 2

                    loss_str = f"MSE: {mse_loss:.2f}"
                    title_color = "black"

                    if mse_loss < 9:
                        title_color = "green"
                    elif mse_loss > 25:
                        title_color = "red"

                    # --- PLOTTING ---
                    ax.plot(data['times'], data['stay'], color='blue', label='V_stay')
                    ax.plot(data['times'], data['leave'], color='orange', linestyle='--', alpha=0.4,
                            label='V_leave')

                    # ax.axhline(v_after, color='gray', alpha=0.5, linestyle=':', label='Threshold')
                    ax.text(0.02, v_after, f" V = {v_after:.3f}", color='darkorange',
                            va='bottom', ha='left', fontsize=8, fontweight='bold',
                            transform=ax.get_yaxis_transform())
                    ax.axvline(actual_time, color='black', alpha=0.8, linestyle='-',
                               label=f'Actual ({actual_time:.1f}s)')
                    ax.axvline(pred_time, color='red', alpha=0.8, linestyle='--', label=f'Pred ({pred_time:.1f}s)')

                    ax.set_title(f"Trial {active_trial_idx + 1} | α={alpha}\n{loss_str}",
                                 color=title_color,
                                 fontweight='bold',
                                 fontsize='small')

                    if active_trial_idx == 0 and col_idx == 0:
                        ax.legend(fontsize='x-small', loc='upper right')
                    if active_trial_idx == len(selected_trials) - 1:
                        ax.set_xlabel("Time (s)")

            # Standard learning step
            if term:
                agent.agent_end(reward)
                if i + 1 < len(transitions): agent.agent_start(transitions[i + 1][0])
            else:
                agent.agent_step(reward, obs_next)

    # --- Save Figure ---
    # save_dir = os.path.join(config.MAIN_DATA_ROOT, config.THESIS_FIGURE_SUBDIR)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # save_path = os.path.join(save_dir, "fig_5_alpha_ranges_mse.png")
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # print(f"Composite figure successfully saved to: {save_path}")

    plt.show()


# --- Helper: Reward Probability ---
def exp_decreasing_prob(t, cumulative=8., starting=1.):
    """Returns the exact probability of reward for a 0.1s bin."""
    a = starting
    b = a / cumulative
    rate = a / np.exp(b * t)
    return min(1.0, max(0.0, rate / 10.0))  # Divide by 10 to convert rate/sec to prob/0.1s


# --- Helper: Lightweight On-the-Fly MC Predictor ---
def quick_mc_predict(trial_data, agent, dt=0.1, max_extrap=20.0, num_sims=5):
    """A faster MC predictor optimized for grid search."""
    times = trial_data['times']
    values = trial_data['values']
    event_timers = trial_data['event_timer']
    threshold = trial_data['v_after']

    # 1. Did it already cross?
    for i, v in enumerate(values):
        if v <= threshold:
            return times[i]

    # 2. Extrapolate
    actual_leave_time = times[-1]
    actual_last_timer = event_timers[-1]
    context = trial_data['context']

    sim_leaves = []

    for _ in range(num_sims):  # Fewer sims (20) to keep grid search fast
        c_time = actual_leave_time
        c_timer = actual_last_timer

        while c_time < (actual_leave_time + max_extrap):
            c_time += dt
            if random.random() < exp_decreasing_prob(c_time):
                c_timer = 0.0
            else:
                c_timer += dt

            # sim_obs = np.array([1.0, c_time, c_timer, context, 4, 0.0])
            sim_obs = build_investment_sim_state(c_time, c_timer, context)
            if agent.get_value(sim_obs) <= threshold:
                sim_leaves.append(c_time)
                break
        else:
            sim_leaves.append(c_time)

    return sum(sim_leaves) / len(sim_leaves)


# --- THE NEW OPTIMIZATION METRIC ---
def calculate_leave_time_error(params, transitions, num_sessions=10):
    """
    Trains the agent and evaluates Mean Squared Error (MSE)
    between Actual and Predicted leave times in the final N sessions.
    """
    alpha, gamma = params

    # Init Agent
    info = AGENT_INFO_TEMPLATE.copy()
    info['step_size'] = alpha
    info['discount'] = gamma

    # ASSUMING MousePlaybackAgent is imported/available in your script
    agent = MousePlaybackAgent()
    agent.agent_init(info)

    # --- Find the start index for the last 'num_sessions' ---
    terminal_indices = [idx for idx, t in enumerate(transitions) if t[4] is True]
    total_sessions = len(terminal_indices)

    if total_sessions <= num_sessions:
        start_index = 0
    else:
        cut_off_session_idx = total_sessions - num_sessions - 1
        start_index = terminal_indices[cut_off_session_idx] + 1

    total_mse = 0.0
    trial_count = 0

    in_gambling_trial = False
    trial_times = []
    trial_vs = []
    trial_timers = []
    trial_context = None

    agent.agent_start(transitions[0][0])

    for i in range(len(transitions)):
        obs_t, _, reward, obs_next, terminal = transitions[i]

        v_current = agent.get_value(obs_t)

        is_gambling = is_investment_state(obs_t)

        # Only evaluate predictions in the target (last N) sessions
        if i >= start_index:
            if is_gambling:
                if not in_gambling_trial:
                    in_gambling_trial = True
                    trial_times = []
                    trial_vs = []
                    trial_timers = []
                    trial_context = obs_t[3]

                trial_vs.append(v_current)
                trial_times.append(obs_t[1])
                trial_timers.append(obs_t[2])

                last_gambling_obs = obs_t

            else:
                if in_gambling_trial:

                    if len(trial_times) > 0:
                        theoretical_1st_travel_state = build_travel_state(last_gambling_obs)
                        v_after = agent.get_value(theoretical_1st_travel_state)

                        trial_data = {
                            'times': trial_times,
                            'values': trial_vs,
                            'event_timer': trial_timers,
                            'context': trial_context,
                            'v_after': v_after
                        }

                        # Predict and calculate error
                        actual_time = trial_times[-1]
                        pred_time = quick_mc_predict(trial_data, agent, num_sims=20)

                        # Mean Squared Error
                        total_mse += (pred_time - actual_time) ** 2
                        trial_count += 1

                    in_gambling_trial = False

        # Standard Learning Step (Agent updates its weights)
        if terminal:
            agent.agent_end(reward)
            if i + 1 < len(transitions):
                agent.agent_start(transitions[i + 1][0])
        else:
            agent.agent_step(reward, obs_next)

    # Return average MSE. Return a high penalty if no trials were found.
    return total_mse / trial_count if trial_count > 0 else 9999.0


def calculate_leave_time_error_postsurg_only(params, transitions):
    """
    Trains the agent and evaluates Mean Squared Error (MSE)
    between Actual and Predicted leave times in 'post-surgery' sessions.
    """
    alpha, gamma = params

    # Init Agent
    info = AGENT_INFO_TEMPLATE.copy()
    info['step_size'] = alpha
    info['discount'] = gamma

    # ASSUMING MousePlaybackAgent is imported/available in your script
    agent = MousePlaybackAgent()
    agent.agent_init(info)

    total_mse = 0.0
    trial_count = 0

    in_gambling_trial = False
    trial_times = []
    trial_vs = []
    trial_timers = []
    trial_context = None

    agent.agent_start(transitions[0][0])

    for i in range(len(transitions)):
        # Unpack the 6 elements, including our new transition_info dictionary
        obs_t, _, reward, obs_next, terminal, transition_info = transitions[i]

        v_current = agent.get_value(obs_t)
        is_gambling = is_investment_state(obs_t)

        # Only evaluate predictions if this transition belongs to a post-surgery session
        if transition_info['session_type'] == 'post-surgery':
            if is_gambling:
                if not in_gambling_trial:
                    in_gambling_trial = True
                    trial_times = []
                    trial_vs = []
                    trial_timers = []
                    trial_context = obs_t[3]

                trial_vs.append(v_current)
                trial_times.append(obs_t[1])
                trial_timers.append(obs_t[2])

                last_gambling_obs = obs_t

            else:
                if in_gambling_trial:
                    if len(trial_times) > 0:
                        theoretical_1st_travel_state = build_travel_state(last_gambling_obs)
                        v_after = agent.get_value(theoretical_1st_travel_state)

                        trial_data = {
                            'times': trial_times,
                            'values': trial_vs,
                            'event_timer': trial_timers,
                            'context': trial_context,
                            'v_after': v_after
                        }

                        # Predict and calculate error
                        actual_time = trial_times[-1]
                        pred_time = quick_mc_predict(trial_data, agent, num_sims=20)

                        # Mean Squared Error
                        total_mse += (pred_time - actual_time) ** 2
                        trial_count += 1

                    in_gambling_trial = False
        else:
            # Ensure the tracking variables stay clean during pre-surgery phases
            in_gambling_trial = False

        # Standard Learning Step (Agent updates its weights on ALL transitions)
        if terminal:
            agent.agent_end(reward)
            if i + 1 < len(transitions):
                agent.agent_start(transitions[i + 1][0])
        else:
            agent.agent_step(reward, obs_next)

    # Return average MSE. Return a high penalty if no trials were found.
    return total_mse / trial_count if trial_count > 0 else 9999.0


# def get_next_alphas(best_a, prev_alphas, min_alpha=0.0001):
#     """Calculates the next alpha grid by shrinking the step size by ~1/3."""
#     prev_alphas = sorted(list(set(prev_alphas)))
#     idx = prev_alphas.index(best_a)
#
#     # 1. Find the distance 'd' to the nearest neighbor
#     if len(prev_alphas) >= 2:
#         if idx == 0:
#             d = prev_alphas[1] - prev_alphas[0]
#         elif idx == len(prev_alphas) - 1:
#             d = prev_alphas[-1] - prev_alphas[-2]
#         else:
#             d = min(best_a - prev_alphas[idx-1], prev_alphas[idx+1] - best_a)
#     else:
#         d = best_a * 0.5  # Fallback
#
#     # 2. Shrink step size to 1/3 but ensure it never goes below 0.001
#     new_step = d/3
#     # new_step = max(0.001, round(d / 3.0, 2))
#
#     # 3. Determine the Absolute Minimum Step Size based on Significant Digits
#     # Convert float to a clean string, avoiding e-notation and trailing zeros
#     s = f"{best_a:.10f}".rstrip('0')
#     if s.endswith('.'):
#         s = s[:-1]
#
#     if '.' in s:
#         decimals = s.split('.')[1]
#         last_digit_pos = len(decimals)
#
#         # Extract the actual non-zero digits as a string
#         non_zeros = s.replace('.', '').replace('0', '')
#
#         # THE FIX: Check if the number is exactly a power of 10 (e.g., '1', '1', '1')
#         if len(non_zeros) == 1 and non_zeros == '1' and last_digit_pos < 4:
#             # EXCEPTION RULE: Only for 0.1, 0.01, 0.001
#             # The minimum step size drops one decimal place lower
#             power = last_digit_pos + 1
#         else:
#             # STANDARD RULE: If it's 0.015, the minimum step matches the last digit (3 decimal places)
#             power = last_digit_pos
#     else:
#         # Fallback if it's somehow a whole number (e.g. 1.0)
#         power = 1
#
#         # Calculate the required threshold (e.g. power=3 becomes 0.001)
#     required_min_step = round(10 ** -power, power)
#
#     # 4. Enforce the constraint and round neatly
#     final_step = max(new_step, required_min_step)
#     final_step = round(final_step, power)
#
#     # 5. Create the new grid
#     if best_a <= min_alpha:
#         # If we hit the absolute floor, expand strictly upwards (e.g., 0.0001, 0.0004, 0.0007)
#         next_grid = [min_alpha, min_alpha + final_step, min_alpha + 2*final_step]
#     else:
#         # Center around the best value (e.g., 0.0006, 0.0007, 0.0008)
#         next_grid = [best_a - final_step, best_a, best_a + final_step]
#
#     return sorted(list(set([max(min_alpha, round(x, power)) for x in next_grid])))

def get_next_alphas(best_a, prev_alphas, min_alpha=0.0001, max_alpha=0.01):
    """Calculates the next alpha grid by shrinking the step size by ~1/3, keeping it within bounds."""
    prev_alphas = sorted(list(set(prev_alphas)))
    idx = prev_alphas.index(best_a)

    # 1. Find the distance 'd' to the nearest neighbor
    if len(prev_alphas) >= 2:
        if idx == 0:
            d = prev_alphas[1] - prev_alphas[0]
        elif idx == len(prev_alphas) - 1:
            d = prev_alphas[-1] - prev_alphas[-2]
        else:
            d = min(best_a - prev_alphas[idx - 1], prev_alphas[idx + 1] - best_a)
    else:
        d = best_a * 0.5  # Fallback

    # 2. Shrink step size to 1/3
    new_step = d / 3.0

    # 3. Determine the Absolute Minimum Step Size based on Significant Digits
    s = f"{best_a:.10f}".rstrip('0')
    if s.endswith('.'):
        s = s[:-1]

    if '.' in s:
        decimals = s.split('.')[1]
        last_digit_pos = len(decimals)
        non_zeros = s.replace('.', '').replace('0', '')

        # THE FIX: Check if the number is exactly a power of 10
        if len(non_zeros) == 1 and non_zeros == '1' and last_digit_pos < 4:
            power = last_digit_pos + 1
        else:
            power = last_digit_pos
    else:
        power = 1

    required_min_step = round(10 ** -power, power)

    # 4. Enforce the constraint and round neatly
    final_step = max(new_step, required_min_step)
    final_step = round(final_step, power)

    # 5. Create the new grid respecting BOTH boundaries
    if best_a <= min_alpha:
        # If we hit the absolute floor, expand strictly upwards
        next_grid = [min_alpha, min_alpha + final_step, min_alpha + 2 * final_step]
    elif best_a >= max_alpha:
        # If we hit the absolute ceiling, expand strictly downwards
        next_grid = [max_alpha - 2 * final_step, max_alpha - final_step, max_alpha]
    else:
        # Center around the best value
        next_grid = [best_a - final_step, best_a, best_a + final_step]

    # 6. Final safety clamp to guarantee no floating point math pushes values out of bounds
    clipped_grid = [min(max_alpha, max(min_alpha, round(x, power))) for x in next_grid]

    return sorted(list(set(clipped_grid)))

def get_next_gammas(best_g, prev_gammas):
    """Calculates the next gamma grid, rounded to 2 decimal places, bounded [0.5, 1.0]."""
    prev_gammas = sorted(list(set(prev_gammas)))
    idx = prev_gammas.index(best_g)

    if len(prev_gammas) >= 2:
        if idx == 0:
            d = prev_gammas[1] - prev_gammas[0]
        elif idx == len(prev_gammas) - 1:
            d = prev_gammas[-1] - prev_gammas[-2]
        else:
            d = min(best_g - prev_gammas[idx-1], prev_gammas[idx+1] - best_g)
    else:
        d = 0.1

    # Shrink step size, but ensure it never goes below 0.01
    new_step = max(0.01, round(d / 3.0, 2))

    next_grid = [best_g - new_step, best_g, best_g + new_step]

    # Clamp between 0.5 and 1.0, and force 2 decimal places
    final_grid = []
    for g in next_grid:
        g_clamped = max(0.5, min(1.0, round(g, 2)))
        if g_clamped not in final_grid:
            final_grid.append(g_clamped)

    return sorted(final_grid)


def save_grid_results(results, filename_base):
    """
    Saves grid search results to both CSV and JSON formats.
    """
    # 1. Create the results directory if it doesn't exist
    # This goes up one level from /notebooks to the root, then into /results
    project_root = config.MODELING_PROJECT_ROOT
    save_path = os.path.join(project_root, config.STEP1_PARAMETER_FITTING_SUBDIR)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # 2. Convert to DataFrame for CSV saving
    df = pd.DataFrame(results)
    csv_file = os.path.join(save_path, f"{filename_base}.csv")
    df.to_csv(csv_file, index=False)

    # 3. Save as JSON for easy reloading in Python later
    json_file = Path(os.path.join(save_path, f"{filename_base}.json"))
    with open(json_file, 'w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=4)

    print(f"✅ Results saved successfully!")
    print(f"CSV: {csv_file}")
    print(f"JSON: {json_file}")


def save_best_params(data, filename_base):
    project_root = config.MODELING_PROJECT_ROOT
    filepath = os.path.join(project_root, config.STEP1_PARAMETER_FITTING_SUBDIR, f"{filename_base}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    # Convert dictionary of Series to a single DataFrame
    df = pd.DataFrame(data)
    # Save to CSV
    filepath = os.path.join(project_root, config.STEP1_PARAMETER_FITTING_SUBDIR, f"{filename_base}.csv")
    df.to_csv(filepath, index=True)


def process_animal(animal_id, parameter_dict):
    # --- PATH SETUP ---
    PROJECT_ROOT = config.MODELING_PROJECT_ROOT
    DATA_FOLDER = os.path.join(PROJECT_ROOT, config.MODELING_DATA_SUBDIR)
    RESULTS_FOLDER = os.path.join(PROJECT_ROOT, config.STEP1_PARAMETER_FITTING_SUBDIR)
    Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)

    all_transitions = load_pooled_transitions(DATA_FOLDER, animal_id)
    alphas = parameter_dict['alphas']
    gammas = parameter_dict['gammas']

    combinations = list(product(alphas, gammas))
    grid_results = []

    print("Starting Grid Search...")
    with tqdm(total=len(combinations), desc=f"🔍 {animal_id} Grid", leave=False, unit="pair") as pbar:
        for a, g in combinations:

            # nvg = calculate_normalized_value_gap([a, g], all_transitions)
            mse_error = calculate_leave_time_error_postsurg_only([a, g], all_transitions)

            print(f"Alpha: {a:.4f}, Gamma: {g:.2f} | Avg MSE: {mse_error:.4f}")
            grid_results.append({'alpha': a, 'gamma': g, 'mse': mse_error})

            # Update the progress bar to show which combination just finished and its MSE
            pbar.set_postfix(a=f"{a:.5f}", g=f"{g:.2f}", mse=f"{mse_error:.4f}")
            pbar.update(1)

    # 2. Find the best pair
    grid_df = pd.DataFrame(grid_results)
    best_row = grid_df.loc[grid_df['mse'].idxmin()]

    print("\n--- ROUND WINNER ---")
    print(f"Best Alpha: {best_row['alpha']}")
    print(f"Best Gamma: {best_row['gamma']}")
    print(f"Lowest MSE: {best_row['mse']}")
    return best_row, grid_df


def visualize_random_trials(transitions, agent_info, best_params, num_samples=10, opportunity_cost=0.0):
    alpha, gamma, sigma = best_params

    agent = MousePlaybackAgent()
    agent.agent_init(agent_info)
    agent.step_size = alpha
    agent.discount = gamma

    # 1. Identify valid 'Enabled' trials across the whole session
    # A trial is valid if it happens in Port 1 AND index 5 is 0.0 (Enabled)
    valid_trial_starts = []
    is_currently_valid = False

    for i in range(len(transitions)):
        obs_t, _, _, obs_next, terminal = transitions[i]

        in_port_1 = (obs_t[0] == 1.0)
        is_enabled = (obs_t[5] == 0.0)

        # We only care about the moments they are actually in the port while enabled
        if in_port_1 and is_enabled:
            if not is_currently_valid:
                valid_trial_starts.append(i)
                is_currently_valid = True

        # End of investment phase
        if obs_next[0] != 1.0 or terminal:
            is_currently_valid = False

    # 2. Randomly sample from the 'Enabled' subset
    if not valid_trial_starts:
        print("❌ No enabled trials found in this dataset!")
        return

    num_samples = min(num_samples, len(valid_trial_starts))
    sample_starts = set(random.sample(valid_trial_starts, num_samples))

    # Storage for plot data
    plot_data = []
    current_trial_v_stay = []
    current_trial_v_leave = []
    recording_this_trial = False

    agent.agent_start(transitions[0][0])

    # 3. Process the full session so the agent's 'brain' is always up to date
    for i in range(len(transitions)):
        obs_t, _, reward, obs_next, terminal = transitions[i]

        # Start recording if this index was in our random sample
        if i in sample_starts:
            recording_this_trial = True

        in_port_1 = (obs_t[0] == 1.0)
        is_enabled = (obs_t[5] == 0.0)

        if recording_this_trial and in_port_1 and is_enabled:
            v_s = agent.get_value(obs_t)
            # Synthetic 'Leave' observation
            obs_l = obs_t.copy()
            obs_l[0] = 2.0
            obs_l[1] = 0.0
            v_l = agent.get_value(obs_l) + opportunity_cost

            current_trial_v_stay.append(v_s)
            current_trial_v_leave.append(v_l)

        # Detect the exit to save the data
        if recording_this_trial and (obs_next[0] != 1.0 or terminal):
            plot_data.append({
                'trial_index': i,
                'v_stay': current_trial_v_stay,
                'v_leave': current_trial_v_leave,
                'actual_exit': obs_t[1]
            })
            current_trial_v_stay, current_trial_v_leave = [], []
            recording_this_trial = False

        # --- Standard TD Learning (Always happens) ---
        if terminal:
            agent.agent_end(reward)
            if i + 1 < len(transitions): agent.agent_start(transitions[i + 1][0])
        else:
            agent.agent_step(reward, obs_next)

    # --- PLOTTING ---
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples), sharex=False)
    if num_samples == 1: axes = [axes]

    for idx, data in enumerate(plot_data):
        ax = axes[idx]
        time_axis = [t * 0.1 for t in range(len(data['v_stay']))]

        ax.plot(time_axis, data['v_stay'], label="V_stay (Investment)", color='forestgreen', linewidth=2.5)
        ax.plot(time_axis, data['v_leave'], label="V_leave (Travel + Offset)", color='crimson', linestyle='--',
                linewidth=2)

        # Mouse's actual decision
        ax.axvline(x=data['actual_exit'], color='black',
                   label=f"Mouse Left at {data['actual_exit']:.2f}s")

        # Agent's predicted decision (crossover)
        pred_exit_idx = next((t for t, (s, l) in enumerate(zip(data['v_stay'], data['v_leave'])) if l > s), None)
        if pred_exit_idx is not None:
            pred_time = pred_exit_idx * 0.1
        else:
            pred_time = 10
        ax.axvline(x=pred_time, color='black', linestyle=':',
                   label=f"Agent Predicts {pred_time:.2f}s")

        ax.set_title(f"Random Enabled Trial (Global Index: {data['trial_index']})")
        ax.set_ylabel("Value (V)")
        ax.legend(loc='upper right', frameon=True, fontsize='small')

    plt.xlabel("Time in Investment Port (seconds)")
    plt.tight_layout()
    plt.show()

def verify_exit_states(animal_id="SZ036", num_trials=15):
    """
    Compares the theoretical build_travel_state() against the empirical next_obs
    for the first X VALID trials of an animal.
    """
    project_root = Path(config.MODELING_PROJECT_ROOT)
    data_dir = project_root / "data"

    # Load the transitions
    transitions = load_pooled_transitions(data_dir, animal_id, type='target')

    if transitions is None:
        print("❌ Could not load transitions.")
        return

    trials_found = 0

    print(f"\n🔍 Verifying Exit States for {animal_id}")
    print("State Vector: [Port, Time_in_Port, Event_Timer, Context, Rewards_in_Ctx, Port_Disabled]")
    print("-" * 80)

    for i in range(len(transitions)):
        obs, action, reward, next_obs, term = transitions[i]

        # --- Exit Logic ---
        is_valid_gambling_state = (obs[0] == 1.0 and obs[5] == 0)
        just_left_port = (next_obs[0] != 1.0)

        # Only trigger if the mouse leaves an ENABLED gambling port
        if is_valid_gambling_state and just_left_port:

            # 1. Build the theoretical state using our centralized function
            theoretical_state = build_travel_state(obs)

            # 2. Grab the actual state the mouse landed in
            actual_state = next_obs

            # Print them clearly
            print(f"Trial {trials_found + 1}")
            print(f"   Last Gambling State: {obs}")
            print(f"   Theoretical Exit   : {theoretical_state}")
            print(f"   Actual Exit State  : {actual_state}")

            # Check for match (using a tiny tolerance for floating point math)
            is_match = all(abs(t - a) < 1e-5 for t, a in zip(theoretical_state, actual_state))

            if is_match:
                print("   ✅ MATCH")
            else:
                print("   ❌ MISMATCH! Check the vectors above to see which index differs.")

            print("-" * 80)

            trials_found += 1
            if trials_found >= num_trials:
                break

def main():
    # --- PATH SETUP ---
    animal_list = ["SZ036", "SZ037", "SZ038", "SZ039", "SZ042", "SZ043", "RK007", "RK008"]
    # animal_list = ["SZ036"]
    # Define the Grid
    # alphas = [0.0005]
    # gammas = [0.55, 0.65, 0.75, 0.85, 0.95]
    # parameter_dict = {'alphas': alphas, 'gammas': gammas}
    # best_params = {}
    MAX_ROUNDS = 4

    main_pbar = tqdm(animal_list, desc="🧬 Total Cohort Progress", unit="animal")
    for animal in main_pbar:
        main_pbar.set_postfix(current_animal=animal)
        try:
            # --- ROUND 1 (Coarse Map) ---
            current_alphas = [0.0001, 0.001, 0.01]
            current_gammas = [0.5, 0.7, 0.9]


            for round_num in range(1, MAX_ROUNDS + 1):

                print(f"\n🚀 --- {animal} | COARSE-TO-FINE ROUND {round_num} ---")
                print(f"Testing Alphas: {current_alphas}")
                print(f"Testing Gammas: {current_gammas}")

                parameter_dict = {'alphas': current_alphas, 'gammas': current_gammas}

                # 1. Run the search
                best_row, grid_results = process_animal(animal, parameter_dict)

                # 2. Save results uniquely for this round
                save_filename = f"mse_search_{animal}_round{round_num}"
                save_grid_results(grid_results, save_filename)

                best_a = best_row['alpha']
                best_g = best_row['gamma']

                # 3. Calculate grids for the NEXT round
                next_alphas = get_next_alphas(best_a, current_alphas)
                next_gammas = get_next_gammas(best_g, current_gammas)

                # 4. Convergence Check
                # If the grid can't shrink anymore (due to our decimal limits), we are done!
                if next_alphas == current_alphas and next_gammas == current_gammas:
                    print(f"✅ Search converged for {animal} at Round {round_num}!")
                    print(f"🏆 Ultimate Best -> Alpha: {best_a}, Gamma: {best_g}")
                    break

                # Otherwise, prepare for the next round
                current_alphas = next_alphas
                current_gammas = next_gammas

        except Exception as e:
            tqdm.write(f"❌ Error processing {animal}: {e}")


# --- MAIN BLOCK ---
if __name__ == "__main__":
    # main()

    # --- QUICK VALIDATION ---
    # PROJECT_ROOT = config.MODELING_PROJECT_ROOT
    # DATA_FOLDER = os.path.join(PROJECT_ROOT, config.MODELING_DATA_SUBDIR)
    #
    # animal = "SZ036"
    # transitions = load_pooled_transitions(DATA_FOLDER, animal)
    #
    # agent_info = AGENT_INFO_TEMPLATE.copy()
    # agent_info["discount"] = 0.9
    # validate_leave_time_error_visual(transitions, agent_info, test_alphas=[0.001, 0.006], num_trials=10)

    # --- TEMPORARY: SAVE THE BEST PARAMETERS ---
    best_params = {
        "SZ036": pd.Series([0.006, 0.9, 5.63], index=["alpha", "gamma", "mse"]),  # alpha, gamma, nvg^2
        "SZ037": pd.Series([0.0012, 0.72, 9.89], index=["alpha", "gamma", "mse"]),
        "SZ038": pd.Series([0.0002, 0.51, 9.16], index=["alpha", "gamma", "mse"]),
        "SZ039": pd.Series([0.003, 0.89, 6.71], index=["alpha", "gamma", "mse"]),
        "SZ042": pd.Series([0.0009, 0.64, 10.02], index=["alpha", "gamma", "mse"]),
        "SZ043": pd.Series([0.0009, 0.72, 9.06], index=["alpha", "gamma", "mse"]),
        "RK007": pd.Series([0.01, 0.8, 15.70], index=["alpha", "gamma", "mse"]),
        "RK008": pd.Series([0.006, 0.81, 11.85], index=["alpha", "gamma", "mse"]),
    }
    save_best_params(best_params, f"best_params")

    # verify_exit_states(animal_id="SZ036", num_trials=15)