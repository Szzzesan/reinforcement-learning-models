import pandas as pd
import os

import pickle
import glob
from pathlib import Path
from src.data_loader import convert_behavior_data_to_state_transitions
import src.config
from src.current_experiment_config import ACTIVE_CONFIG
from src.rl_config import ENV_PARAMS

def pool_animal_transitions(animal_str, env_params, subdir, is_circular=True):
    """
    Finds all processed .parquet files for an animal, loads them,
    converts them to transitions, and pools them together.
    """
    print(f"--- Pooling data for animal: {animal_str} ---")
    pooled_transitions = []

    # 1. Find the directory with the processed files
    animal_dir = os.path.normpath(os.path.join(src.config.MAIN_DATA_ROOT, animal_str))
    processed_dir = os.path.join(animal_dir, subdir)

    # 2. Create a "pattern" to find all .parquet files
    file_pattern = os.path.join(processed_dir, "*_pi_events_processed.parquet")

    # 3. Use glob to get a list of all files matching the pattern
    session_files = sorted(glob.glob(file_pattern))

    if not session_files:
        print(f"⚠️ No .parquet files found for {animal_str} in {processed_dir}")
        return []

    print(f"Found {len(session_files)} sessions to pool.")

    # 4. Loop through the list of files
    for file_path in session_files:
        try:
            df = pd.read_parquet(file_path)

            if df.empty:
                print(f"Skipping empty session file: {os.path.basename(file_path)}")
                continue

            session_transitions = convert_behavior_data_to_state_transitions(df, env_params, is_circular=is_circular)

            pooled_transitions.extend(session_transitions)

        except Exception as e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"⚠️ ERROR: Failed to process file: {os.path.basename(file_path)}")
            print(f"   Error details: {e}")
            print(f"   Skipping this file.")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue

    print(f"--- Finished pooling for {animal_str} ---")
    print(f"Total transitions pooled: {len(pooled_transitions)}")

    return pooled_transitions


def get_session_files(animal_str, subdir):
    """Finds and returns a list of processed .parquet file paths."""
    animal_dir = os.path.normpath(os.path.join(src.config.MAIN_DATA_ROOT, animal_str))
    processed_dir = os.path.join(animal_dir, subdir)

    file_pattern = os.path.join(processed_dir, "*_pi_events_processed.parquet")
    return glob.glob(file_pattern)  # We will sort them globally later


def extract_datetime_from_filename(filepath):
    """
    Extracts the 'YYYY-MM-DD_HH-MM-SS' portion from filenames like:
    'SZ036_2023-10-15_19-18-52_pi_events_processed.parquet'
    This ensures flawless chronological sorting.
    """
    basename = os.path.basename(filepath)
    try:
        parts = basename.split('_')
        # parts[1] is '2023-10-15', parts[2] is '19-18-52'
        return f"{parts[1]}_{parts[2]}"
    except IndexError:
        return basename  # Fallback to standard alphabetical sort if format differs


# def process_sessions_to_transitions(session_dicts, env_params, is_circular=True):
#     """
#     Loads sessions, injects ID/type metadata, converts them, and pools them.
#     Expects session_dicts as: [{'path': filepath, 'type': '...', 'id': 1}, ...]
#     """
#     pooled_transitions = []
#
#     for session in session_dicts:
#         file_path = session['path']
#         session_id = session['id']
#         session_type = session['type']
#
#         try:
#             df = pd.read_parquet(file_path)
#
#             if df.empty:
#                 print(f"Skipping empty session file: {os.path.basename(file_path)}")
#                 continue
#
#             # 1. Add columns to the raw dataframe before conversion
#             df['session_id'] = session_id
#             df['session_type'] = session_type
#
#             # 2. Convert behavior data
#             session_transitions = convert_behavior_data_to_state_transitions(df, env_params, is_circular=is_circular)
#
#             # 3. Safety check: Ensure the metadata carried over into the final product
#             # This guarantees the columns are present regardless of how the converter handles extra df columns
#             if isinstance(session_transitions, pd.DataFrame):
#                 session_transitions['session_id'] = session_id
#                 session_transitions['session_type'] = session_type
#             elif isinstance(session_transitions, list):
#                 for transition in session_transitions:
#                     if isinstance(transition, dict):
#                         transition['session_id'] = session_id
#                         transition['session_type'] = session_type
#
#             pooled_transitions.extend(session_transitions)
#
#         except Exception as e:
#             print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#             print(f"⚠️ ERROR: Failed to process file: {os.path.basename(file_path)}")
#             print(f"   Error details: {e}")
#             print(f"   Skipping this file.")
#             print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#             continue
#
#     return pooled_transitions

def process_sessions_to_transitions(session_dicts, env_params, is_circular=True):
    """
    Loads sessions, converts them to tuples, appends ID/type metadata via an info dict,
    and pools them.
    """
    pooled_transitions = []

    for session in session_dicts:
        file_path = session['path']
        session_id = session['id']
        session_type = session['type']

        try:
            df = pd.read_parquet(file_path)

            if df.empty:
                print(f"Skipping empty session file: {os.path.basename(file_path)}")
                continue

            # 1. Convert behavior data (returns list of 5-element tuples)
            session_transitions = convert_behavior_data_to_state_transitions(df, env_params, is_circular=is_circular)

            # 2. Inject metadata by appending an 'info' dictionary to make a 6-element tuple
            for transition in session_transitions:
                # Unpack the original 5 elements
                obs_t, action_t, reward_t_plus_1, obs_t_plus_1, terminal_flag = transition

                # Create the metadata dictionary
                info = {
                    'session_id': session_id,
                    'session_type': session_type
                }

                # Create the new 6-element tuple
                annotated_transition = (obs_t, action_t, reward_t_plus_1, obs_t_plus_1, terminal_flag, info)

                # Add to our final pool
                pooled_transitions.append(annotated_transition)

        except Exception as e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"⚠️ ERROR: Failed to process file: {os.path.basename(file_path)}")
            print(f"   Error details: {e}")
            print(f"   Skipping this file.")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue

    return pooled_transitions


def save_transitions(transitions, data_dir, filename):
    """Helper function to cleanly save the pickle files."""
    filepath = os.path.join(data_dir, filename)
    print(f"Saving {len(transitions)} transitions to {filepath}...")
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(transitions, file)
        print("List saved successfully.")
    except IOError as e:
        print(f"Error saving file: {e}")
    except pickle.PicklingError as e:
        print(f"Error pickling object: {e}")

def main():
    env_params = ENV_PARAMS

    # --- Path Setup ---
    data_dir = os.path.join(src.config.MODELING_PROJECT_ROOT, src.config.MODELING_DATA_SUBDIR)
    os.makedirs(data_dir, exist_ok=True)
    # ------------------

    SZ_animals = ['SZ036', 'SZ037', 'SZ038', 'SZ039', 'SZ042', 'SZ043']
    RK_animals = ['RK007', 'RK008']
    animals_to_pool = SZ_animals + RK_animals
    # animals_to_pool = ["RK007"]

    split_method = ACTIVE_CONFIG["split_method"]
    print(f"\n{'=' * 50}")
    print(f"Executing pooling pipeline")
    print(f"Split Method: {split_method}")
    print(f"Output Directory: {data_dir}")
    print(f"{'=' * 50}\n")

    for animal in animals_to_pool:
        print(f"\n--- Pooling data for animal: {animal} ---")

        # 1. Gather all file paths without sorting them yet
        pre_surgery_paths = get_session_files(animal, src.config.PRETRAINING_PROCESSED_DATA_SUBDIR)
        post_surgery_paths = get_session_files(animal, src.config.PROCESSED_DATA_SUBDIR)

        # 2. Wrap them in dictionaries to keep track of their source type
        all_sessions = []
        all_sessions.extend([{'path': p, 'type': 'pre-surgery'} for p in pre_surgery_paths])
        all_sessions.extend([{'path': p, 'type': 'post-surgery'} for p in post_surgery_paths])

        if not all_sessions:
            print(f"⚠️ No files found for {animal}. Skipping.")
            continue

        # 3. Sort ALL sessions chronologically by the datetime embedded in the filename
        all_sessions.sort(key=lambda x: extract_datetime_from_filename(x['path']))

        # 4. Assign continuous session_ids starting from 1
        for i, session in enumerate(all_sessions, start=1):
            session['id'] = i

        # 5. Re-separate them to apply the configuration split logic
        annotated_pre = [s for s in all_sessions if s['type'] == 'pre-surgery']
        annotated_post = [s for s in all_sessions if s['type'] == 'post-surgery']

        # 6. Partition based on the active experiment configuration
        if split_method == "presurgery_only":
            train_sessions = annotated_pre
            target_sessions = annotated_post

        elif split_method == "mixed_20_percent":
            split_idx = int(len(annotated_post) * 0.2)
            train_sessions = annotated_pre + annotated_post[:split_idx]
            target_sessions = annotated_post[split_idx:]

        else:
            raise ValueError(f"Unknown split_method configured: {split_method}")

        print(f"Found {len(train_sessions)} files for training/base transitions.")
        print(f"Found {len(target_sessions)} files for target transitions.")

        # 7. Process and save the training/base transitions
        if train_sessions:
            train_transitions = process_sessions_to_transitions(train_sessions, env_params, is_circular=True)
            save_transitions(train_transitions, data_dir, f"pooled_transitions_{animal}.pkl")

        # 8. Process and save the target transitions
        if target_sessions:
            target_transitions = process_sessions_to_transitions(target_sessions, env_params, is_circular=True)
            save_transitions(target_transitions, data_dir,
                             f"target_sessions_pooled_transitions_{animal}.pkl")

        print(f"--- Finished pooling for {animal} ---")

if __name__ == "__main__":
    main()