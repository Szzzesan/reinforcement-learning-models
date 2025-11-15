import pandas as pd
import os
import config
import pickle
import glob
from data_loader import convert_behavior_data_to_state_transitions

def pool_animal_transitions(animal_str, env_params, subdir):
    """
    Finds all processed .parquet files for an animal, loads them,
    converts them to transitions, and pools them together.
    """
    print(f"--- Pooling data for animal: {animal_str} ---")
    pooled_transitions = []

    # 1. Find the directory with the processed files
    animal_dir = os.path.normpath(os.path.join(config.MAIN_DATA_ROOT, animal_str))
    processed_dir = os.path.join(animal_dir, subdir)

    # 2. Create a "pattern" to find all .parquet files
    if subdir == config.PRETRAINING_PROCESSED_DATA_SUBDIR:
        file_pattern = os.path.join(processed_dir, "*_pi_events_proccessed.parquet")
    elif subdir == config.PROCESSED_DATA_SUBDIR:
        file_pattern = os.path.join(processed_dir, "*_pi_events_processed.parquet")
    # apologies for the typo lol that's just how I saved my own files

    # 3. Use glob to get a list of all files matching the pattern
    session_files = glob.glob(file_pattern)

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

            session_transitions = convert_behavior_data_to_state_transitions(df, env_params)

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

def main():
    # configuration
    env_params = {  # Params needed by the data loader
        "time_step_duration": 0.1,
        "session_duration_min": 18,
        "context_rewards_max": 4,
        "block_duration_min": 3
    }

    SZ_animals = ['SZ036', 'SZ038', 'SZ039', 'SZ042', 'SZ043']
    RK_animals = ['RK007', 'RK008']
    animals_to_pool = SZ_animals + RK_animals
    for animal in animals_to_pool:
        pooled_transitions = pool_animal_transitions(animal, env_params,
                                                     subdir=config.PRETRAINING_PROCESSED_DATA_SUBDIR)
        # Save pooled transitions
        filename = f'pooled_transitions_{animal}.pkl'
        print(f"Saving list to {filename}...")
        try:
            # 'wb' mode means 'write binary'
            with open(filename, 'wb') as file:
                # pickle.dump() serializes the object and writes to the file
                pickle.dump(pooled_transitions, file)
            print("List saved successfully.")

        except IOError as e:
            print(f"Error saving file: {e}")
        except pickle.PicklingError as e:
            print(f"Error pickling object: {e}")

if __name__ == "__main__":
    main()