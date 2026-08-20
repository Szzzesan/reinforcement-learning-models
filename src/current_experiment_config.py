import os
import json

# ==========================================
# 1. SET ACTIVE EXPERIMENT HERE
# ==========================================
# Change this single variable to switch the entire pipeline's context
ACTIVE_EXP_ID = "exp_01"

# ==========================================
# 2. PROJECT ROOT & EXPERIMENT CONFIGURATIONS
# ==========================================
# Adjust this to point to your actual root directory
MODELING_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXPERIMENTS = {
    "exp_01": {
        "split_method": "presurgery_only",
        "tiling_method": "uniform",
        "description": "Baseline model: pre-surgery train, post-surgery test with uniform tiling."
    },
    "exp_02": {
        "split_method": "mixed_20_percent",
        "tiling_method": "uniform",
        "description": "Train includes all pre-surgery + first 20% post-surgery. Uniform tiling."
    },
    "exp_03": {
        "split_method": "mixed_20_percent",
        "tiling_method": "logarithmic",
        "description": "Train includes all pre-surgery + first 20% post-surgery. Log-expanded temporal tiling."
    },
    "exp_04": {
        "split_method": "mixed_40_percent",
        "tiling_method": "uniform",
        "description": "Train includes all pre-surgery + first 40% post-surgery. Uniform tiling."
    }
}



# ==========================================
# 3. DYNAMIC PATH GENERATION
# ==========================================
if ACTIVE_EXP_ID not in EXPERIMENTS:
    raise ValueError(f"Experiment {ACTIVE_EXP_ID} not found in configurations.")

ACTIVE_CONFIG = EXPERIMENTS[ACTIVE_EXP_ID]
ACTIVE_CONFIG["exp_id"] = ACTIVE_EXP_ID  # Inject the ID back in for completeness

# Generate the folder suffix based on your naming convention
FOLDER_SUFFIX = f"{ACTIVE_EXP_ID}_{ACTIVE_CONFIG['split_method']}_{ACTIVE_CONFIG['tiling_method']}"

# Base directories
BASE_DIR_DATA = os.path.join(MODELING_PROJECT_ROOT, "data")
BASE_DIR_OUTPUTS = os.path.join(MODELING_PROJECT_ROOT, "outputs")

# Active directories
DIR_DATA = os.path.join(BASE_DIR_DATA, FOLDER_SUFFIX)
DIR_OUTPUTS = os.path.join(BASE_DIR_OUTPUTS, FOLDER_SUFFIX)

# Output subfolders for easy importing
DIR_MODEL_FITTING = os.path.join(DIR_OUTPUTS, "1_model_fitting")
DIR_TRAINED_AGENTS = os.path.join(DIR_OUTPUTS, "2_trained_agents")
DIR_EVAL_METRICS = os.path.join(DIR_OUTPUTS, "3_evaluation_metrics")
DIR_FIGURES = os.path.join(DIR_OUTPUTS, "4_figures")

def initialize_active_experiment():
    """Creates the necessary directories for the currently active experiment."""
    print(f"\n{'=' * 50}\nInitializing {ACTIVE_EXP_ID}: {ACTIVE_CONFIG['description']}\n{'=' * 50}")

    # Create data dir
    os.makedirs(DIR_DATA, exist_ok=True)

    # Create output subdirs
    os.makedirs(DIR_MODEL_FITTING, exist_ok=True)
    os.makedirs(DIR_TRAINED_AGENTS, exist_ok=True)
    os.makedirs(DIR_EVAL_METRICS, exist_ok=True)
    os.makedirs(DIR_FIGURES, exist_ok=True)

    # Save config to outputs
    config_path = os.path.join(DIR_OUTPUTS, "exp_config.json")
    with open(config_path, "w") as f:
        json.dump(ACTIVE_CONFIG, f, indent=4)

if __name__ == "__main__":
    initialize_active_experiment()