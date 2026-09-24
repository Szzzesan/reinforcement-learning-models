import os
import json
from datetime import datetime

# ==========================================
# 1. SET ACTIVE EXPERIMENT HERE
# ==========================================
# Change this single variable to switch the entire pipeline's context.
# run_pipeline.py can override it for one run via the RL_ACTIVE_EXP_ID environment variable.
ACTIVE_EXP_ID = os.environ.get("RL_ACTIVE_EXP_ID", "exp_06")

# Test-time learning on the target (held-out) sessions, used by 03 and plot_state_value_trajectories:
#   'frozen'   -> alpha = 0, the agent's weights never change on the target sessions (default)
#   'learning' -> keep each animal's fitted alpha, so the agent keeps learning through the target sessions
# 'learning' results go to a 'learning_alpha' subfolder of 3_evaluation_metrics/ and 4_outputs/<date>/, so both
# versions coexist. run_pipeline.py can override this for one run via RL_TEST_ALPHA_MODE.
TEST_ALPHA_MODE = os.environ.get("RL_TEST_ALPHA_MODE", "frozen")
if TEST_ALPHA_MODE not in ("frozen", "learning"):
    raise ValueError(f"TEST_ALPHA_MODE must be 'frozen' or 'learning', got {TEST_ALPHA_MODE!r}")
TEST_ALPHA_SUBFOLDERS = {"frozen": "", "learning": "learning_alpha"}
TEST_ALPHA_SUBFOLDER = TEST_ALPHA_SUBFOLDERS[TEST_ALPHA_MODE]

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
    },
    "exp_05": {
        "split_method": "holdout_last_10",
        "tiling_method": "uniform",
        "description": ("Chronological holdout: the last 10 sessions (all post-surgery) are the target set and are never "
                        "seen during fitting. alpha, gamma and lambda are fit on prequential leave-time MSE over "
                        "pre-surgery trials only; early post-surgery training sessions are replayed but not scored. "
                        "Uniform tiling."),
        "score_session_type": "pre-surgery"
    },
    "exp_06": {
        "split_method": "mixed_50_percent",
        "tiling_method": "uniform",
        "description": ("Train = all pre-surgery + first 50% of post-surgery sessions; target = last 50% of post-surgery "
                        "(odd counts put the extra session in the target set). alpha, gamma and lambda are fit on "
                        "prequential leave-time MSE over the post-surgery TRAINING sessions only; pre-surgery sessions "
                        "are replayed but not scored. Uniform tiling."),
        "score_session_type": "post-surgery"
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
DIR_STEP4_OUTPUTS = os.path.join(DIR_OUTPUTS, "4_outputs")  # figures + result tables, one dated subfolder per run day
DIR_FIGURES = DIR_STEP4_OUTPUTS  # old name kept so existing imports keep working

# Fixed once at import, so a run that crosses midnight still writes into a single folder
RUN_DATE = datetime.now().strftime("%Y_%m_%d")


def get_dated_output_dir(*subdirs, mode_subfolder=True):
    """
    Returns (and creates) outputs/<exp>/4_outputs/<YYYY_MM_DD>/<subdirs...> for the active experiment,
    e.g. get_dated_output_dir() -> .../4_outputs/2026_09_23
         get_dated_output_dir("leave_time_by_session", "SZ036") -> .../4_outputs/2026_09_23/leave_time_by_session/SZ036
    In 'learning' test-alpha mode the path gains a 'learning_alpha' level right after the date
    (pass mode_subfolder=False to get the shared dated folder, e.g. for frozen-vs-learning comparisons).
    """
    parts = [DIR_STEP4_OUTPUTS, RUN_DATE]
    if mode_subfolder and TEST_ALPHA_SUBFOLDER:
        parts.append(TEST_ALPHA_SUBFOLDER)
    path = os.path.join(*parts, *subdirs)
    os.makedirs(path, exist_ok=True)
    return path

def get_eval_metrics_dir(test_alpha_mode=None):
    """outputs/<exp>/3_evaluation_metrics[/learning_alpha] for the given test-alpha mode (default: active mode)."""
    sub = TEST_ALPHA_SUBFOLDERS[test_alpha_mode or TEST_ALPHA_MODE]
    return os.path.join(DIR_EVAL_METRICS, sub) if sub else DIR_EVAL_METRICS


def initialize_active_experiment():
    """Creates the necessary directories for the currently active experiment."""
    print(f"\n{'=' * 50}\nInitializing {ACTIVE_EXP_ID}: {ACTIVE_CONFIG['description']}\n{'=' * 50}")

    # Create data dir
    os.makedirs(DIR_DATA, exist_ok=True)

    # Create output subdirs
    os.makedirs(DIR_MODEL_FITTING, exist_ok=True)
    os.makedirs(DIR_TRAINED_AGENTS, exist_ok=True)
    os.makedirs(DIR_EVAL_METRICS, exist_ok=True)
    os.makedirs(DIR_STEP4_OUTPUTS, exist_ok=True)

    # Save config to outputs
    config_path = os.path.join(DIR_OUTPUTS, "exp_config.json")
    with open(config_path, "w") as f:
        json.dump(ACTIVE_CONFIG, f, indent=4)

if __name__ == "__main__":
    initialize_active_experiment()