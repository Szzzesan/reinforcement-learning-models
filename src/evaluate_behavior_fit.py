import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from src.plot_state_value_trajectories import load_master_predictions
import src.config
from src.current_experiment_config import get_dated_output_dir


def _extract_valid_arrays(results):
    """
    Helper function to extract 'actual' and 'predicted' arrays,
    filtering out any trials where the prediction failed (None or NaN).
    """
    actuals = []
    predicteds = []

    for row in results:
        actual = row.get('actual')
        predicted = row.get('predicted')

        # Ensure both values exist and are valid numbers
        if actual is not None and predicted is not None and not np.isnan(predicted):
            actuals.append(actual)
            predicteds.append(predicted)

    return np.array(actuals), np.array(predicteds)


def calculate_mae(results):
    """
    Calculates the Mean Absolute Error (MAE) between the actual
    mouse leave times and the RL agent's predicted leave times.

    Returns:
        float: The average error in seconds.
    """
    actuals, predicteds = _extract_valid_arrays(results)

    if len(actuals) == 0:
        return np.nan

    # Calculate the absolute difference for each trial
    absolute_errors = np.abs(actuals - predicteds)

    # Return the mean of these errors
    return np.mean(absolute_errors)


def calculate_pearson_correlation(results):
    """
    Calculates the Pearson correlation coefficient (r) and p-value.
    This measures how well the agent's predictions track the trend
    of the mouse's actual behavior.

    Returns:
        tuple: (r_value, p_value)
    """
    actuals, predicteds = _extract_valid_arrays(results)

    if len(actuals) < 2:
        return np.nan, np.nan

    r_val, p_val = pearsonr(actuals, predicteds)
    return r_val, p_val


def evaluate_subset(subset_results):
    """Helper to bundle metrics for any slice of data."""
    if not subset_results:
        return {'N_trials': 0, 'MAE': np.nan, 'Pearson_r': np.nan, 'p_value': np.nan}

    r, p = calculate_pearson_correlation(subset_results)
    return {
        'N_trials': len(subset_results),
        'MAE': calculate_mae(subset_results),
        'Pearson_r': r,
        'p_value': p
    }


def evaluate_by_context(results):
    """Evaluates metrics grouped by Context only."""
    contexts = set(r.get('context') for r in results if 'context' in r)
    return {ctx: evaluate_subset([r for r in results if r.get('context') == ctx]) for ctx in sorted(contexts)}


def evaluate_by_animal(results):
    """Evaluates metrics grouped by Animal only."""
    animals = set(r.get('animal_id') for r in results if 'animal_id' in r)
    return {animal: evaluate_subset([r for r in results if r.get('animal_id') == animal]) for animal in sorted(animals)}


def evaluate_by_animal_and_context(results):
    """Evaluates metrics for each context, within each animal."""
    animals = set(r.get('animal_id') for r in results if 'animal_id' in r)
    contexts = set(r.get('context') for r in results if 'context' in r)

    nested_eval = {}
    for animal in sorted(animals):
        nested_eval[animal] = {}
        for ctx in sorted(contexts):
            subset = [r for r in results if r.get('animal_id') == animal and r.get('context') == ctx]
            nested_eval[animal][f"Context_{ctx}"] = evaluate_subset(subset)

    return nested_eval


def evaluate_by_animal_and_session(results):
    """Evaluates metrics for each target session, within each animal (needs 'session_id' in the master predictions)."""
    out = {}
    for animal in sorted(set(r.get('animal_id') for r in results if 'animal_id' in r)):
        sessions = sorted(set(r.get('session_id') for r in results
                              if r.get('animal_id') == animal and r.get('session_id') is not None))
        out[animal] = {sid: evaluate_subset([r for r in results
                                             if r.get('animal_id') == animal and r.get('session_id') == sid])
                       for sid in sessions}
    return out


def build_evaluation_table(results):
    """
    Collects every evaluation level into one long table:
    level | animal_id | context | session_id | N_trials | MAE | Pearson_r | p_value
    """
    rows = [dict(level='global', **evaluate_subset(results))]
    for ctx, m in evaluate_by_context(results).items():
        rows.append(dict(level='context', context=ctx, **m))
    for animal, m in evaluate_by_animal(results).items():
        rows.append(dict(level='animal', animal_id=animal, **m))
    for animal, ctx_dict in evaluate_by_animal_and_context(results).items():
        for ctx_name, m in ctx_dict.items():
            rows.append(dict(level='animal_context', animal_id=animal, context=ctx_name.replace('Context_', ''), **m))
    for animal, sess_dict in evaluate_by_animal_and_session(results).items():
        for sid, m in sess_dict.items():
            rows.append(dict(level='animal_session', animal_id=animal, session_id=sid, **m))
    cols = ['level', 'animal_id', 'context', 'session_id', 'N_trials', 'MAE', 'Pearson_r', 'p_value']
    return pd.DataFrame(rows).reindex(columns=cols)


def save_evaluation_table(results, filename="behavior_fit_evaluation.csv"):
    """Saves the full evaluation table to outputs/<exp>/4_outputs/<YYYY_MM_DD>/."""
    save_file = Path(get_dated_output_dir()) / filename
    build_evaluation_table(results).to_csv(save_file, index=False)
    print(f"💾 Evaluation table saved to {save_file}")
    return save_file


if __name__ == "__main__":
    print("Loading master predictions...")
    master_data = load_master_predictions()

    if master_data:
        print(f"Loaded {len(master_data)} valid trials.\n")
        print("=" * 50)
        print("🌍 1. GLOBAL EVALUATION")
        print("=" * 50)
        global_eval = evaluate_subset(master_data)
        print(f"Total Trials : {global_eval['N_trials']}")
        print(f"Global MAE   : {global_eval['MAE']:.3f} seconds")
        print(f"Global r     : {global_eval['Pearson_r']:.3f} (p={global_eval['p_value']:.3e})")

        print("\n" + "=" * 50)
        print("📊 2. EVALUATION BY CONTEXT (All Animals)")
        print("=" * 50)
        for ctx, metrics in evaluate_by_context(master_data).items():
            print(f"Context {ctx}: MAE={metrics['MAE']:.3f}s | r={metrics['Pearson_r']:.3f} (n={metrics['N_trials']})")

        print("\n" + "=" * 50)
        print("🐁 3. EVALUATION BY ANIMAL (Across Contexts)")
        print("=" * 50)
        for animal, metrics in evaluate_by_animal(master_data).items():
            print(f"{animal}: MAE={metrics['MAE']:.3f}s | r={metrics['Pearson_r']:.3f} (n={metrics['N_trials']})")

        print("\n" + "=" * 50)
        print("🔬 4. EVALUATION BY ANIMAL AND CONTEXT")
        print("=" * 50)
        animal_ctx_eval = evaluate_by_animal_and_context(master_data)
        for animal, contexts in animal_ctx_eval.items():
            print(f"{animal}:")
            for ctx_name, metrics in contexts.items():
                print(
                    f"  └─ {ctx_name}: MAE={metrics['MAE']:.3f}s | r={metrics['Pearson_r']:.3f} (n={metrics['N_trials']})")
        print("=" * 50)

        # Per-session metrics are written to the table only (80 rows is too much to print)
        save_evaluation_table(master_data)