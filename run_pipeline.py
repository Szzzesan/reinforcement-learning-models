"""
run_pipeline.py -- run the whole modeling pipeline for one experiment, unattended.

    00 pool transitions -> 01 fit alpha/gamma/lambda -> 02 train agents -> 03 target-session TD errors
    -> plot_state_value_trajectories -> evaluate_behavior_fit -> plot_DA_vs_temporal_features

Each step runs as its own Python process (exactly as if you clicked Run on that file), one after
another. The pipeline stops at the first step that fails, so later steps never run on stale inputs.

Usage (from PyCharm: just Run this file; or from a terminal at the repo root):
    python run_pipeline.py                        # all steps, experiment set in current_experiment_config
    python run_pipeline.py --exp exp_06           # override the experiment for this run only
    python run_pipeline.py --from 02              # resume from a step (e.g. after 00/01 already finished)
    python run_pipeline.py --from 03 --to eval    # run a slice
    python run_pipeline.py --list                 # show step names
    python run_pipeline.py --test-alpha learning --from 03   # redo 03 onward with the fitted alpha kept on the
                                                             # target sessions (results in */learning_alpha/),
                                                             # then compare with the frozen run

Notes
- Figures are rendered off-screen (MPLBACKEND=Agg), so plt.show() does not block the run.
  Every figure is saved to outputs/<exp>/4_outputs/<YYYY_MM_DD>/ by the plotting scripts.
- Console output of every step is also written to outputs/<exp>/4_outputs/<YYYY_MM_DD>/pipeline_log_<HH-MM-SS>.txt
- Each step reads its settings from the scripts themselves (e.g. N_JOBS / grids in 01,
  FORCE_RECOMPILE / make_plots in plot_state_value_trajectories).
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")

# (short name, script path relative to repo root)
STEPS = [
    ("init", "src/current_experiment_config.py"),   # creates the experiment folders + exp_config.json
    ("00", "src/00_pool_animal_transitions.py"),
    ("01", "src/01_model_fitting.py"),
    ("02", "src/02_mouse_playback_pooled_training.py"),
    ("03", "src/03_target_session_training_with_expert_agent.py"),
    ("trajectories", "src/plot_state_value_trajectories.py"),
    ("eval", "src/evaluate_behavior_fit.py"),
    ("da", "figure_making/plot_DA_vs_temporal_features.py"),
    ("compare", "src/compare_test_alpha_modes.py"),  # frozen vs learning; skips itself if one mode is missing
]
STEP_NAMES = [name for name, _ in STEPS]


def build_env(exp_id=None, test_alpha=None):
    env = os.environ.copy()
    # Same import setup as PyCharm: 'from src.x import ...' needs the repo root,
    # 'import tiles3' / 'from mouse_playback_environment import ...' need src/.
    env["PYTHONPATH"] = os.pathsep.join(p for p in [REPO_ROOT, SRC_DIR, env.get("PYTHONPATH", "")] if p)
    env["MPLBACKEND"] = "Agg"          # no blocking figure windows
    env["PYTHONUNBUFFERED"] = "1"      # stream output live
    env["PYTHONIOENCODING"] = "utf-8"  # the scripts print emoji; avoid cp1252 errors on Windows
    if exp_id is not None:
        env["RL_ACTIVE_EXP_ID"] = exp_id
    if test_alpha is not None:
        env["RL_TEST_ALPHA_MODE"] = test_alpha
    return env


def resolve_log_path(env):
    """Asks current_experiment_config (with the same env) where this run's dated output folder is."""
    code = ("from src.current_experiment_config import get_dated_output_dir, ACTIVE_EXP_ID, TEST_ALPHA_MODE; "
            "print(ACTIVE_EXP_ID + ' | test alpha: ' + TEST_ALPHA_MODE); print(get_dated_output_dir())")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()
    exp_id, out_dir = out[-2], out[-1]
    return exp_id, os.path.join(out_dir, f"pipeline_log_{datetime.now().strftime('%H-%M-%S')}.txt")


def run_step(name, script, env, log):
    header = f"\n{'#' * 70}\n# STEP {name}: {script}\n# started {datetime.now():%Y-%m-%d %H:%M:%S}\n{'#' * 70}\n"
    print(header, end="")
    log.write(header)
    log.flush()

    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-u", os.path.join(REPO_ROOT, script)],
                            cwd=REPO_ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
    proc.wait()
    minutes = (time.time() - t0) / 60

    footer = f"# STEP {name} {'finished' if proc.returncode == 0 else 'FAILED'} " \
             f"(exit code {proc.returncode}) after {minutes:.1f} min\n"
    print(footer, end="")
    log.write(footer)
    log.flush()
    return proc.returncode, minutes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", default=None, help="experiment id, e.g. exp_06 (default: ACTIVE_EXP_ID)")
    parser.add_argument("--from", dest="start", default=STEP_NAMES[0], choices=STEP_NAMES)
    parser.add_argument("--to", dest="stop", default=STEP_NAMES[-1], choices=STEP_NAMES)
    parser.add_argument("--test-alpha", dest="test_alpha", default=None, choices=["frozen", "learning"],
                        help="alpha on the target sessions for 03 and the trajectories step "
                             "(default: TEST_ALPHA_MODE in current_experiment_config, i.e. frozen)")
    parser.add_argument("--list", action="store_true", help="list steps and exit")
    args = parser.parse_args()

    if args.list:
        for name, script in STEPS:
            print(f"{name:14s} {script}")
        return 0

    i0, i1 = STEP_NAMES.index(args.start), STEP_NAMES.index(args.stop)
    if i0 > i1:
        parser.error("--from comes after --to")
    selected = STEPS[i0:i1 + 1]

    env = build_env(args.exp, args.test_alpha)
    exp_id, log_path = resolve_log_path(env)
    print(f"Experiment: {exp_id} | steps: {', '.join(n for n, _ in selected)}\nLog: {log_path}")

    summary = []
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"Experiment: {exp_id}\nSteps: {[n for n, _ in selected]}\nPython: {sys.executable}\n")
        for name, script in selected:
            code, minutes = run_step(name, script, env, log)
            summary.append((name, code, minutes))
            if code != 0:
                break

        lines = ["\n" + "=" * 70, f"PIPELINE SUMMARY ({exp_id})"]
        lines += [f"  {name:14s} {'OK    ' if code == 0 else 'FAILED'} {minutes:7.1f} min" for name, code, minutes in summary]
        not_run = [n for n, _ in selected[len(summary):]]
        if not_run:
            lines.append(f"  not run: {', '.join(not_run)}  (fix the failed step, then use --from {summary[-1][0]})")
        lines.append("=" * 70)
        print("\n".join(lines))
        log.write("\n".join(lines) + "\n")

    return 0 if all(code == 0 for _, code, _ in summary) and len(summary) == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
