# src/state_utils.py
import numpy as np
from src.rl_config import ENV_PARAMS


def build_travel_state(current_obs):
    """
    Constructs the 1-step lookahead state for initiating travel.
    Used for V_leave calculations.
    """
    obs_l = current_obs.copy()
    obs_l[0] = 2.0  # Port ID: Traveling
    obs_l[1] = 0.0  # Time in Port: Just started traveling
    obs_l[2] = 0.0  # Event Timer: Reset after significant exit event
    obs_l[5] = 1 # Once the animal exits, the Investment Port becomes disabled

    # Note: obs_l[3] (context), obs_l[4] (rewards from the context port)
    # automatically carry over natively from the .copy()!
    return obs_l


def build_investment_sim_state(time_in_port, time_from_prior_event, context):
    """
    Constructs the hypothetical future state vector for Monte Carlo rollouts
    inside the active foraging phase.
    """
    # [Port=Investment, Time in Port, Event Timer, Context, Rewards=4, Disabled=False]
    return np.array([1.0, time_in_port, time_from_prior_event, context, 4, 0.0])

def is_investment_state(obs):
    """
    Checks if the observation vector corresponds to an active, valid Investment (foraging) state.
    Requires: Port=Investment (1.0), Rewards=4.0, Disabled=False (0.0).
    """
    return (obs[0] == 1.0) and (obs[4] == 4.0) and (obs[5] == 0.0)

def get_investment_reward_prob(time_in_port,
                               cumulative=ENV_PARAMS["gambling_cumulative"],
                               starting=ENV_PARAMS["gambling_starting"],
                               dt=ENV_PARAMS["time_step_duration"]):
    """
    Probability of a reward in a single dt bin in the investment port, from the
    exponentially decaying rate a * exp(-(a / cumulative) * t).

    Single source of truth: replaces exp_decreasing_prob (01_model_fitting.py),
    exp_decreasing (plot_state_value_trajectories.py) and
    ForagingEnvironment._get_gambling_reward_prob (foraging_environment.py).
    """
    a = starting
    b = a / cumulative
    rate = a / np.exp(b * time_in_port)
    prob_this_step = rate * dt

    if not (0 <= prob_this_step <= 1):
        raise ValueError(
            f"Calculated probability is outside the valid [0, 1] range.\n"
            f"  - Calculated Value: {prob_this_step}\n"
            f"  - Time in Port: {time_in_port}s\n"
            f"This is likely caused by the combination of the parameters.\n"
            f"Please check: starting={starting}, cumulative={cumulative}, dt={dt}"
        )
    return prob_this_step