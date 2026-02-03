import numpy as np
from environment import BaseEnvironment


class ForagingEnvironment(BaseEnvironment):
    """
    An RL environment for a patch-foraging task.
    """

    def __init__(self):
        super().__init__()
        self.CONTEXT_PORT = 0
        self.GAMBLING_PORT = 1

    def env_init(self, env_info={}):
        """
        Set up the environment with parameters.
        """
        # --- time ---
        self.dt = env_info.get("time_step_duration", 0.1)
        self.travel_time = env_info.get("travel_time",
                                        0.4)  # minimum duration animals need to travel (to prevent teleporting)
        self.session_duration = env_info.get("session_duration_min", 18) * 60
        self.block_duration = env_info.get("block_duration_min", 3) * 60

        # --- Context Port Parameters ---
        self.high_rate_interval = env_info.get("high_rate_interval_s", 1.25)
        self.low_rate_interval = env_info.get("low_rate_interval_s", 2.5)
        self.context_rewards_max = env_info.get("context_rewards_max", 4)

        # --- Gambling Port Parameters ---
        self.gambling_cumulative = env_info.get("gambling_cumulative", 8.0)
        self.gambling_starting = env_info.get("gambling_starting", 1.0)
        self.gambling_max_time = env_info.get("gambling_max_time_s", 30.0)

        self.rand_generator = np.random.RandomState(env_info.get("seed", 0))

    def _get_gambling_reward_prob(self, time_in_port):
        """
        Calculates the independent probability of reward at a given time
        """
        a = self.gambling_starting
        b = a / self.gambling_cumulative
        raw_prob = a / np.exp(b * time_in_port)
        prob_this_step = raw_prob * self.dt
        if not (0 <= prob_this_step <= 1):
            raise ValueError(
                f"Calculated probability is outside the valid [0, 1] range.\n"
                f"  - Calculated Value: {prob_this_step}\n"
                f"  - Time in Port: {time_in_port}s\n"
                f"This is likely caused by the combination of the parameters.\n"
                f"Please check: starting={self.gambling_starting}, cumulative={self.gambling_cumulative}, dt={self.dt}"
            )
        return prob_this_step

    def env_start(self):
        self.total_time_elapsed = 0
        self.current_context = self.rand_generator.choice([0, 1])

        # Initialize state variables
        self.port_id = self.CONTEXT_PORT
        self.time_in_port = 0.0
        self.event_timer = 0.0  # time since the last significant event
        self.rewards_in_context = 0

        self.gambling_port_disabled = False
        self.time_since_last_context_reward = 0.0

        self.is_traveling = False
        self.destination_port = None

        self.trial_number = 1
        self.block_switching_pending = False

        return self._get_observation()

    def env_step(self, action):
        reward = 0.0
        self.total_time_elapsed += self.dt
        if self.total_time_elapsed > 0 and self.total_time_elapsed % self.block_duration < self.dt:
            self.block_switching_pending = True

        terminal = self.total_time_elapsed >= self.session_duration
        interval = self.low_rate_interval if self.current_context == 0 else self.high_rate_interval

        if self.is_traveling:
            self.time_in_port += self.dt
            self.event_timer += self.dt
            if (action == 1) and (self.time_in_port >= self.travel_time):
                if self.destination_port == self.CONTEXT_PORT:
                    if not self.gambling_port_disabled:
                        self.trial_number += 1
                        if self.block_switching_pending:
                            self.current_context = 1 - self.current_context
                            self.block_switching_pending = False
                self.is_traveling = False
                self.time_in_port = 0.0
                self.event_timer = 0.0
                self.port_id = self.destination_port
                self.destination_port = None
            else:
                pass

        elif action == 0:  # Stay
            self.time_in_port += self.dt
            self.event_timer += self.dt

            if self.port_id == self.CONTEXT_PORT:
                time_to_wait = interval - self.time_since_last_context_reward
                if (self.rewards_in_context < self.context_rewards_max) and (
                        self.event_timer - self.dt < time_to_wait) and (self.event_timer >= time_to_wait):
                    reward = 1.0
                    self.rewards_in_context += 1
                    self.event_timer = 0.0  # Reset port timer after each reward
                    self.time_since_last_context_reward = 0.0  # Reset resume timer

                    if self.rewards_in_context == self.context_rewards_max:
                        self.gambling_port_disabled = False

            elif self.port_id == self.GAMBLING_PORT:
                if not self.gambling_port_disabled:
                    prob = self._get_gambling_reward_prob(self.time_in_port)
                    if self.rand_generator.uniform() < prob:
                        reward = 1.0
                        self.event_timer = 0.0
                        # print(
                        #     f"*** Gambling Reward delivered at total_time: {self.total_time_elapsed:.2f}s, time_in_port: {self.time_in_port:.2f}s ***")

        elif action == 1:  # Leave
            if self.port_id == self.CONTEXT_PORT:
                self.destination_port = self.GAMBLING_PORT

                # --- Check for premature leave ---
                if self.rewards_in_context < self.context_rewards_max:
                    self.gambling_port_disabled = True
                    self.time_since_last_context_reward = self.event_timer
                else:  # Left after finishing, reset everything
                    self.rewards_in_context = 0
                    self.time_since_last_context_reward = 0.0

            elif self.port_id == self.GAMBLING_PORT:
                self.destination_port = self.CONTEXT_PORT

            self.is_traveling = True
            self.time_in_port = 0.0
            self.event_timer = 0.0

        next_observation = self._get_observation()
        return (reward, next_observation, terminal)

    def _get_observation(self):
        current_port = 2 if self.is_traveling else self.port_id
        gambling_disabled_feature = 1.0 if self.gambling_port_disabled else 0.0
        return np.array([
            current_port,
            self.time_in_port,
            self.event_timer,
            self.current_context,
            self.rewards_in_context,
            gambling_disabled_feature
        ])

    def env_cleanup(self):
        pass

    def env_message(self, message):
        if message == "get_num_features":
            return 6
        if message == "get_trial_number":
            return self.trial_number
        return None
