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

    def _initialize_custom_gambling_dist(self, cumulative=8., starting=1.):
        def exp_decreasing(x, c, s):
            a = s
            b = a / c
            density = a / np.exp(b * x)
            return density
        raw_density = exp_decreasing(self.time_bins, c=cumulative, s=starting)
        self.gambling_pmf = raw_density / 10
        self.gambling_cdf = np.cumsum(self.gambling_pmf)

    def env_init(self, env_info={}):
        """
        Setup the environment with parameters.
        """
        # --- time ---
        self.dt = env_info.get("time_step_duration", 0.1)
        self.travel_time = env_info.get("travel_time", 1.0)
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

        # --- Pre-calculate reward probabilities ---
        precision = str(self.dt)[::-1].find('.')
        self.time_bins = np.round(np.arange(0, self.gambling_max_time + self.dt, self.dt), precision)

        # This helper method handles the normalization and CDF calculation
        self._initialize_custom_gambling_dist(
            cumulative=self.gambling_cumulative,
            starting=self.gambling_starting
        )

    def _calculate_gambling_reward(self):
        """
        Calculates if a reward should be delivered based on the custom hazard rate.
        """
        if self.time_in_port >= self.gambling_max_time:
            return 0.0

        time_idx = int(self.time_in_port / self.dt)

        prob_never_rewarded_yet = 1 - self.gambling_cdf[time_idx]
        prob_reward_at_this_instant = self.gambling_pmf[time_idx]  # PMF value is already scaled by dt implicitly

        if prob_never_rewarded_yet > 1e-9:
            hazard_rate = prob_reward_at_this_instant / prob_never_rewarded_yet
            if self.rand_generator.uniform() < hazard_rate:
                return 1.0

        return 0.0

    def env_start(self):
        self.total_time_elapsed = 0
        self.current_context = self.rand_generator.choice([0, 1])
        self.port_id = self.CONTEXT_PORT
        self.time_in_port = 0.0
        self.rewards_in_context = 0
        self.is_traveling = False
        return self._get_observation()

    def env_step(self, action):
        reward = 0.0
        self.total_time_elapsed += self.dt
        if self.total_time_elapsed % self.block_duration < self.dt:
            self.current_context = 1 - self.current_context
        terminal = self.total_time_elapsed >= self.session_duration

        if self.is_traveling:
            self.time_in_port += self.dt
            if self.time_in_port >= self.travel_time:
                self.is_traveling = False
                self.time_in_port = 0.0
                self.port_id = 1 - self.port_id

        elif action == 0:  # Stay
            self.time_in_port += self.dt

            if self.port_id == self.CONTEXT_PORT:
                interval = self.low_rate_interval if self.current_context == 0 else self.high_rate_interval
                next_reward_time = (self.rewards_in_context + 1) * interval
                if self.rewards_in_context < self.context_rewards_max and abs(self.time_in_port - next_reward_time) < (
                        self.dt / 2):
                    reward = 1.0
                    self.rewards_in_context += 1

            elif self.port_id == self.GAMBLING_PORT:
                reward = self._calculate_gambling_reward()

        elif action == 1:  # Leave
            self.is_traveling = True
            self.time_in_port = 0.0
            if self.port_id == self.CONTEXT_PORT:
                self.rewards_in_context = 0

        next_observation = self._get_observation()
        return (reward, next_observation, terminal)

    def _get_observation(self):
        current_port = 2 if self.is_traveling else self.port_id
        return np.array([current_port, self.time_in_port, self.current_context, self.rewards_in_context])

    def env_cleanup(self):
        pass

    def env_message(self, message):
        if message == "get_num_features":
            return 4
        return None