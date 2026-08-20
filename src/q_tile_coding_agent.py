import numpy as np
from src.agent import BaseAgent  # Assuming you have a BaseAgent class
import tiles3 as tc


class QLearningTileCodingAgent(BaseAgent):
    """
    An autonomous Q-learning agent using tile coding for linear function approximation.
    Designed to interact with the ForagingEnvironment.
    """

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        self.discount = agent_info.get("discount", 0.78)
        self.step_size = agent_info.get("step_size", 0.001)

        # New parameter for autonomous exploration
        self.epsilon = agent_info.get("epsilon_start", 0.9)
        self.epsilon_decay = agent_info.get("epsilon_decay", 0.99998) #0.9999732 for it to get below 0.05 over 10 sessions
        self.epsilon_min = agent_info.get("epsilon_min", 0.05)
        self.num_actions = 2  # 0: Stay, 1: Leave

        self.num_tilings = agent_info.get("num_tilings", 8)
        self.iht_size = agent_info.get("iht_size", 32768)
        self.iht = tc.IHT(self.iht_size)

        # The weight matrix. One weight vector for each possible action.
        self.w = np.zeros((self.num_actions, self.iht_size))
        # self.w = np.full((self.num_actions, self.iht_size), 0.5)

        # State feature scales for the tile coder.
        # [port, time_in_port, event_timer, context, rewards_in_context, gambling_disabled]
        default_scales = [
            -1,  # Port ID (0, 1, 2)
            1 / 2.0,  # Time in Port
            0.0,  # Event Timer (disabled)
            -1,  # Context (0, 1)
            0,  # Rewards in Context
            -1  # Gambling Disabled (0, 1)
        ]

        self.scales = agent_info.get("scales", default_scales)

        self.last_state_tiles = None
        self.last_action = None
        self.rpe_log = []  # Renamed to reflect Reward Prediction Error

    # def _get_active_tiles(self, raw_state):
    #     """Takes the raw state vector and returns the list of active tiles."""
    #     my_floats = []
    #     my_ints = []
    #
    #     for i, raw_val in enumerate(raw_state):
    #         scale = self.scales[i]
    #
    #         if scale > 0.0:
    #             my_floats.append(raw_val * scale)
    #         elif scale < 0.0:
    #             my_ints.append(int(raw_val))
    #         else:
    #             pass
    #
    #     active_tiles = tc.tiles(self.iht, self.num_tilings, my_floats, my_ints)
    #     return active_tiles

    def _get_active_tiles(self, raw_state, action):
        """Takes the raw state vector and returns active tiles for a SPECIFIC action."""
        my_floats = []
        my_ints = []

        for i, raw_val in enumerate(raw_state):
            scale = self.scales[i]

            # --- MVT ARCHITECTURE FIX ---
            # If evaluating 'Leave' (Action 1), the value should NOT depend on time_in_port (index 1)
            # This 'action' comes from the function argument above!
            if action == 1 and i == 1:
                raw_val = 0.0

            if scale > 0.0:
                my_floats.append(raw_val * scale)
            elif scale < 0.0:
                my_ints.append(int(raw_val))
            else:
                pass

        active_tiles = tc.tiles(self.iht, self.num_tilings, my_floats, my_ints)
        return active_tiles

    # def _get_q_values(self, active_tiles):
    #     """Calculates Q-values for all actions given the active tiles."""
    #     q_values = np.zeros(self.num_actions)
    #     for a in range(self.num_actions):
    #         q_values[a] = np.sum(self.w[a, active_tiles])
    #     return q_values

    def _get_q_values(self, raw_state):
        """Calculates Q-values by fetching tiles specific to each action's dependencies."""
        q_values = np.zeros(self.num_actions)

        # We loop through both actions and get unique tiles for each!
        for a in range(self.num_actions):
            # We pass 'a' (which is 0 or 1) into the function as the 'action' argument
            active_tiles_for_this_action = self._get_active_tiles(raw_state, a)
            q_values[a] = np.sum(self.w[a, active_tiles_for_this_action])

        return q_values

    def _argmax(self, q_values):
        """Tie-breaking argmax to prevent action bias."""
        top = float("-inf")
        ties = []
        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = [i]
            elif q_values[i] == top:
                ties.append(i)
        return self.rand_generator.choice(ties) if hasattr(self, 'rand_generator') else np.random.choice(ties)

    def agent_start(self, observation):
        """The first method called when a trial/episode starts."""
        self.last_state_tiles = self._get_active_tiles(observation)
        q_values = self._get_q_values(self.last_state_tiles)

        # Modified Epsilon-greedy action selection
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
        if np.random.rand() < self.epsilon:
            # Instead of a 50/50 split (np.random.choice), we strongly bias toward Stay (0)
            # 95% chance to stay, 5% chance to leave when exploring
            self.last_action = 0 if np.random.rand() < 0.99 else 1
        else:
            self.last_action = self._argmax(q_values)  # or q_values_next in agent_step

        return self.last_action

    def agent_step(self, reward, observation):
        """A step taken by the agent updating weights and choosing the next action."""
        current_state_tiles = self._get_active_tiles(observation)
        q_values_next = self._get_q_values(current_state_tiles)

        # Find the max Q-value for the next state for the Q-learning update
        max_q_next = np.max(q_values_next)

        # Value of the action we just took
        q_last = np.sum(self.w[self.last_action, self.last_state_tiles])

        # Calculate the Reward Prediction Error (RPE / TD-error)
        rpe = reward + self.discount * max_q_next - q_last
        self.rpe_log.append(rpe)

        # Perform gradient descent update
        update_size = self.step_size / self.num_tilings * rpe
        for tile_index in self.last_state_tiles:
            self.w[self.last_action, tile_index] += update_size

        # Modified Epsilon-greedy action selection
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
        if np.random.rand() < self.epsilon:
            # Instead of a 50/50 split (np.random.choice), we strongly bias toward Stay (0)
            # 95% chance to stay, 5% chance to leave when exploring
            action = 0 if np.random.rand() < 0.99 else 1
        else:
            action = self._argmax(q_values_next)  # or q_values_next in agent_step

        # Update state and action for the next step
        self.last_state_tiles = current_state_tiles
        self.last_action = action

        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates (e.g., end of session)."""
        q_last = np.sum(self.w[self.last_action, self.last_state_tiles])

        # Terminal state value is explicitly 0
        rpe = reward + self.discount * 0 - q_last
        self.rpe_log.append(rpe)

        update_size = self.step_size / self.num_tilings * rpe
        for tile_index in self.last_state_tiles:
            self.w[self.last_action, tile_index] += update_size

    def agent_cleanup(self):
        self.last_state_tiles = None
        self.last_action = None

    def agent_message(self, message):
        if message == "get_rpe_log":
            return self.rpe_log
        return "Message not supported"