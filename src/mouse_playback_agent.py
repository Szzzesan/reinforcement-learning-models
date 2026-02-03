import numpy as np
from agent import BaseAgent
import tiles3 as tc


class MousePlaybackAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        self.discount = agent_info.get("discount", 0.95)
        self.step_size = agent_info.get("step_size", 0.1)

        self.num_tilings = agent_info.get("num_tilings", 8)

        # The Index Hash Table (IHT) stores the tile indices. The size determines memory usage.
        self.iht_size = agent_info.get("iht_size", 4096)
        self.iht = tc.IHT(self.iht_size)

        # The weight vector. One weight for each possible tile.
        self.w = np.zeros(self.iht_size)

        # State feature scales for the tile coder. These normalize the inputs.
        # [port, time_in_port, event_timer, context, rewards_in_context, gambling_disabled]
        default_scales = [
            -1,  # Port ID (0, 1, 2)
            1 / 2.0,  # Time in Port
            0.0,  # Event Timer (disabled)
            -1,  # Context (0, 1)
            -1,  # Rewards in Context
            -1  # Gambling Disabled (0, 1)
        ]

        self.scales = agent_info.get("scales", default_scales)

        self.last_state_tiles = None
        self.td_error_log = []

    # Helper function to get the value of a state using the tile coder
    def _get_active_tiles(self, raw_state):
        """
                Takes the raw state vector and returns the list of active tiles.
                """

        my_floats = []
        my_ints = []

        # Loop through the raw state and your scales list
        for i, raw_val in enumerate(raw_state):
            scale = self.scales[i]  # Get the scale for this feature

            if scale > 0.0:
                # 1. It's a FLOAT. Scale it and add to floats list.
                my_floats.append(raw_val * scale)
            elif scale < 0.0:
                # 2. It's an INT. Add the raw value (unscaled) to ints list.
                my_ints.append(int(raw_val))
            else:
                # 3. scale == 0.0, so it's DISABLED. Do nothing.
                pass

        # Now, call the 'tiles' function from tiles3.py with your sorted lists
        active_tiles = tc.tiles(self.iht, self.num_tilings, my_floats, my_ints)

        return active_tiles

    def get_value(self, state_features):
        """
        Calculates the state-value estimate using the current weights and active tiles.
        """

        active_tiles = self._get_active_tiles(state_features)
        return np.sum(self.w[active_tiles])

    def get_weights(self, state_features):
        """
        Get the weights of the active tiles based on the state features.
        """
        active_tiles = self._get_active_tiles(state_features)
        return self.w[active_tiles]

    def agent_start(self, observation):
        """The first method called when the experiment starts."""
        # Get the active tiles for the initial state
        self.last_state_tiles = self._get_active_tiles(observation)

        # No action is returned because the experiment loop will provide it.
        return None

    def agent_step(self, reward, observation):
        """A step taken by the agent. The action is implicit in the transition."""

        # Get the value of the new state (S_t+1)
        v_next_state = self.get_value(observation)

        # Get the value of the previous state (S_t)
        v_last_state = np.sum(self.w[self.last_state_tiles])

        # Calculate the TD-error (delta)
        td_error = reward + self.discount * v_next_state - v_last_state
        self.td_error_log.append(td_error)

        # Perform gradient descent update.
        # The update is applied to the weights of the tiles active in the *last* state.
        # Normalize the step size by the number of tilings for stability.
        update_size = self.step_size / self.num_tilings * td_error
        for tile_index in self.last_state_tiles:
            self.w[tile_index] += update_size

        # Update the last state's tiles for the next iteration
        self.last_state_tiles = self._get_active_tiles(observation)

        # No action is returned as we are in playback mode.
        return None

    def agent_end(self, reward):
        """Run when the agent terminates."""
        v_last_state = np.sum(self.w[self.last_state_tiles])

        # The value of the terminal state is 0.
        td_error = reward + self.discount * 0 - v_last_state
        self.td_error_log.append(td_error)

        update_size = self.step_size / self.num_tilings * td_error
        for tile_index in self.last_state_tiles:
            self.w[tile_index] += update_size

    def agent_cleanup(self):
        self.last_state_tiles = None

    def agent_message(self, message):
        if message == "get_td_errors":
            return self.td_error_log
        return "Message not supported"