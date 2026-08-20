import pickle
import numpy as np

with open('data/pooled_transitions_noncircular_SZ036.pkl', 'rb') as f:
    data = pickle.load(f)

# Find the first terminal state that isn't the end of a session
for i, transition in enumerate(data):
    obs_t, action_t, reward, obs_next, terminal = transition
    if terminal:
        print(f"--- TERMINAL EVENT AT INDEX {i} ---")
        print(f"Step N-1: {data[i-1][0]}") # What was the state BEFORE terminal?
        print(f"Step N  : {obs_t}")      # What was the state WHEN terminal hit?
        break