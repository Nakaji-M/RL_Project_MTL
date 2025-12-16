import numpy as np
import gymnasium as gym

def get_discrete_actions(env: gym.Env, n_xvals: int = 2, n_yvals: int = 11) -> np.ndarray:
    """
    Generates a discrete action space from the continuous action space of the environment.
    
    Args:
        env: The gymnasium environment.
        n_xvals: Number of discretization steps for the first action dimension (Main Engine).
        n_yvals: Number of discretization steps for the second action dimension (Orientation).
        
    Returns:
        np.ndarray: A matrix of shape (n_actions, 2) where each row is a continuous action vector.
    """
    low_action_value = env.action_space.low
    high_action_value = env.action_space.high
    
    ax_vals = np.linspace(low_action_value[0], high_action_value[0], n_xvals)
    ay_vals = np.linspace(low_action_value[1], high_action_value[1], n_yvals)
    
    actions = np.array(np.meshgrid(ax_vals, ay_vals)).T.reshape(-1, 2)
    return actions

def get_action_size(n_xvals: int = 2, n_yvals: int = 11) -> int:
    return n_xvals * n_yvals

class ObservationNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.0):
        super().__init__(env)
        self.noise_std = noise_std
        
    def observation(self, observation):
        if self.noise_std > 0:
            # The first 6 are continuous, last 2 are boolean (legs)
            # Proposal says: "six of these observations are continuous and will be randomized"
            noise = np.random.normal(0, self.noise_std, size=6)
            observation[:6] += noise
        return observation

