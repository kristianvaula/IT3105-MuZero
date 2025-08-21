import gymnasium as gym

class SingleLifeWrapper(gym.Wrapper):
    """
    Gymnasium wrapper to terminate an episode after the first life is lost.
    Assumes the environment provides 'lives' in the info dictionary (standard for ALE).
    """
    def __init__(self, env):
        super().__init__(env)
        self._initial_lives = 0
        self._current_lives = 0
        print("INFO: SingleLifeWrapper initialized.")

    def step(self, action):
        """Steps the environment and checks if a life was lost."""
        observation, reward, terminated, truncated, info = self.env.step(action)

        self._current_lives = info.get('lives', self._current_lives)

        life_lost = False
        if self._initial_lives > 0 and self._current_lives < self._initial_lives:
            # print(f"DEBUG [SingleLifeWrapper]: Life lost ({self._current_lives} < {self._initial_lives}). Terminating episode.")
            life_lost = True

        final_terminated = terminated or life_lost

        final_truncated = truncated

        return observation, reward, final_terminated, final_truncated, info

    def reset(self, **kwargs):
        """Resets the environment and captures the initial number of lives."""
        observation, info = self.env.reset(**kwargs)
        self._initial_lives = info.get('lives', 0)
        self._current_lives = self._initial_lives
        if self._initial_lives == 0:
            print("WARN [SingleLifeWrapper]: Detected 0 initial lives at reset. Wrapper termination logic might not function correctly.")
        # print(f"DEBUG [SingleLifeWrapper]: Reset successful. Initial lives = {self._initial_lives}")
        return observation, info