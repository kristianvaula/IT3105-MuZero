from src.gsm.gsm import GameStateManager 
import copy

class RiverraidGSM(GameStateManager):
    def __init__(self, env, seed):
        super().__init__(env)
        self.seed = seed
        self.cache = {}

    def get_initial_state(self):
        state = self.env.reset(seed=self.seed)
        return state

    def get_legal_actions(self, state):
        # Typically, this would query the environment's action space
        return list(range(self.env.action_space.n))

    def get_next_state_reward(self, state, action):
        # Set the environment to the given state, perform the action
        
        cache_key = (state.tobytes(), action)
        if cache_key in self.cache:
            return self.cache[cache_key]
          
        env_clone = copy.deepcopy(self.env)
        next_state, reward, done, _ = env_clone.step(action)
        
        self.cache[cache_key] = (next_state, reward, done)
        
        return next_state, reward, done

    def evaluate_state(self, state):
        return 0.0

    def get_policy(self, state):
        # Return a uniform or learned policy
        return [1.0 / self.env.action_space.n] * self.env.action_space.n

    def is_terminal(self, state):
        # Determine if the state is terminal
        return False
        
