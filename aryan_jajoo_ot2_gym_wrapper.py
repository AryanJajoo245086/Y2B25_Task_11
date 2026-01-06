import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        
        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)
        
        # Workspace limits (meters) - from your previous task
        self.min_x, self.max_x = -0.1873, 0.2534
        self.min_y, self.max_y = -0.1709, 0.2199
        self.min_z, self.max_z = 0.1195, 0.2898
        
        # Define action and observation space
        # They must be gym.spaces objects
        
        # Action space: 3D velocity commands [dx, dy, dz] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Observation space: [pipette_x, pipette_y, pipette_z, goal_x, goal_y, goal_z]
        self.observation_space = spaces.Box(
            low=np.array([self.min_x, self.min_y, self.min_z,
                         self.min_x, self.min_y, self.min_z], dtype=np.float32),
            high=np.array([self.max_x, self.max_y, self.max_z,
                          self.max_x, self.max_y, self.max_z], dtype=np.float32),
            dtype=np.float32
        )
        
        # Keep track of the number of steps
        self.steps = 0
        
        # Additional parameters for control
        self.velocity_scale = 0.5  # Scale velocity commands
        self.control_horizon = 5   # Physics steps per RL step
        
    def reset(self, seed=None):
        # Being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the state of the environment to an initial state
        # Set a random goal position for the agent, consisting of x, y, and z coordinates within the working area
        self.goal_position = np.array([
            np.random.uniform(self.min_x, self.max_x),
            np.random.uniform(self.min_y, self.max_y),
            np.random.uniform(self.min_z, self.max_z)
        ], dtype=np.float32)
        
        # Call the environment reset function
        self.sim.reset(num_agents=1)
        
        # Now we need to process the observation and extract the relevant information
        # Get the pipette position, convert it to a numpy array, and append the goal position
        states = self.sim.get_states()
        robot_key = list(states.keys())[0]  # Get the first (and only) robot
        pipette_pos = np.array(states[robot_key]["pipette_position"], dtype=np.float32)
        
        observation = np.concatenate([pipette_pos, self.goal_position]).astype(np.float32)
        
        # Reset the number of steps
        self.steps = 0

        # For gymnasium
        info = {}
        
        return observation, info
    
    def step(self, action):
        # Execute one time step within the environment
        # Since we are only controlling the pipette position, we accept 3 values for the action 
        # and need to append 0 for the drop action
        action = self.velocity_scale * np.array(action, dtype=np.float32)
        action = [action[0], action[1], action[2], 0]
        
        # Call the environment step function
        # Run for multiple physics steps for smoother control
        for _ in range(self.control_horizon):
            self.sim.run([action])  # Pass as list because sim_class expects a list of actions for multiple agents
        
        # Now we need to process the observation and extract the relevant information
        # Get the pipette position, convert it to a numpy array, and append the goal position
        states = self.sim.get_states()
        robot_key = list(states.keys())[0]  # Get the first (and only) robot
        pipette_pos = np.array(states[robot_key]["pipette_position"], dtype=np.float32)
        
        observation = np.concatenate([pipette_pos, self.goal_position]).astype(np.float32)
        
        # Calculate the reward
        # Use negative distance as reward (agent wants to minimize distance)
        distance = np.linalg.norm(pipette_pos - self.goal_position)
        reward = float(-distance)
        
        # Check if the task has been completed
        # Distance threshold: 2mm (0.002m) - reasonable for pipette tip precision
        # This is smaller than typical plant features but achievable with good control
        if distance < 0.002:
            terminated = True
            # Give the agent a positive reward for completing the task
            reward += 10.0
        else:
            terminated = False
            
        # Check if the episode should be truncated
        # Truncate if we've exceeded the maximum number of steps
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
            
        info = {}  # We don't need to return any additional information
        
        # Increment the number of steps
        self.steps += 1
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()