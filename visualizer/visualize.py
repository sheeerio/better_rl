import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from sklearn.manifold import TSNE
import inspect

class PolicyVisualizer:
    def __init__(self, checkpoint_path, env_name, agent_class, perplexity=100, max_steps=1000):
        """
        Initialize the PolicyVisualizer.

        :param checkpoint_path: Path to the folder containing checkpoints
        :param env_name: Name of the environment
        :param agent_class: Class of the agent
        :param perplexity: Perplexity parameter for t-SNE
        :param max_steps: Maximum steps for each episode in performance evaluation
        """
        self.checkpoint_path = checkpoint_path
        self.env = make_vec_env(env_name)
        self.agent_class = agent_class
        self.perplexity = perplexity
        self.max_steps = max_steps
        self.agent_loader = self.detect_and_set_model_loading_method()

    def detect_model_loading_methods(self):
        model_loading_methods = []
        
        for method_name, method in inspect.getmembers(self.agent_class, predicate=inspect.isfunction):
            if 'load' in method_name.lower():
                method_source = inspect.getsource(method)
                if 'torch.load' in method_source or 'jax.numpy.load' in method_source or 'load_state_dict' in method_source:
                    model_loading_methods.append(method_name)
        
        return model_loading_methods

    def detect_and_set_model_loading_method(self):
        loading_methods = self.detect_model_loading_methods()
        if not loading_methods:
            raise ValueError("No load model function found in agent class.")
        
        # Use the first detected loading method
        loading_method = loading_methods[0]
        
        def agent_loader(checkpoint_file):
            agent = self.agent_class()
            getattr(agent, loading_method)(checkpoint_file)
            return agent
        
        return agent_loader

    def collect_state_action_pairs(self, agent, steps=20000):
        """Collect state-action pairs from the environment using the given agent."""
        states, actions = [], []
        obs = self.env.reset()
        for _ in range(steps):
            action = agent.predict(obs)
            next_obs, _, done, _ = self.env.step(action)
            states.append(obs)
            actions.append(action)
            obs = next_obs
            if done:
                obs = self.env.reset()
        return np.hstack((np.array(states), np.array(actions)))

    def load_checkpoints(self):
        """Load all checkpoints from the checkpoint path."""
        checkpoints = {}
        for filename in os.listdir(self.checkpoint_path):
            if filename.endswith((".ckpt", ".pth", ".pt")):
                steps = int(filename.split("_")[-1].split(".")[0])
                agent = self.agent_loader(os.path.join(self.checkpoint_path, filename))
                checkpoints[steps] = agent
        return checkpoints

    def compute_tsne(self, state_action_pairs):
        """Compute t-SNE for the given state-action pairs."""
        tsne = TSNE(n_components=2, random_state=42, perplexity=self.perplexity)
        return tsne.fit_transform(state_action_pairs)

    def plot_tsne_results(self, tsne_results, title, color):
        """Plot t-SNE results."""
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5, c=color, label=title)

    def plot_all(self, steps_to_plot=None):
        """Plot t-SNE visualizations for all or specified checkpoints."""
        checkpoints = self.load_checkpoints()
        if steps_to_plot is None:
            steps_to_plot = sorted(checkpoints.keys())

        plt.figure(figsize=(10, 10))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(steps_to_plot)))

        for step, color in zip(steps_to_plot, colors):
            agent = checkpoints[step]
            state_action_pairs = self.collect_state_action_pairs(agent)
            tsne_results = self.compute_tsne(state_action_pairs)
            self.plot_tsne_results(tsne_results, f'Step {step}', color)

        plt.legend()
        plt.title(f't-SNE of State-Action Visitation (Perplexity: {self.perplexity})')
        plt.xlabel('t-SNE dim 1')
        plt.ylabel('t-SNE dim 2')
        plt.show()

    def plot_performance(self, episodes=10):
        """Plot the performance of the agent at each checkpoint."""
        checkpoints = self.load_checkpoints()
        steps = sorted(checkpoints.keys())
        rewards = []

        for step in steps:
            agent = checkpoints[step]
            episode_rewards = []
            for _ in range(episodes):
                obs = self.env.reset()
                total_reward = 0
                for _ in range(self.max_steps):
                    action = agent.predict(obs)
                    obs, reward, done, _ = self.env.step(action)
                    total_reward += reward
                    if done:
                        break
                episode_rewards.append(total_reward)
            rewards.append(np.mean(episode_rewards))

        plt.figure(figsize=(10, 5))
        plt.plot(steps, rewards, marker='o')
        plt.title(f'Performance over training steps (Averaged over {episodes} episodes)')
        plt.xlabel('Training steps')
        plt.ylabel(f'Average total reward (max {self.max_steps} steps)')
        plt.show()