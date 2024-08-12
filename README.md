# better_rl
Simple infrastructure for deep RL experimentation, with a focus on interpretability and reproducibility (seed).

Philosophy: We hope that researchers do not overlook this tool as another package that helps them know *if* their algorithm works, but rather for complex tasks/algorithms and be able to peek (to some point) where an agent can leverage the information it gathers.

The kind of questions, that researchers typically have, that we aim to resolve in order:
1. How does the agent state-visitation distribution change as training progresses and across multiple trajectories (steps separated)?
    - Visualize the replay buffer (state): When you run an agent with a replay buffer, you can see how filled the buffer is as well as what the t-SNE looks like for the buffer. This will help allow researchers to investigate what type of information the agent currently stores. For advanced buffers like PER, it can help investigate what information the agent prioritizes, which may be essential for debugging planning algorithms.
        - What effect do noteworthy, influential states (group state-actions based on TD-error/any other method) have on the policy?

    - Are there repetitive patterns across space and time that result in the observed agent: When you run an agent, you can look at the state-action temporal interaction distribution, projected using t-SNE in real-time. This is typically useful for model-based RL algorithms.
        - Saliency maps on image inputs.

2. How does the agent expect rewards for the state-action distribution?
3. How can we visualize/qualitatively-categorize high-level policies (skills)?

> We anticipate that the best features yet to be built will emerge through iterative feed- back, deployment, and usage in the broader reinforcement learning and interpretability research communities.

To this end, we do not focus on just MBRL/online RL, but transfer from [simple_rl](https://github.com/david-abel/simple_rl/tree/master/simple_rl) for classic MDP based as well as methods specifically targetting continual reinforcement learning. We wish to package this hopefully valuable tool as a python package and subsequently a jupyter notebook widget with real-time updates.

# Policy Visualizer

A Python package for visualizing policy performance and t-SNE of state-action pairs.

## Installation

You can install the package using pip:

```bash
pip install policy_visualizer
```

## Usage

```python
from policy_visualizer import PolicyVisualizer

# Example usage
visualizer = PolicyVisualizer(checkpoint_path='path/to/checkpoints', 
                               env_name='CartPole-v1', 
                               agent_class=YourAgentClass)
visualizer.plot_all()
```