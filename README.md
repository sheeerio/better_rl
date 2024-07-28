# better_rl
Simple infrastructure for deep RL experimentation, with a focus on better interpretability.

What it does - 

When you run an agent with a replay buffer, you can see how filled the buffer is as well as what the t-SNE looks like for the buffer. This will help allow researchers to investigate what type of information the agent currently stores. For advanced buffers like PER, it can help investigate what information the agent prioritizes, which may be essential for debugging planning algorithms.

When you run an agent, you can look at the state-action temporal interaction distribution, projected using t-SNE in real-time. This is typically useful for model-based RL algorithms.

The kind of questions, that researchers typically have, that we aim to resolve:
- How does the agent state-visitation distribution change as training progresses?
- What effect do noteworthy, influential states have on the policy?
- Are there repetitive patterns across space and time that result in the observed agent
behavior?
- How does the agent expect rewards for the state-action distribution?