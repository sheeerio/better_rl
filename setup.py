from setuptools import setup, find_packages

setup(
    name='policy_visualizer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'stable-baselines3',
        'scikit-learn',
        'torch',
        'jax',
    ],
    author='Gunbir Singh Baveja',
    author_email='gbaveja@student.ubc.ca',
    description='A package for visualizing policy performance and t-SNE of state-action pairs',
    url='https://github.com/sheeerio/better_rl',
)