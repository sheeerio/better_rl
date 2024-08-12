import unittest
from unittest.mock import MagicMock, patch
from visualizer import PolicyVisualizer

class TestPolicyVisualizer(unittest.TestCase):

    def setUp(self):
        """Setup the test case."""
        self.checkpoint_path = 'path/to/checkpoints'
        self.env_name = 'CartPole-v1'
        self.agent_class = MagicMock()  # Use MagicMock to simulate the agent class
        self.visualizer = PolicyVisualizer(
            checkpoint_path=self.checkpoint_path,
            env_name=self.env_name,
            agent_class=self.agent_class
        )

    def test_detect_model_loading_methods(self):
        """Test detection of model loading methods."""
        # Mock the inspect.getsource to simulate method source code
        with patch('inspect.getsource') as mock_getsource:
            mock_getsource.return_value = 'torch.load'  # Simulate that 'torch.load' is in method source
            self.visualizer.agent_class.load_model = MagicMock()
            methods = self.visualizer.detect_model_loading_methods()
            self.assertIn('load_model', methods, "Model loading method not detected")

    def test_no_model_loading_methods(self):
        """Test behavior when no model loading methods are found."""
        with patch('inspect.getsource', return_value='') as mock_getsource:
            self.visualizer.agent_class.some_other_method = MagicMock()
            with self.assertRaises(ValueError) as context:
                self.visualizer.detect_and_set_model_loading_method()
            self.assertEqual(str(context.exception), "No load model function found in agent class.")

    def test_collect_state_action_pairs(self):
        """Test collection of state-action pairs."""
        mock_agent = MagicMock()
        mock_agent.predict.return_value = [0]  # Mock predict method
        mock_agent.predict.side_effect = [1, 0, 1]  # Mock different actions

        # Mock environment methods
        self.visualizer.env.reset = MagicMock(return_value=[0])
        self.visualizer.env.step = MagicMock(side_effect=[
            ([0], 0, False, None), ([0], 1, True, None), ([0], 0, False, None)
        ])

        state_action_pairs = self.visualizer.collect_state_action_pairs(mock_agent, steps=3)
        expected_pairs = np.hstack((np.array([[0], [0], [0]]), np.array([[0], [1], [0]])))
        np.testing.assert_array_equal(state_action_pairs, expected_pairs, "State-action pairs not collected correctly")

    def test_load_checkpoints(self):
        """Test loading of checkpoints."""
        self.visualizer.agent_loader = MagicMock(return_value=MagicMock())

        # Mock os.listdir
        with patch('os.listdir', return_value=['model_1000.ckpt', 'model_2000.ckpt']):
            checkpoints = self.visualizer.load_checkpoints()
            self.assertEqual(len(checkpoints), 2, "Checkpoints not loaded correctly")

    def test_compute_tsne(self):
        """Test t-SNE computation."""
        state_action_pairs = np.array([[0, 1], [2, 3], [4, 5]])
        tsne_results = self.visualizer.compute_tsne(state_action_pairs)
        self.assertEqual(tsne_results.shape[1], 2, "t-SNE results do not have the correct shape")

    def test_plot_tsne_results(self):
        """Test plotting of t-SNE results."""
        tsne_results = np.array([[0, 1], [2, 3]])
        try:
            self.visualizer.plot_tsne_results(tsne_results, 'Test', 'blue')
        except Exception as e:
            self.fail(f"plot_tsne_results raised an exception: {e}")

    def test_plot_all(self):
        """Test plotting of all checkpoints."""
        self.visualizer.load_checkpoints = MagicMock(return_value={1000: MagicMock(), 2000: MagicMock()})
        self.visualizer.collect_state_action_pairs = MagicMock(return_value=np.array([[0, 1], [2, 3]]))
        self.visualizer.compute_tsne = MagicMock(return_value=np.array([[0, 1], [2, 3]]))
        
        try:
            self.visualizer.plot_all(steps_to_plot=[1000, 2000])
        except Exception as e:
            self.fail(f"plot_all raised an exception: {e}")

    def test_plot_performance(self):
        """Test plotting of agent performance."""
        self.visualizer.load_checkpoints = MagicMock(return_value={1000: MagicMock()})
        self.visualizer.env.reset = MagicMock(return_value=[0])
        self.visualizer.env.step = MagicMock(side_effect=[([0], 1, False, None)] * 10)
        
        try:
            self.visualizer.plot_performance(episodes=1)
        except Exception as e:
            self.fail(f"plot_performance raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()