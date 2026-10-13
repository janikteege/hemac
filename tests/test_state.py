import numpy as np
import pytest
from hemac import HeMAC_v0


class TestState:
    """Test the global state layout and encoding."""

    kwargs = {
        "area_size": (1000, 1000),  # Fits the existing road network.
        "render_mode": None,
        "max_cycles": 10,
        "n_observers": 1,
        "n_drones": 1,
        "n_provisioners": 1,
        "min_obstacles": 1,
        "max_obstacles": 2,
        "drone_config": {
            "discrete_action_space": True,
            "drones_starting_pos": [],
        },
        "poi_config": [
            {
                "speed": 0,
                "variable_speed": False,
                "spawn_mode": "random",
                "dimension": [8, 8],
            },
        ],
    }

    @pytest.fixture
    def env(self):
        """Create a reset environment and close it after each test."""
        env = HeMAC_v0.env(**self.kwargs)
        try:
            env.reset(seed=31)
            yield env
        finally:
            env.close()

    def test_state_output(self, env):
        """Return a float32 vector with the expected shape and bounds."""
        state = env.state()

        # 15 agent + 2 target + 25 action + 3 scenario + 14 node + 49 adjacency features.
        assert state.shape == (108,)
        assert state.dtype == np.float32
        assert env.state_space.contains(state)

    def test_state_content(self, env):
        """Encode known positions and scenario features in the documented order."""
        simulation = env.unwrapped.env
        positions = {
            "observer_0": (250, 750),
            "drone_0": (500, 125),
            "provisioner_0": (400, 650),
        }
        for name, (x, y) in positions.items():
            agent = simulation.agents_list[simulation.agent_name_mapping[name]]
            agent.x, agent.y = x, y
        simulation.goals[0].x, simulation.goals[0].y = (125, 500)
        # Keep one obstacle so the normalized count is always 1 / 2.
        simulation.world.obstacles[:] = simulation.world.obstacles[:1]
        # Base coordinates use the screen's top-left origin, unlike the state.
        simulation.world.base.center = (250, 750)

        state = env.state()

        # First 17 values: three agents with five features each, then target x/y.
        # Agent features: normalized x/y, is_drone, is_observer, is_provisioner.
        np.testing.assert_allclose(
            state[:17],
            [
                # Observer
                0.25,
                0.75,
                0,
                1,
                0,
                # Drone
                0.5,
                0.125,
                1,
                0,
                0,
                # Provisioner
                0.4,
                0.65,
                0,
                0,
                1,
                # Target: normalized x/y
                0.125,
                0.5,
            ],
        )

        # Skip the 25 last-action values (10 observer + 10 drone + 5 provisioner).
        # Scenario starts at 17 + 25 = 42: obstacle fraction, base x, base y.
        np.testing.assert_allclose(state[42:45], [0.5, 0.25, 0.25])

        # Road positions start at 42 + 3 = 45: seven nodes, two coordinates each.
        # Rows follow node IDs 1 through 7, with coordinates divided by 1000.
        np.testing.assert_allclose(
            state[45:59].reshape(7, 2),
            [
                [0.2, 0.2],  # node 1
                [0.2, 0.5],  # node 2
                [0.3, 0.5],  # node 3
                [0.4, 0.5],  # node 4
                [0.4, 0.65],  # node 5
                [0.4, 0.3],  # node 6
                [0.6, 0.65],  # node 7
            ],
        )

        # Adjacency starts at 45 + 14 = 59: a flattened 7-by-7 matrix.
        # Node IDs are one-based; matrix indices are zero-based.
        expected_adjacency = np.zeros((7, 7), dtype=np.float32)
        for start, end in [(1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 7)]:
            expected_adjacency[start - 1, end - 1] = 1
            expected_adjacency[end - 1, start - 1] = 1
        np.testing.assert_array_equal(state[59:].reshape(7, 7), expected_adjacency)

    def test_state_last_actions(self, env):
        """Encode actions as one-hot vectors and clear them on reset."""
        np.testing.assert_array_equal(env.state()[17:42], np.zeros(25, dtype=np.float32))

        # One complete agent cycle; action zero must also produce a one-hot vector.
        for name, action in [("observer_0", 3), ("drone_0", 0), ("provisioner_0", 0)]:
            assert env.agent_selection == name
            env.step(action)

        expected_actions = np.zeros(25, dtype=np.float32)
        expected_actions[[3, 10, 20]] = 1
        np.testing.assert_array_equal(env.state()[17:42], expected_actions)

        env.reset(seed=31)
        np.testing.assert_array_equal(env.state()[17:42], np.zeros(25, dtype=np.float32))
