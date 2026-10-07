from hemac import HeMAC_v0
from hemac.environment.HeMAC import HeMAC
from hemac.environment.drone import Drone
from hemac.environment.observer import Observer
from hemac.environment.provisioner import Provisioner


class TestSeeding:
    kwargs = {"n_observers": 3, "n_drones": 3, "poi_config": [{"speed": 8}]}

    @staticmethod
    def snapshot(env):
        """Get a snapshot of the current state of the environment. Is comparible via == with other snapshots.

        Beware that getting a snapshot actually changes the environment
        due to the observers observation setting the communication. So one should
        always "snapshot equally" on the envs to compare.
        """
        simulation: HeMAC = env.unwrapped.env
        return {
            "rewards": dict(env.rewards),
            "terminations": dict(env.terminations),
            "truncations": dict(env.truncations),
            "agent_selection": env.agent_selection,
            "pois": tuple((poi.x, poi.y, poi.orientation) for poi in simulation.goals),
            "agents": {
                "drones": {
                    agent_str: {
                        "position": (agent.x, agent.y),
                        "velocity": (agent.vx, agent.vy, agent.orientation),
                        "carried_targets": agent.carried_targets,
                        "observation": env.observe(agent_str).tolist(),
                    }
                    for agent_str, agent in zip(simulation.agents, simulation.agents_list)
                    if isinstance(agent, Drone)
                },
                "observers": {
                    agent_str: {
                        "position": (agent.x, agent.y),
                        "orientation": agent.orientation,
                        # NOTE: this actually modifies the env. not ideal, but ok.
                        "observation": env.observe(agent_str).tolist(),
                    }
                    for agent_str, agent in zip(simulation.agents, simulation.agents_list)
                    if isinstance(agent, Observer)
                },
                "provisioners": {
                    agent_str: {
                        "position": (agent.x, agent.y),
                        "orientation": agent.orientation,
                        "observation": env.observe(agent_str).tolist(),
                    }
                    for agent_str, agent in zip(simulation.agents, simulation.agents_list)
                    if isinstance(agent, Provisioner)
                },
            },
            "world": {
                "base": simulation.world.base.center,
                "obstacles": tuple(
                    (obstacle.x, obstacle.y, obstacle.width, obstacle.height) for obstacle in simulation.world.obstacles
                ),
            },
        }

    def test_simple_reset(self):
        env1 = HeMAC_v0.env(**self.kwargs)
        env2 = HeMAC_v0.env(**self.kwargs)
        env1.reset(42)
        env2.reset(42)
        snapshot1 = self.snapshot(env1)
        snapshot2 = self.snapshot(env2)
        assert snapshot1 == snapshot2

    def test_double_reset(self):
        env = HeMAC_v0.env(**self.kwargs)
        env.reset(42)
        expected = self.snapshot(env)

        for _ in range(20):
            unwrapped = env.unwrapped.env
            current_agent = unwrapped.agents_list[unwrapped.agent_name_mapping[env.agent_selection]]
            # NOTE: taking a random action may not be ideal for a reproducible test
            env.step(current_agent.action_space.sample())

        env.reset(42)
        snapshot = self.snapshot(env)
        assert snapshot == expected

    def test_steps(self):
        env1 = HeMAC_v0.env(**self.kwargs)
        env2 = HeMAC_v0.env(**self.kwargs)
        env1.reset(42)
        env2.reset(42)

        for _ in range(20):
            unwrapped1 = env1.unwrapped.env
            unwrapped2 = env2.unwrapped.env

            current_agent1 = unwrapped1.agents_list[unwrapped1.agent_name_mapping[env1.agent_selection]]
            current_agent2 = unwrapped2.agents_list[unwrapped2.agent_name_mapping[env2.agent_selection]]

            assert type(current_agent1) is type(current_agent2)

            # NOTE: taking a random action may not be ideal for a reproducible test
            action = current_agent1.action_space.sample()

            env1.step(action)
            env2.step(action)
            snapshot1 = self.snapshot(env1)
            snapshot2 = self.snapshot(env2)
            assert snapshot1 == snapshot2

    def test_different_seeds(self):
        env1 = HeMAC_v0.env(**self.kwargs)
        env2 = HeMAC_v0.env(**self.kwargs)
        env1.reset(42)
        env2.reset(43)
        snapshot1 = self.snapshot(env1)
        snapshot2 = self.snapshot(env2)
        assert snapshot1 != snapshot2
