"""Regression tests for reset-time environment spawning.

Mid-episode POI respawning is not covered here because exercising that path
requires coordinating AEC agent order, movement, and target detection.
"""

from hemac import HeMAC_v0
from hemac.environment.HeMAC import HeMAC
from hemac.environment.world import World


class TestSpawning:
    """Test obstacle, base, and POI placement during environment reset."""

    def test_equal_obstacle_bounds_generate_exact_count(self):
        """Generate the configured count when both obstacle bounds are equal."""
        for obstacle_count in range(8):
            kwargs = {"min_obstacles": obstacle_count, "max_obstacles": obstacle_count}
            env = HeMAC_v0.env(**kwargs)
            env.reset()
            simulation = env.unwrapped.env
            assert isinstance(simulation, HeMAC)
            world = simulation.world
            assert isinstance(world, World)
            assert len(world.obstacles) == obstacle_count
            env.close()

    def test_pois_do_not_overlap_obstacles_after_reset(self):
        """Keep POIs outside obstacles after resets with different seeds."""
        kwargs = {"poi_config": [{"speed": 8}], "min_obstacles": 5, "max_obstacles": 8}
        env = HeMAC_v0.env(**kwargs)
        simulation = env.unwrapped.env
        for seed in range(100):
            env.reset(seed=seed)
            assert all(
                not goal.rect.colliderect(obstacle)
                for goal in simulation.goals
                for obstacle in simulation.world.obstacles
            ), f"seed={seed}"

    def test_base_does_not_overlap_obstacles_after_reset(self):
        """Keep the base outside obstacles after resets with different seeds."""
        kwargs = {"min_obstacles": 5, "max_obstacles": 8}
        env = HeMAC_v0.env(**kwargs)
        simulation = env.unwrapped.env
        for seed in range(100):
            env.reset(seed=seed)
            assert all(not simulation.world.base.colliderect(obstacle) for obstacle in simulation.world.obstacles), (
                f"seed={seed}"
            )
