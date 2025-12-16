import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="RL Project")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "sarsa", "ppo"],
        required=True,
        help="Algorithm to run: dqn, sarsa or ppo.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "render"],
        default="train",
        help="Mode: train, eval or render (default: train).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="Number of training episodes for DQN/SARSA (default: 2000).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max steps per episode for DQN/SARSA (default: 1000).",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Number of training timesteps for PPO (default: 200000).",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for saving/loading (e.g. dqn_seed0, ppo_seed1).",
    )

    parser.add_argument(
        "--enable-wind",
        action="store_true",
        help="Enable wind in the environment.",
    )
    parser.add_argument(
        "--wind-power",
        type=float,
        default=0.0,
        help="Wind power (default: 0.0).",
    )
    parser.add_argument(
        "--turbulence-power",
        type=float,
        default=0.0,
        help="Turbulence power (default: 0.0).",
    )
    parser.add_argument(
        "--obs-noise-std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to observations (default: 0.0).",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor gamma (default: 0.99).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4).",
    )

    args = parser.parse_args()
    return args
