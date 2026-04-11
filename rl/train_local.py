"""
Train the local navigation policy using PPO (stable-baselines3).

Usage:
    python -m rl.train_local                          # defaults
    python -m rl.train_local --env simple --steps 500000
    python -m rl.train_local --env office --diff 2 --steps 1000000

The trained model is saved to  rl/weights/<env>_diff<d>_local_nav.zip
and can be loaded in prm_rl.py for Monte Carlo edge validation.
"""

import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from rl.rl_env import LocalNavEnv


def make_env(env_name, difficulty, gui=False):
    """Factory that returns a no-arg callable for make_vec_env."""
    def _init():
        return LocalNavEnv(
            obstacle_config=env_name,
            difficulty=difficulty,
            gui=gui,
        )
    return _init


def train(env_name="simple", difficulty=0, total_timesteps=500_000,
          n_envs=4, seed=42, save_dir="rl/weights"):

    os.makedirs(save_dir, exist_ok=True)

    print(f"Training local nav policy: env={env_name}, diff={difficulty}")
    print(f"  timesteps={total_timesteps}, parallel_envs={n_envs}")

    # ── Vectorised training envs ─────────────────────────────────────
    train_envs = make_vec_env(
        make_env(env_name, difficulty),
        n_envs=n_envs,
        seed=seed,
    )

    # ── Eval env (single, for callbacks) ─────────────────────────────
    eval_env = make_vec_env(
        make_env(env_name, difficulty),
        n_envs=1,
        seed=seed + 1000,
    )

    # ── PPO with MLP policy ──────────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        train_envs,
        verbose=1,
        seed=seed,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
    )

    # ── Eval callback: saves best model ──────────────────────────────
    model_name = f"{env_name}_diff{difficulty}_local_nav"
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=max(total_timesteps // (20 * n_envs), 1000),
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    # ── Train ────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(save_dir, model_name)
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}.zip")

    # Also save the best model with a clear name
    best_src = os.path.join(save_dir, "best_model.zip")
    best_dst = os.path.join(save_dir, f"{model_name}_best.zip")
    if os.path.exists(best_src):
        import shutil
        shutil.copy2(best_src, best_dst)
        print(f"Best model copied to {best_dst}")

    train_envs.close()
    eval_env.close()

    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Train local navigation RL policy")
    parser.add_argument("--env", default="simple",
                        choices=["simple", "office", "cityscape"])
    parser.add_argument("--diff", type=int, default=0,
                        choices=[0, 1, 2, 3])
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", default="rl/weights")
    args = parser.parse_args()

    train(
        env_name=args.env,
        difficulty=args.diff,
        total_timesteps=args.steps,
        n_envs=args.n_envs,
        seed=args.seed,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()