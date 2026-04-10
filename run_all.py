"""
Run all three PRM simulations in sequence.
Usage:
    python run_all.py              # headless (DIRECT) — matplotlib plots only
    python run_all.py --gui        # PyBullet GUI + matplotlib
    python run_all.py --sim 1      # run only simulation 1 (simple)
    python run_all.py --sim 2      # run only simulation 2 (office)
    python run_all.py --sim 3      # run only simulation 3 (cityscape)
"""

import argparse
import sim_simple
import sim_office
import sim_cityscape


SIMS = {
    1: ("Simple 2D Plane",  sim_simple.main),
    2: ("Office Floor Plan", sim_office.main),
    3: ("3D Cityscape",     sim_cityscape.main),
}


def main():
    parser = argparse.ArgumentParser(description="PRM PyBullet Simulations")
    parser.add_argument("--gui", action="store_true",
                        help="Open PyBullet GUI for each simulation")
    parser.add_argument("--sim", type=int, choices=[1, 2, 3], default=None,
                        help="Run a single simulation (1, 2, or 3)")
    args = parser.parse_args()

    targets = [args.sim] if args.sim else [1, 2, 3]

    for sim_id in targets:
        name, fn = SIMS[sim_id]
        print(f"\n{'='*60}")
        print(f"  Simulation {sim_id}: {name}")
        print(f"{'='*60}\n")
        fn(use_gui=args.gui)

    print("\nAll simulations complete.")


if __name__ == "__main__":
    main()
