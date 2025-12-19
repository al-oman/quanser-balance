from __future__ import annotations
import argparse

def run_sim() -> None:
    parser = argparse.ArgumentParser(prog="run-sim")
    parser.add_argument("--x0", type=float, default=0.)
    parser.add_argument("--v0", type=float, default=0.)
    parser.add_argument("--th0", type=float, default=0.)
    args = parser.parse_args()

    from quanser_balance.sim.run import run
    run()
