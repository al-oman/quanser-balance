import numpy as np


class RotaryPendulumCurriculum:
    """
    Curriculum for swing-up training.

    Stages:
        0 - Start near upright, learn to balance
        1 - Start at wider angles, learn to recover
        2 - Start from bottom, learn full swing-up

    Call advance() to move to the next stage. The env uses
    get_initial_pend_angle() during reset.
    """

    STAGES = [
        {"pend_angle": 0.0,    "noise": 0.1,  "label": "balance",  "pend_limit": 0.2*np.pi},
        {"pend_angle": 0.0,    "noise": 0.3,  "label": "recover",  "pend_limit": np.pi},
        {"pend_angle": np.pi,  "noise": 0.05, "label": "swing-up", "pend_limit": 3*np.pi},
        {"pend_angle": 0.0,  "noise": 0.1, "label": "test", "pend_limit": 0.2*np.pi},

    ]

    def __init__(self, start_stage=0):
        self.stage = start_stage

    @property
    def current(self):
        return self.STAGES[self.stage]

    @property
    def label(self):
        return self.current["label"]

    def advance(self):
        if self.stage < len(self.STAGES) - 1:
            self.stage += 1
            return True
        return False

    @property
    def termination_limits(self):
        return self.current["pend_limit"]

    def get_initial_pend_angle(self, rng):
        """Returns initial pendulum angle with noise for reset."""
        cfg = self.current
        return cfg["pend_angle"] + rng.uniform(-cfg["noise"], cfg["noise"])