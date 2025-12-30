#100% ripped from chatgpt for testing the limits and boundaries of
#the actions, observations, etc.
import gymnasium as gym
import numpy as np
import quanser_balance.envs  # triggers register()

from pynput import keyboard

env = gym.make("RotPendEnv-v0", render_mode="human")

action = np.array([0.0], dtype=np.float32)
ACTION_STEP = 0.05
QUIT = {"flag": False}

def key_char(key):
    try:
        return key.char.lower()
    except Exception:
        return None

pressed = set()

def update_action():
    # Decide action from currently held keys
    if "a" in pressed and "d" in pressed:
        action[0] = 0.0
    elif "a" in pressed:
        action[0] = +ACTION_STEP   # choose sign convention you want
    elif "d" in pressed:
        action[0] = -ACTION_STEP
    else:
        action[0] = 0.0

def on_press(key):
    k = key_char(key)
    if k is None:
        return
    if k == "q":
        QUIT["flag"] = True
        return False
    if k in ("a", "d"):
        pressed.add(k)
        update_action()

def on_release(key):
    k = key_char(key)
    if k in ("a", "d"):
        pressed.discard(k)
        update_action()

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    obs, info = env.reset(seed=123, options={"low": -0.1, "high": 0.1})

    low = float(env.action_space.low[0])
    high = float(env.action_space.high[0])

    done = False
    total_reward = 0.0

    while not done and not QUIT["flag"]:
        action[0] = float(np.clip(action[0], low, high))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated

        print("\r\033[K", end="")
        print(
            f"\ract={action[0]:7.3f}  rew={float(reward):8.3f}  total={total_reward:10.3f}",
            end="",
            flush=True,
        )
finally:
    env.close()
    print()
