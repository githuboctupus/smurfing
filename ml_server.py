import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical
from flask import Flask, request, jsonify

# ============================================================
# Directions + heading utils (mirror training logic)
# ============================================================

DIRS = ["up", "down", "left", "right"]


def bs_to_env_y(y_bs, board_h):
    """
    Convert Battlesnake Y (0 = bottom, y+1 = up)
    to env Y (0 = top, y+1 = down), which matches
    the training environment.
    """
    return board_h - 1 - y_bs


def get_heading(body_env):
    """
    body_env: list of {"x": int, "y": int} in ENV coordinates
              (0,0 top-left, y+1 is "down"), head first.

    Uses same logic as training env's _get_heading:
    heading = sign of (head - neck).
    """
    if len(body_env) >= 2:
        hx, hy = body_env[0]["x"], body_env[0]["y"]
        nx, ny = body_env[1]["x"], body_env[1]["y"]
        dx = hx - nx
        dy = hy - ny
        if dx == 0 and dy == -1:
            return "up"
        if dx == 0 and dy == 1:
            return "down"
        if dx == 1 and dy == 0:
            return "right"
        if dx == -1 and dy == 0:
            return "left"
    return "up"


def relative_to_absolute(heading, rel):
    """
    Convert relative action {0: forward, 1: left, 2: right}
    to absolute move "up"/"down"/"left"/"right", using the
    SAME mapping as in your training env.
    """
    if heading == "up":
        return ["up", "left", "right"][rel]
    if heading == "down":
        return ["down", "right", "left"][rel]
    if heading == "left":
        return ["left", "down", "up"][rel]
    if heading == "right":
        return ["right", "up", "down"][rel]
    # fallback
    return "up"


# ============================================================
# Model definition (same as training)
# ============================================================

class SnakePolicyNet(nn.Module):
    def __init__(self, in_channels=20, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 23 * 23, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.policy = nn.Linear(hidden_dim, 3)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        z = self.conv(x)
        z = z.view(z.size(0), -1)
        z = self.fc(z)
        logits = self.policy(z)
        value = self.value(z).squeeze(-1)
        return logits, value


def fix_state_dict_keys(sd):
    """
    Allow loading both new-style and old-style checkpoints.
    If keys are like 'policy_head.*' / 'value_head.*', remap them.
    Safe to call even if keys are already correct.
    """
    if not isinstance(sd, dict):
        return sd
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("policy_head."):
            new_k = k.replace("policy_head.", "policy.")
        elif k.startswith("value_head."):
            new_k = k.replace("value_head.", "value.")
        else:
            new_k = k
        new_sd[new_k] = v
    return new_sd


# ============================================================
# Observation builder (20 x 23 x 23 egocentric)
# ============================================================

def map_to_obs_coords(x_env, y_env, hx_env, hy_env, heading):
    """
    Rotate/translate env board (x_env, y_env) into egocentric (u,v)
    coordinates with our head at (11,11) facing 'up'.

    This matches your training env's _map_to_obs_coords:
    - env coords: (0,0) top-left, y+1 is "down"
    - u,v in [0,22], with head at (11,11)
    """
    dx = x_env - hx_env
    dy = y_env - hy_env

    if heading == "up":
        dxp, dyp = dx, dy
    elif heading == "right":
        dxp, dyp = dy, -dx
    elif heading == "down":
        dxp, dyp = -dx, -dy
    elif heading == "left":
        dxp, dyp = -dy, dx
    else:
        dxp, dyp = dx, dy

    u = 11 + dxp
    v = 11 + dyp

    if 0 <= u < 23 and 0 <= v < 23:
        return int(u), int(v)
    return None


def build_obs(game_state):
    """
    Convert Battlesnake game_state JSON (dict) into a 20x23x23
    torch.FloatTensor, matching the channels used during training.
    """
    board_w = game_state["board"]["width"]
    board_h = game_state["board"]["height"]
    snakes = game_state["board"]["snakes"]
    you = game_state["you"]

    # This model was trained on 11x11 only.
    if board_w != 11 or board_h != 11:
        raise ValueError(
            f"Board size {board_w}x{board_h} not supported (expected 11x11)."
        )

    C, H, W = 20, 23, 23
    board = np.zeros((C, H, W), dtype=np.float32)

    my_body_bs = you["body"]
    if not my_body_bs:
        return torch.from_numpy(board)

    # Convert our body to ENV coords (flip Y).
    my_body_env = [
        {"x": seg["x"], "y": bs_to_env_y(seg["y"], board_h)}
        for seg in my_body_bs
    ]

    # Our head + heading in ENV coords.
    hx_env, hy_env = my_body_env[0]["x"], my_body_env[0]["y"]
    heading = get_heading(my_body_env)

    len_self = len(my_body_env)
    all_lengths = [len(s["body"]) for s in snakes]
    max_len = max(all_lengths) if all_lengths else len_self
    length_diff = len_self - max_len

    # channel 5: playable board area (in env coords)
    for x_env in range(board_w):
        for y_env in range(board_h):
            mapped = map_to_obs_coords(x_env, y_env, hx_env, hy_env, heading)
            if mapped:
                u, v = mapped
                board[5, v, u] = 1.0

    alive_count = len(snakes)
    my_id = you["id"]

    # snakes
    for s in snakes:
        body_bs = s["body"]
        if not body_bs:
            continue

        # Convert this snake’s body to ENV coords.
        body_env = [
            {"x": seg["x"], "y": bs_to_env_y(seg["y"], board_h)}
            for seg in body_bs
        ]

        L = len(body_env)
        double_tail = (
            L >= 2
            and body_env[-1]["x"] == body_env[-2]["x"]
            and body_env[-1]["y"] == body_env[-2]["y"]
        )

        for k, seg_env in enumerate(body_env):
            x_env, y_env = seg_env["x"], seg_env["y"]
            mapped = map_to_obs_coords(x_env, y_env, hx_env, hy_env, heading)
            if not mapped:
                continue
            u, v = mapped

            board[1, v, u] = 1.0              # body mask
            board[2, v, u] = min(k, 255)      # segment index

            if k == 0:
                board[6, v, u] = 1.0          # head mask
                board[0, v, u] = float(s["health"])  # head health layer

            # our snake-specific channels
            if s["id"] == my_id:
                if k == 0:
                    board[17, v, u] = 1.0     # our head
                board[18, v, u] = min(k, 255) # our body index

            # relative size channels (vs us)
            if L >= len_self:
                board[3, v, u] = 1.0          # length >= us
                board[8, v, u] = 1.0
            else:
                board[9, v, u] = 1.0          # length < us

        if double_tail:
            tx_env, ty_env = body_env[-1]["x"], body_env[-1]["y"]
            mapped = map_to_obs_coords(tx_env, ty_env, hx_env, hy_env, heading)
            if mapped:
                u, v = mapped
                board[7, v, u] = 1.0          # double-tail mask

    # food
    for f in game_state["board"]["food"]:
        x_env = f["x"]
        y_env = bs_to_env_y(f["y"], board_h)
        mapped = map_to_obs_coords(x_env, y_env, hx_env, hy_env, heading)
        if mapped:
            u, v = mapped
            board[4, v, u] = 1.0

    # alive-count one-hot channels [10..16]
    idx = max(1, min(7, alive_count))
    ch = 10 + (idx - 1)
    mask = board[5] > 0
    board[ch][mask] = 1.0

    # broadcast length difference
    board[19, :, :] = float(length_diff)

    return torch.from_numpy(board)


# ============================================================
# Load model once at startup
# ============================================================

MODEL_PATH = os.environ.get("BATTLESNAKE_MODEL", "snake_update_00600_wr_0p340.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SnakePolicyNet(20, 256).to(DEVICE)
try:
    if os.path.exists(MODEL_PATH):
        sd = torch.load(MODEL_PATH, map_location=DEVICE)
        try:
            model.load_state_dict(sd)
        except RuntimeError:
            sd = fix_state_dict_keys(sd)
            model.load_state_dict(sd)
        print(f"[Battlesnake] Loaded model from {MODEL_PATH}", flush=True)
    else:
        print(f"[Battlesnake] WARNING: model file {MODEL_PATH} not found, using random weights.", flush=True)
except Exception as e:
    print("[Battlesnake] ERROR loading model:", e, file=sys.stderr, flush=True)

model.eval()
for p in model.parameters():
    p.requires_grad = False

print("[Battlesnake] Successful launch (after model load)", flush=True)


# ============================================================
# Flask app with Battlesnake endpoints
# ============================================================

app = Flask(__name__)


@app.get("/")
def index():
    print("[DEBUG] HIT / route", flush=True)
    """
    Battlesnake info + cosmetics.
    """
    return {
        "apiversion": "1",
        "author": "githuboctupus",
        "color": "#00FF00",
        "head": "pixel",
        "tail": "pixel",
        "version": "1.0.0"
    }


@app.get("/ping")
def ping():
    print("[DEBUG] HIT /ping", flush=True)
    return "ok"


@app.post("/start")
def start():
    data = request.get_json()
    game_id = data["game"]["id"]
    print(f"[Battlesnake] Game started: {game_id}", flush=True)
    return "ok"


@app.post("/move")
def move():
    data = request.get_json()
    try:
        # Build obs (env-coord correct) and run model
        obs = build_obs(data).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(obs)
            dist = Categorical(logits=logits)
            action = dist.sample().item()
    except Exception as e:
        print(f"[Battlesnake] ERROR during move: {e}", file=sys.stderr, flush=True)
        action = 0  # default: forward

    # Compute heading from ENV-coord body
    board_h = data["board"]["height"]
    my_body_bs = data["you"]["body"]
    my_body_env = [
        {"x": seg["x"], "y": bs_to_env_y(seg["y"], board_h)}
        for seg in my_body_bs
    ]
    heading = get_heading(my_body_env)
    move_abs = relative_to_absolute(heading, action)

    print(f"[MOVE] heading={heading}, action={action}, move={move_abs}", flush=True)

    return jsonify({"move": move_abs})


@app.post("/end")
def end():
    data = request.get_json()
    game_id = data["game"]["id"]
    print(f"[Battlesnake] Game ended: {game_id}", flush=True)
    return "ok"


app.run(host="0.0.0.0", port=8000, debug=False)
