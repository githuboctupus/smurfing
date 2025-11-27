import os
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical
from flask import Flask, request, jsonify

# ============================================================
# Directions + heading utils (mirror training logic)
# ============================================================

DIRS = ["up", "down", "left", "right"]


def get_heading(body):
    """
    body: list of {"x": int, "y": int} segments from Battlesnake.
    Uses the same logic as the training env's _get_heading:
    heading = sign of (head - neck).
    """
    if len(body) >= 2:
        hx, hy = body[0]["x"], body[0]["y"]
        nx, ny = body[1]["x"], body[1]["y"]
        dx = hx - nx
        dy = hy - ny
        # Match training env:
        # if dx == 0 and dy == -1: "up"
        # if dx == 0 and dy ==  1: "down"
        # if dx ==  1 and dy == 0: "right"
        # if dx == -1 and dy == 0: "left"
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

def map_to_obs_coords(x, y, hx, hy, heading):
    """
    Rotate/translate global board (x,y) into egocentric (u,v)
    coordinates with our head at (11,11) facing 'up'.

    This matches your training env's _map_to_obs_coords.
    """
    dx = x - hx
    dy = y - hy

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

    my_body = you["body"]
    if not my_body:
        return torch.from_numpy(board)

    # Our head + heading
    hx, hy = my_body[0]["x"], my_body[0]["y"]
    heading = get_heading(my_body)

    len_self = len(my_body)
    all_lengths = [len(s["body"]) for s in snakes]
    max_len = max(all_lengths) if all_lengths else len_self
    length_diff = len_self - max_len

    # channel 5: playable board area
    for x in range(board_w):
        for y in range(board_h):
            mapped = map_to_obs_coords(x, y, hx, hy, heading)
            if mapped:
                u, v = mapped
                board[5, v, u] = 1.0

    alive_count = len(snakes)
    my_id = you["id"]

    # snakes
    for s in snakes:
        body = s["body"]
        if not body:
            continue

        L = len(body)
        # "double tail" if last two segments overlap
        double_tail = (
            L >= 2
            and body[-1]["x"] == body[-2]["x"]
            and body[-1]["y"] == body[-2]["y"]
        )

        for k, seg in enumerate(body):
            x, y = seg["x"], seg["y"]
            mapped = map_to_obs_coords(x, y, hx, hy, heading)
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
            tx, ty = body[-1]["x"], body[-1]["y"]
            mapped = map_to_obs_coords(tx, ty, hx, hy, heading)
            if mapped:
                u, v = mapped
                board[7, v, u] = 1.0          # double-tail mask

    # food
    for f in game_state["board"]["food"]:
        x, y = f["x"], f["y"]
        mapped = map_to_obs_coords(x, y, hx, hy, heading)
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
if os.path.exists(MODEL_PATH):
    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        sd = fix_state_dict_keys(sd)
        model.load_state_dict(sd)
    print(f"[Battlesnake] Loaded model from {MODEL_PATH}")
else:
    print(f"[Battlesnake] WARNING: model file {MODEL_PATH} not found, using random weights.")

model.eval()
for p in model.parameters():
    p.requires_grad = False


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
        "author": "githuboctupus",     # TODO: update
        "color": "#00FF00",
        "head": "pixel",
        "tail": "pixel",
        "version": "1.0.0"
    }

@app.get("/ping")
def ping():
    return "ok"

@app.post("/start")
def start():
    data = request.get_json()
    game_id = data["game"]["id"]
    print(f"[Battlesnake] Game started: {game_id}")
    return "ok"


@app.post("/move")
def move():
    data = request.get_json()
    try:
        obs = build_obs(data).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(obs)
            dist = Categorical(logits=logits)
            action = dist.sample().item()
    except Exception as e:
        # If something goes wrong, log and default to "up"
        print(f"[Battlesnake] ERROR during move: {e}")
        action = 0  # forward

    my_body = data["you"]["body"]
    heading = get_heading(my_body)
    move_abs = relative_to_absolute(heading, action)

    # You can add "shout" here if you want debugging text
    return jsonify({"move": move_abs})


@app.post("/end")
def end():
    data = request.get_json()
    game_id = data["game"]["id"]
    print(f"[Battlesnake] Game ended: {game_id}")
    return "ok"

print("Succesful Launch")
