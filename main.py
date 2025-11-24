import json
import math
import random
import copy
from collections import deque

# Flask imports for hosting Battlesnake
from flask import Flask, request, jsonify

# Global search parameters (tune for speed/strength)
I_DEFAULT = 500    # number of simulations per turn
D_DEFAULT = 6     # maximum depth of a single simulation

# Risk-averse root scoring
# alpha ~ 0.3 = very risk averse, ~0.6 = more opportunistic
ALPHA_SAFE = 0.6
ALPHA_DANGER = 0.3

# How to handle unvisited DecisionNodes at root:
# True  -> do a one-step simulation (my move + random enemies) to get a value
# False -> set a prior based on visited siblings' values
USE_ONE_STEP_FALLBACK_FOR_UNVISITED = True

# Random seed (set to None for fully random behavior)
r0 = None
if r0 is not None:
    random.seed(r0)


DIRS = ["up", "down", "left", "right"]
DELTA = {
    "up":    (0, 1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": (1, 0),
}

def in_bounds(x, y, w, h):
    return 0 <= x < w and 0 <= y < h

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def is_stacked_body(body):
    """
    Returns True if the snake body has duplicate coordinates,
    which means it's "stacked" and its tail will NOT move this turn.
    """
    seen = set()
    for x, y in body:
        if (x, y) in seen:
            return True
        seen.add((x, y))
    return False


def s_from_json(j):
    """
    Convert Battlesnake JSON game state into internal state dict.

    Internal state format:
      s = {
        "w": width,
        "h": height,
        "sn": [
          {
            "id": snake_id (str),
            "b": [(x_head, y_head), ..., (x_tail, y_tail)],
            "hp": health (int),
            "alive": bool,
          }, ...
        ],
        "yi": index of our snake in sn,
        "f": [(fx, fy), ...],  # food positions
        "init_len": initial length of our snake at root
      }
    """
    board = j["board"]
    you_id = j["you"]["id"]

    w = board["width"]
    h = board["height"]

    snakes = []
    yi = None

    for idx, s_json in enumerate(board["snakes"]):
        snake_id = s_json["id"]
        body = [(seg["x"], seg["y"]) for seg in s_json["body"]]
        hp = s_json["health"]
        alive = True  # JSON only provides living snakes here

        snake = {
            "id": snake_id,
            "b": body,
            "hp": hp,
            "alive": alive,
        }
        snakes.append(snake)

        if snake_id == you_id:
            yi = idx

    food = [(f["x"], f["y"]) for f in board["food"]]

    s = {
        "w": w,
        "h": h,
        "sn": snakes,
        "yi": yi,
        "f": food,
    }
    s["init_len"] = len(s["sn"][s["yi"]]["b"])
    return s

def state_copy(s):
    """Deep copy of state dict."""
    return copy.deepcopy(s)

def is_alive(s, idx):
    """Return True if snake idx is alive."""
    return 0 <= idx < len(s["sn"]) and s["sn"][idx]["alive"]


def compute_occupied_for_collision(s):
    """
    Compute occupied cells for body-collision purposes.

    Rule:
      - All body segments except tails are blocked.
      - Tail of a NON-stacked snake is considered vacating, so not blocked.
      - Tail of a stacked snake stays put, so is blocked.
    """
    occ = set()
    for snake in s["sn"]:
        if not snake["alive"]:
            continue
        body = snake["b"]
        if not body:
            continue
        tail = body[-1]
        stacked = is_stacked_body(body)
        # Add all body segments except tail
        for seg in body[:-1]:
            occ.add(seg)
        # Tail is blocked only if stacked
        if stacked:
            occ.add(tail)
    return occ

def safe_moves(s, idx):
    """
    Return list of safe moves for snake idx, based on:
      - Board bounds
      - Not colliding with bodies (with tail-vacate logic)
    Head-to-head risk is handled in evaluation, not here.
    """
    snake = s["sn"][idx]
    if not snake["alive"]:
        return []

    w, h = s["w"], s["h"]
    head_x, head_y = snake["b"][0]

    # Occupied cells for body collisions
    occ = compute_occupied_for_collision(s)

    safe = []
    for mv in DIRS:
        dx, dy = DELTA[mv]
        nx, ny = head_x + dx, head_y + dy
        if not in_bounds(nx, ny, w, h):
            continue
        # If this is our own tail, and we are NOT stacked, treat it as free.
        if (nx, ny) in occ:
            # Check special case: our own tail and non-stacked
            body = snake["b"]
            if len(body) > 0 and (nx, ny) == body[-1] and not is_stacked_body(body):
                # allowed – tail vacates
                safe.append(mv)
            else:
                # blocked
                continue
        else:
            safe.append(mv)
    return safe


def step_state_inplace(s, moves):
    """
    Advance the game state by one full turn:
      - moves: dict snake_id -> "up"/"down"/"left"/"right"
      - updates positions, food, and alive flags in-place.

    This is an approximate but reasonably faithful implementation
    of Battlesnake rules for standard play.
    """
    w, h = s["w"], s["h"]
    snakes = s["sn"]
    food = s["f"]
    food_set = set(food)

    # 1) Compute tentative new heads and whether each snake will eat
    new_heads = {}
    will_eat = {}
    for snake in snakes:
        if not snake["alive"]:
            continue
        sid = snake["id"]
        body = snake["b"]
        if not body:
            snake["alive"] = False
            continue

        head_x, head_y = body[0]
        mv = moves.get(sid, "up")  # default if missing, shouldn't happen
        dx, dy = DELTA[mv]
        nx, ny = head_x + dx, head_y + dy

        new_heads[sid] = (nx, ny)
        will_eat[sid] = (nx, ny) in food_set

    # 2) Build tentative new bodies (before collision resolution)
    new_bodies = {}
    for snake in snakes:
        if not snake["alive"]:
            continue
        sid = snake["id"]
        body = snake["b"]
        if sid not in new_heads:
            snake["alive"] = False
            continue
        nx, ny = new_heads[sid]
        eat = will_eat[sid]

        if eat:
            # Grow: add new head, keep tail
            new_body = [(nx, ny)] + body
        else:
            # Normal move: add new head, drop tail
            new_body = [(nx, ny)] + body[:-1]
        new_bodies[sid] = new_body

    # 3) Out-of-bounds and self/body collisions
    # First, compute all body cells (excluding heads) for collision checks
    body_cells = set()
    for snake in snakes:
        if not snake["alive"]:
            continue
        sid = snake["id"]
        nb = new_bodies.get(sid)
        if not nb:
            continue
        for (x, y) in nb[1:]:
            body_cells.add((x, y))

    # Mark snakes dead for wall or body collision
    for snake in snakes:
        if not snake["alive"]:
            continue
        sid = snake["id"]
        nb = new_bodies.get(sid)
        if not nb:
            snake["alive"] = False
            continue
        hx, hy = nb[0]

        # Wall collision
        if not in_bounds(hx, hy, w, h):
            snake["alive"] = False
            continue

        # Body collision (into any body cell, including its own)
        if (hx, hy) in body_cells:
            snake["alive"] = False
            continue

    # 4) Head-to-head collisions
    heads_map = {}
    for snake in snakes:
        if not snake["alive"]:
            continue
        sid = snake["id"]
        nb = new_bodies.get(sid)
        if not nb:
            snake["alive"] = False
            continue
        hx, hy = nb[0]
        heads_map.setdefault((hx, hy), []).append(snake)

    for pos, snakes_here in heads_map.items():
        if len(snakes_here) <= 1:
            continue
        # Multiple heads in same cell: kill shorter ones; if tie, all die
        max_len = max(len(s["b"]) for s in snakes_here)
        longest = [s for s in snakes_here if len(s["b"]) == max_len]
        if len(longest) == 1:
            # All others die
            for s2 in snakes_here:
                if s2 is not longest[0]:
                    s2["alive"] = False
        else:
            # All die on equal length tie
            for s2 in snakes_here:
                s2["alive"] = False

    # 5) Apply new bodies, update health, remove eaten food
    eaten_food = set()
    for snake in snakes:
        if not snake["alive"]:
            continue
        sid = snake["id"]
        nb = new_bodies.get(sid)
        if not nb:
            snake["alive"] = False
            continue

        # Update body
        snake["b"] = nb

        # Health
        snake["hp"] -= 1
        if will_eat.get(sid, False):
            snake["hp"] = 100
            eaten_food.add(nb[0])

        if snake["hp"] <= 0:
            snake["alive"] = False

    # Remove eaten food
    if eaten_food:
        s["f"] = [f for f in food if f not in eaten_food]


def compute_areas(s):
    """
    Compute (my_area, max_enemy_area) via multi-source BFS.

    Approximation:
      - Bodies (with stacked-tail logic) are treated as walls.
      - We flood-fill from all heads simultaneously.
      - Each cell is "owned" by the snake that can reach it earliest.
      - Ties are broken by length (longer snake wins).

    Returns:
      my_area:         number of cells owned by our snake
      max_enemy_area:  maximum area owned by any single enemy
    """
    w, h = s["w"], s["h"]
    snakes = s["sn"]
    yi = s["yi"]
    my_id = yi

    # Occupied as per body collisions
    occ = compute_occupied_for_collision(s)

    dist = [[math.inf] * w for _ in range(h)]
    owner = [[-1] * w for _ in range(h)]

    # Initial queue: all alive snake heads
    q = deque()
    lengths = [len(sn["b"]) for sn in snakes]

    for idx, sn in enumerate(snakes):
        if not sn["alive"]:
            continue
        if not sn["b"]:
            continue
        hx, hy = sn["b"][0]
        if not in_bounds(hx, hy, w, h):
            continue
        if (hx, hy) in occ:
            continue
        dist[hy][hx] = 0
        owner[hy][hx] = idx
        q.append((hx, hy, idx))

    # BFS
    while q:
        x, y, idx = q.popleft()
        d0 = dist[y][x]
        for dx, dy in DELTA.values():
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny, w, h):
                continue
            if (nx, ny) in occ:
                continue
            nd = d0 + 1
            if nd < dist[ny][nx]:
                dist[ny][nx] = nd
                owner[ny][nx] = idx
                q.append((nx, ny, idx))
            elif nd == dist[ny][nx] and owner[ny][nx] != idx:
                # Tie: break by length (longer snake wins)
                current_owner = owner[ny][nx]
                if current_owner == -1:
                    owner[ny][nx] = idx
                else:
                    if lengths[idx] > lengths[current_owner]:
                        owner[ny][nx] = idx

    area_counts = [0] * len(snakes)
    for y in range(h):
        for x in range(w):
            o = owner[y][x]
            if o >= 0:
                area_counts[o] += 1

    my_area = area_counts[my_id] if 0 <= my_id < len(area_counts) else 0
    enemy_areas = [area_counts[i] for i in range(len(area_counts)) if i != my_id]
    max_enemy_area = max(enemy_areas) if enemy_areas else 0

    return my_area, max_enemy_area


def evaluate_state(s):
    """
    Scalar evaluation of the state from our perspective.

    Priorities:
      1. Death  -> huge negative
      2. Win    -> huge positive
      3. Avoid starvation / low HP
      4. Get food when low
      5. Prefer safe head positioning (avoid adjacency to bigger heads)
      6. Then maximize territory / control
    """
    yi = s["yi"]
    snakes = s["sn"]
    me = snakes[yi]

    # If we're dead, huge negative
    if not me["alive"]:
        return -1e9

    # Count alive snakes
    alive_indices = [i for i, sn in enumerate(snakes) if sn["alive"]]
    if len(alive_indices) == 1 and alive_indices[0] == yi:
        # Only us alive -> huge positive (win)
        return 1e8

    my_body = me["b"]
    my_len = len(my_body)
    my_hp = me["hp"]
    my_head = my_body[0]

    # Territory
    my_area, max_enemy_area = compute_areas(s)

    # Food bonus (stronger when low HP)
    food_bonus = 0.0
    for fx, fy in s["f"]:
        d = manhattan(my_head, (fx, fy))
        if d == 0:
            food_bonus += 6.0
        elif d <= 6:
            food_bonus += (6 - d)

    # HP weight: stronger when low
    if my_hp <= 40:
        health_weight = 0.06
        food_weight = 0.35
    else:
        health_weight = 0.03
        food_weight = 0.12

    # Length gain since root
    init_len = s.get("init_len", my_len)
    length_gain = my_len - init_len
    effective_len_weight = 0.25 if my_hp <= 40 else 0.15

    # Simple starvation risk: if HP is very low and no food -> penalty
    starvation_penalty = 0.0
    if my_hp <= 25:
        if not s["f"]:
            starvation_penalty -= 40.0

    # Head danger: adjacency to bigger/equal heads
    head_danger_penalty = 0.0
    for i, sn in enumerate(snakes):
        if i == yi or not sn["alive"]:
            continue
        other_head = sn["b"][0]
        other_len = len(sn["b"])
        if manhattan(my_head, other_head) == 1 and other_len >= my_len:
            head_danger_penalty -= 25.0

    # Territory weights
    space_weight = 0.5
    control_weight = 0.25

    space_term = space_weight * my_area
    control_term = control_weight * (my_area - max_enemy_area)
    health_term = health_weight * my_hp
    food_term = food_weight * food_bonus
    length_term = effective_len_weight * length_gain

    score = (
        space_term +
        control_term +
        health_term +
        food_term +
        length_term +
        starvation_penalty +
        head_danger_penalty
    )
    return score


def random_enemy_moves(s, my_idx):
    """
    For each enemy snake, pick a random safe move if possible.
    If no safe move exists, pick any direction (may die).
    Returns: dict snake_id -> move
    """
    snakes = s["sn"]
    moves = {}
    for i, sn in enumerate(snakes):
        if not sn["alive"]:
            continue
        if i == my_idx:
            continue
        sid = sn["id"]
        sm = safe_moves(s, i)
        if sm:
            mv = random.choice(sm)
        else:
            mv = random.choice(DIRS)
        moves[sid] = mv
    return moves


def make_game_node(state):
    return {
        "type": "game",
        "state": state,
        "children": {},      # move (str) -> DecisionNode
        "n": 0,              # visit count
        "sum_v": 0.0,        # sum of values
        "min_v": float("inf") # worst value seen
    }

def make_decision_node(move, parent_game_node):
    return {
        "type": "decision",
        "move": move,
        "parent": parent_game_node,
        "children": [],             # list of GameNodes (samples for this move)
        "n": 0,
        "sum_v": 0.0,
        "min_v": float("inf"),
    }

def ucb_score(node, parent_visits, c=1.4):
    """Upper Confidence Bound for tree selection."""
    if node["n"] == 0:
        return float("inf")
    mean = node["sum_v"] / node["n"]
    return mean + c * math.sqrt(math.log(parent_visits + 1e-9) / node["n"])

def simulate_once(root, max_depth=D_DEFAULT):
    """
    Run one simulation from root using MCTS-style selection.
    """
    path = []
    node = root
    depth = 0

    while True:
        path.append(node)

        if node["type"] == "game":
            s = node["state"]
            yi = s["yi"]

            # Terminal or depth limit reached
            if not is_alive(s, yi):
                v = evaluate_state(s)
                break
            alive_idxs = [i for i, sn in enumerate(s["sn"]) if sn["alive"]]
            if len(alive_idxs) <= 1:
                v = evaluate_state(s)
                break
            if depth >= max_depth:
                v = evaluate_state(s)
                break

            # Expand GameNode if necessary
            my_idx = yi
            my_moves = safe_moves(s, my_idx)
            if not my_moves:
                v = evaluate_state(s)
                break

            if not node["children"]:
                for mv in my_moves:
                    node["children"][mv] = make_decision_node(mv, node)

            # Select a DecisionNode child via UCB
            parent_visits = node["n"] if node["n"] > 0 else 1
            children = list(node["children"].values())
            chosen = max(children, key=lambda cn: ucb_score(cn, parent_visits))
            node = chosen
            continue

        else:
            # DecisionNode: our move is fixed, enemies are random
            parent_game = node["parent"]
            base_state = parent_game["state"]
            my_idx = base_state["yi"]
            my_id = base_state["sn"][my_idx]["id"]

            if node["children"]:
                child = random.choice(node["children"])
                node = child
                depth += 1
                continue
            else:
                new_state = state_copy(base_state)
                mv = node["move"]
                moves = {my_id: mv}
                moves.update(random_enemy_moves(new_state, my_idx))
                step_state_inplace(new_state, moves)

                child = make_game_node(new_state)
                node["children"].append(child)
                node = child
                depth += 1
                continue

    # Backpropagate v along the path
    for n in path:
        n["n"] += 1
        n["sum_v"] += v
        if v < n["min_v"]:
            n["min_v"] = v


def danger_level(s):
    """Heuristic danger detection."""
    yi = s["yi"]
    snakes = s["sn"]
    me = snakes[yi]
    my_len = len(me["b"])
    my_hp = me["hp"]

    if my_hp <= 30:
        return "danger"

    for i, sn in enumerate(snakes):
        if i == yi or not sn["alive"]:
            continue
        if len(sn["b"]) >= my_len + 3:
            return "danger"

    return "safe"

def risk_averse_score(dnode, mode="safe"):
    """Score a DecisionNode based on mean and worst values."""
    if dnode["n"] == 0:
        return 0.0

    mean_v = dnode["sum_v"] / dnode["n"]
    worst_v = dnode["min_v"] if dnode["min_v"] != float("inf") else mean_v

    if mode == "danger":
        alpha = ALPHA_DANGER
    else:
        alpha = ALPHA_SAFE

    return alpha * mean_v + (1.0 - alpha) * worst_v

def fallback_value_for_unvisited(root_state, dnode, siblings, use_one_step=True):
    """Compute a value for an unvisited root DecisionNode."""
    mv = dnode["move"]

    if use_one_step:
        s1 = state_copy(root_state)
        my_idx = s1["yi"]
        my_id = s1["sn"][my_idx]["id"]

        moves = {my_id: mv}
        moves.update(random_enemy_moves(s1, my_idx))
        step_state_inplace(s1, moves)
        v = evaluate_state(s1)

        dnode["n"] = 1
        dnode["sum_v"] = v
        dnode["min_v"] = v
        return v
    else:
        visited_scores = []
        mode = danger_level(root_state)
        for sib in siblings:
            if sib["n"] > 0:
                visited_scores.append(risk_averse_score(sib, mode=mode))

        if visited_scores:
            prior_mean = sum(visited_scores) / len(visited_scores)
            prior_worst = min(visited_scores)
            v = 0.5 * prior_mean + 0.5 * prior_worst
        else:
            v = evaluate_state(root_state)

        dnode["n"] = 1
        dnode["sum_v"] = v
        dnode["min_v"] = v
        return v

def choose_move_minimax_tree_from_json(
    j,
    I=I_DEFAULT,
    max_depth=D_DEFAULT,
    use_one_step_fallback=USE_ONE_STEP_FALLBACK_FOR_UNVISITED,
    debug=False,
):
    """Main entry: given Battlesnake JSON 'j', choose our move."""
    s0 = s_from_json(j)
    root = make_game_node(state_copy(s0))

    # Run simulations
    for _ in range(I):
        simulate_once(root, max_depth=max_depth)

    # No children -> just pick any safe move or 'up'
    if not root["children"]:
        yi = s0["yi"]
        my_moves = safe_moves(s0, yi)
        if not my_moves:
            return "up"
        return random.choice(my_moves)

    mode = danger_level(s0)
    move_scores = {}
    dnodes = list(root["children"].values())

    for mv, dnode in root["children"].items():
        if dnode["n"] == 0:
            siblings = [sib for sib in dnodes if sib is not dnode]
            _ = fallback_value_for_unvisited(s0, dnode, siblings, use_one_step=use_one_step_fallback)
            score = risk_averse_score(dnode, mode=mode)
        else:
            score = risk_averse_score(dnode, mode=mode)
        move_scores[mv] = score

    best_move = max(move_scores.items(), key=lambda kv: kv[1])[0]

    if debug:
        print("Move scores:", move_scores)
        print("Chosen move:", best_move)
        print("Danger mode:", mode)

    return best_move


def handle_move_request(
    request_body,
    I=I_DEFAULT,
    max_depth=D_DEFAULT,
    use_one_step_fallback=USE_ONE_STEP_FALLBACK_FOR_UNVISITED,
    debug=False,
):
    """Adapter for Battlesnake /move endpoint."""
    mv = choose_move_minimax_tree_from_json(
        request_body,
        I=I,
        max_depth=max_depth,
        use_one_step_fallback=use_one_step_fallback,
        debug=debug,
    )
    return {
        "move": mv,
        "shout": f"tree: {mv}",
    }


# -----------------------------
# Flask server wiring
# -----------------------------

app = Flask(__name__)

@app.get("/")
def index():
    # Optional health/info endpoint
    return jsonify({"apiversion": "1", "author": "geeked", "color": "#00FFAA", "head": "default", "tail": "default"})


@app.post("/start")
def start():
    # You can log or adjust global parameters based on game setup here if you want
    data = request.get_json()
    # Example: print game id and board size
    # print(f"Game {data['game']['id']} started on {data['board']['width']}x{data['board']['height']}")
    return jsonify({"taunt": "glhf"})


@app.post("/move")
def move():
    data = request.get_json()
    move_resp = handle_move_request(data, I=I_DEFAULT, max_depth=D_DEFAULT,
                                    use_one_step_fallback=USE_ONE_STEP_FALLBACK_FOR_UNVISITED,
                                    debug=False)
    return jsonify(move_resp)


@app.post("/end")
def end():
    data = request.get_json()
    # Example: print result or cleanup; nothing required for logic
    # print(f"Game {data['game']['id']} ended.")
    return jsonify({"status": "ok"})

app.run(host="0.0.0.0", port=8080, debug=False)
