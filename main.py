import json
import random
import copy
from collections import deque

# Global search parameters (tune for speed/strength)
I_DEFAULT = 700   # number of simulations per turn
D_DEFAULT = 8    # maximum depth of a single simulation

# Random seed (set to None for fully random behavior)
r0 = None

# Directions and vector deltas
dirs = ["up", "down", "left", "right"]
dirm = {
    "up":    (0, 1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": (1, 0),
}

def in_bounds(x, y, w, h):
    return 0 <= x < w and 0 <= y < h

def s_from_json(j):
    """Convert Battlesnake JSON game state into internal state dict.

State format:
  s = {
    'w': width,
    'h': height,
    'sn': [
        {
          'id': snake_id,
          'b': [(x_head, y_head), ..., (x_tail, y_tail)],
          'hp': health,
          'alive': True/False,
        }, ...
    ],
    'yi': index of our snake in sn,
    'f': [(fx, fy), ...]  # food positions
  }
    """
    board = j["board"]
    w, h = board["width"], board["height"]
    snakes_json = board["snakes"]
    food_json = board.get("food", [])
    you_id = j["you"]["id"]

    sn = []
    yi = 0
    for idx, sj in enumerate(snakes_json):
        sid = sj["id"]
        body_coords = [(seg["x"], seg["y"]) for seg in sj["body"]]
        hp = sj.get("health", 100)
        alive = sj.get("eliminated_cause") is None if "eliminated_cause" in sj else True
        sn.append({
            "id": sid,
            "b": body_coords,
            "hp": hp,
            "alive": alive,
        })
        if sid == you_id:
            yi = idx

    food = [(f["x"], f["y"]) for f in food_json]

    s = {
        "w": w,
        "h": h,
        "sn": sn,
        "yi": yi,
        "f": food,
    }
    s["init_len"] = len(s["sn"][s["yi"]]["b"])
    return s

def state_copy(s):
    """Deep copy of state dict."""
    return copy.deepcopy(s)


def compute_occupied_cells(s):
    """Return a set of (x, y) cells occupied by any alive snake body."""
    occ = set()
    for a in s["sn"]:
        if not a["alive"]:
            continue
        for (x, y) in a["b"]:
            occ.add((x, y))
    return occ

def safe_moves(s, idx):
    """Return a list of moves that do not immediately hit a wall or body.

This is conservative: it does not allow moving into any currently occupied
cell, even our own tail.
    """
    w, h = s["w"], s["h"]
    sn = s["sn"]
    me = sn[idx]
    head = me["b"][0]
    hx, hy = head

    occ = compute_occupied_cells(s)
    safe = []
    for mv in dirs:
        dx, dy = dirm[mv]
        nx, ny = hx + dx, hy + dy
        if not in_bounds(nx, ny, w, h):
            continue
        if (nx, ny) in occ:
            continue
        safe.append(mv)
    return safe


def step_state_inplace(s, moves):
    """Advance the game state by one turn using Battlesnake-like rules.
    - moves: dict snake_id -> move_str ('up', 'down', 'left', 'right')
    Modifies s in place.
    """
    w, h = s["w"], s["h"]
    sn = s["sn"]
    food = s["f"]
    food_set = set(food)

    # Pre-move snapshot
    old_bodies = [list(a["b"]) for a in sn]
    old_hps = [a["hp"] for a in sn]
    lengths = [len(a["b"]) for a in sn]
    alive = [a["alive"] for a in sn]

    # Compute intended new heads
    new_heads = [None] * len(sn)
    will_eat = [False] * len(sn)

    for i, a in enumerate(sn):
        if not alive[i]:
            continue
        mv = moves.get(a["id"], None)
        if mv is None:
            # If no move given, snake dies
            alive[i] = False
            continue
        hx, hy = old_bodies[i][0]
        dx, dy = dirm[mv]
        nx, ny = hx + dx, hy + dy
        new_heads[i] = (nx, ny)
        if (nx, ny) in food_set:
            will_eat[i] = True

    # Wall collisions
    for i, pos in enumerate(new_heads):
        if not alive[i]:
            continue
        if pos is None:
            alive[i] = False
            continue
        x, y = pos
        if not in_bounds(x, y, w, h):
            alive[i] = False

    # Body collisions (bodies that remain after tails move if not eating)
    occupied_next = set()
    for i, body in enumerate(old_bodies):
        if not alive[i]:
            continue
        L = len(body)
        for j, (x, y) in enumerate(body):
            # If tail and not eating, this cell will be vacated
            if j == L - 1 and not will_eat[i]:
                continue
            occupied_next.add((x, y))

    for i, pos in enumerate(new_heads):
        if not alive[i] or pos is None:
            continue
        if pos in occupied_next:
            alive[i] = False

    # Head-to-head collisions
    head_pos_groups = {}
    for i, pos in enumerate(new_heads):
        if not alive[i] or pos is None:
            continue
        head_pos_groups.setdefault(pos, []).append(i)

    for pos, idxs in head_pos_groups.items():
        if len(idxs) > 1:
            max_len = max(lengths[i] for i in idxs)
            # shorter snakes die
            for i in idxs:
                if lengths[i] < max_len:
                    alive[i] = False
            # if all same length, all die
            if all(lengths[i] == max_len for i in idxs):
                for i in idxs:
                    alive[i] = False

    # Update snakes: bodies, health, deaths from starvation
    new_food_set = set(food_set)
    for i, a in enumerate(sn):
        if not alive[i]:
            a["alive"] = False
            continue

        old_body = old_bodies[i]
        head = new_heads[i]
        if head is None:
            a["alive"] = False
            alive[i] = False
            continue

        if will_eat[i]:
            new_body = [head] + old_body
            new_hp = 100
            if head in new_food_set:
                new_food_set.remove(head)
        else:
            new_body = [head] + old_body[:-1]
            new_hp = old_hps[i] - 1

        if new_hp <= 0:
            a["alive"] = False
            alive[i] = False
            continue

        a["b"] = new_body
        a["hp"] = new_hp
        a["alive"] = True

    s["f"] = list(new_food_set)


from collections import defaultdict
import math

def compute_areas(s):
    """
    Dynamic territory floodfill with progressive tail-decay.
    Does NOT mutate bodies. Instead precomputes when each body tile becomes walkable.
    This is the most efficient and correct approach.
    """

    sn = s["sn"]
    yi = s["yi"]
    w, h = s["w"], s["h"]
    dirs = list(dirm.values())

    # ---------------------------------------------------
    # 1. Precompute walkable_at time for every tile
    # ---------------------------------------------------
    walkable_at = {}

    # all empty tiles default to walkable at time 0
    for x in range(w):
        for y in range(h):
            walkable_at[(x, y)] = 0

    # overwrite body tiles with actual times
    for snake in sn:
        if not snake["alive"]:
            continue

        body = snake["b"]
        L = len(body)

        # head is blocked forever (other snakes can't walk on it)
        hx, hy = body[0]
        walkable_at[(hx, hy)] = math.inf

        # inner segments open when tail eventually reaches them
        # e.g., k steps from head: opens at (L-1 - k)
        for k in range(1, L - 1):
            x, y = body[k]
            walkable_at[(x, y)] = (L - 1) - k

        # tail becomes free next turn
        if L > 1:
            tx, ty = body[-1]
            walkable_at[(tx, ty)] = 1

    # ---------------------------------------------------
    # 2. Multi-source BFS initialization
    # ---------------------------------------------------
    dist = {}                # (x,y) -> earliest distance
    claimants = defaultdict(set)
    frontier = []

    for i, snake in enumerate(sn):
        if not snake["alive"]:
            continue
        hx, hy = snake["b"][0]
        dist[(hx, hy)] = 0
        claimants[(hx, hy)].add(i)
        frontier.append((hx, hy, i))

    d = 0  # BFS layer

    # ---------------------------------------------------
    # 3. BFS with dynamic walkability
    # ---------------------------------------------------
    while frontier:
        next_frontier = []

        for (x, y, idx) in frontier:
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy

                # in bounds
                if not in_bounds(nx, ny, w, h):
                    continue

                # can only enter when tile becomes walkable
                if d + 1 < walkable_at[(nx, ny)]:
                    continue

                # first time reached
                if (nx, ny) not in dist:
                    dist[(nx, ny)] = d + 1
                    claimants[(nx, ny)].add(idx)
                    next_frontier.append((nx, ny, idx))

                # already reached at same min distance → tie
                elif dist[(nx, ny)] == d + 1:
                    claimants[(nx, ny)].add(idx)

        # STOP when BFS adds no new tiles
        if not next_frontier:
            break

        frontier = next_frontier
        d += 1

    # ---------------------------------------------------
    # 4. Resolve tile ownership
    # ---------------------------------------------------
    lengths = {i: len(a["b"]) for i, a in enumerate(sn)}
    area_count = defaultdict(int)

    for cell, idxs in claimants.items():

        # Only one snake reached → that snake owns
        if len(idxs) == 1:
            area_count[next(iter(idxs))] += 1
            continue

        # Multiple → bigger snake wins
        mx = max(lengths[i] for i in idxs)
        best = [i for i in idxs if lengths[i] == mx]

        if len(best) == 1:
            area_count[best[0]] += 1
        # if multiple same length: ignore tile (contested)

    my_area = area_count.get(yi, 0)

    max_enemy_area = 0
    for i, snake in enumerate(sn):
        if i != yi and snake["alive"]:
            max_enemy_area = max(max_enemy_area, area_count.get(i, 0))

    return my_area, max_enemy_area

def evaluate_state(s):
    """
    Heuristic score for a state.
    - huge negative if we are dead
    - otherwise: combine:
      * our reachable area (floodfill)
      * enemy area
      * our health
      * nearby food
      * length gained (current_len - start_len), weighted slightly more if HP low
    """
    sn = s["sn"]
    yi = s["yi"]
    me = sn[yi]

    # Hard death penalty
    if not me["alive"]:
        return -1e9

    # ---------- 1) Space control ----------
    my_area, max_enemy_area = compute_areas(s)
    for i, a in enumerate(sn):
        if i == yi or not a["alive"]:
            continue

    # ---------- 2) Food proximity ----------
    head = me["b"][0]
    hx, hy = head
    food_bonus = 0
    for (fx, fy) in s["f"]:
        dist = abs(fx - hx) + abs(fy - hy)
        if dist <= 6:
            if me["hp"] < 30:
                food_bonus += max(0, 6 - dist)
            else:
                food_bonus += max(0, 3 - dist)

    # ---------- 3) Length gained ----------
    current_len = len(me["b"])
    start_len   = s["init_len"]      # MUST be stored at root
    hp          = me["hp"]

    base_len_weight = 0.2

    if hp < 20:
        low_hp_factor = 1.0 + 0.3 * (30 - hp) / 30.0
    else:
        low_hp_factor = 1.0

    effective_len_weight = base_len_weight * low_hp_factor
    length_gain = current_len - start_len
    # ---------- 4) Combine terms ----------
    space_weight   = 1.0
    control_weight = 0.5
    health_weight  = 0.05
    food_weight    = 0.2

    control_term = my_area - max_enemy_area
    score  = 0.0
    score += space_weight   * my_area
    score += control_weight * control_term
    score += health_weight  * hp
    score += food_weight    * food_bonus
    score += effective_len_weight * length_gain  # NEW LENGTH LOGIC

    return score


def random_enemy_moves(s, my_idx):
    """For each enemy snake, pick a random safe move (or random legal move)."""
    sn = s["sn"]
    moves = {}
    for i, a in enumerate(sn):
        if not a["alive"]:
            continue
        if i == my_idx:
            continue
        safe = safe_moves(s, i)
        if safe:
            mv = random.choice(safe)
        else:
            mv = random.choice(dirs)
        moves[a["id"]] = mv
    return moves


def make_game_node(state):
    return {
        "type": "game",
        "state": state,
        "children": {},   # move_str -> decision_node
        "value": None,
    }


def make_decision_node(move, parent_state):
    return {
        "type": "decision",
        "move": move,
        "parent_state": parent_state,
        "children": [],   # list of game_nodes
        "value": None,
    }


def expand_game_node(node):
    """Expand a game node by creating one Decision node per safe move."""
    assert node["type"] == "game"
    s = node["state"]
    yi = s["yi"]

    my_moves = safe_moves(s, yi)
    if not my_moves:
        my_moves = dirs[:]  # trapped fallback

    for mv in my_moves:
        if mv not in node["children"]:
            node["children"][mv] = make_decision_node(mv, s)


def expand_decision_node(node):
    """Expand a decision node by sampling ONE random enemy response
    and creating ONE new GameNode child.
    """
    assert node["type"] == "decision"
    parent_state = node["parent_state"]
    s1 = state_copy(parent_state)
    sn1 = s1["sn"]
    yi = s1["yi"]
    me1 = sn1[yi]

    moves = {me1["id"]: node["move"]}
    enemy_moves = random_enemy_moves(s1, yi)
    moves.update(enemy_moves)

    step_state_inplace(s1, moves)

    child_game = make_game_node(s1)
    node["children"].append(child_game)
    return child_game


def simulate_once(root, max_depth=D_DEFAULT):
    """Run one tree simulation starting from root GameNode.
    Uses random selection among children, expands nodes when needed,
    and backs up values using minimax semantics.
    """
    node = root
    path = []
    depth = 0

    while True:
        path.append(node)

        if depth >= max_depth:
            break

        if node["type"] == "game":
            s = node["state"]
            sn = s["sn"]
            yi = s["yi"]
            me = sn[yi]
            if not me["alive"]:
                break

            if not node["children"]:
                expand_game_node(node)

            node = random.choice(list(node["children"].values()))

        else:  # decision node
            if not node["children"]:
                node = expand_decision_node(node)
            else:
                node = random.choice(node["children"])

        depth += 1

        # Ensure we end on a GameNode for evaluation
        if depth >= max_depth and node["type"] == "decision":
            node = expand_decision_node(node)
            path.append(node)
            break

    if node["type"] != "game":
        return

    leaf_value = evaluate_state(node["state"])
    node["value"] = leaf_value

    # Backup along the path
    for n in reversed(path):
        if n["type"] == "game":
            if not n["children"]:
                if n["value"] is None:
                    n["value"] = evaluate_state(n["state"])
            else:
                vals = [child["value"] for child in n["children"].values() if child["value"] is not None]
                if vals:
                    n["value"] = max(vals)
        else:  # decision node
            if n["children"]:
                vals = [child["value"] for child in n["children"] if child["value"] is not None]
                if vals:
                    n["value"] = min(vals)


def choose_move_minimax_tree_from_json(j, I=I_DEFAULT, max_depth=D_DEFAULT, debug=False):
    """Full minimax-style tree search with random rollouts for enemy moves."""
    s0 = s_from_json(j)
    root = make_game_node(s0)

    for _ in range(I):
        simulate_once(root, max_depth=max_depth)

    if not root["children"]:
        yi = s0["yi"]
        my_moves = safe_moves(s0, yi)
        if not my_moves:
            my_moves = dirs[:]
        return random.choice(my_moves)

    move_values = {}
    for mv, dnode in root["children"].items():
        move_values[mv] = dnode["value"] if dnode["value"] is not None else -1e9

    best_move = max(move_values.keys(), key=lambda mv: move_values[mv])

    if debug:
        print("Root move values (minimax tree):")
        for mv in sorted(move_values.keys()):
            print(f"{mv:>5}: {move_values[mv]:8.2f}")
        print("Chosen move:", best_move)

    return best_move


def battlesnake_move_response_minimax_tree(j_str, I=I_DEFAULT, max_depth=D_DEFAULT):
    mv = choose_move_minimax_tree_from_json(j_str, I=I, max_depth=max_depth)
    return {
        "move": mv,
        "shout": "",
    }


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.get("/")
def index():
    return jsonify({
        "apiversion": "1",
        "author": "me",
        "color": "#00FF00",
        "head": "safe",
        "tail": "sharp",
        "version": "0.1.0"
    })

@app.post("/start")
def start():
    return jsonify({"color": "#00FF00", "head": "safe", "tail": "sharp"})

@app.post("/move")
def move():
    j = request.get_json()
    return battlesnake_move_response_minimax_tree(j, I=I_DEFAULT, max_depth=D_DEFAULT)

@app.post("/end")
def end():
    return jsonify({})

app.run(host="0.0.0.0", port=8000, debug=False)
