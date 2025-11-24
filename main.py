import json
import random
import copy

# =========================
# CONFIG
# =========================

I = 300   # simulations per root move
D = 8    # depth per simulation

dirs = ["up", "down", "left", "right"]
dirm = {
    "up":    (0, 1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": (1, 0),
}

# =========================
# BASIC GEOMETRY / HELPERS
# =========================

def p_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def in_bounds(s, p):
    w = s["w"]
    h = s["h"]
    x, y = p
    return 0 <= x < w and 0 <= y < h


# =========================
# STATE PARSING
# =========================

def s_from_json(j):
    g = j["board"]
    w = g["width"]
    h = g["height"]

    f = [(a["x"], a["y"]) for a in g.get("food", [])]
    hz = g.get("hazards", [])
    z = [(a["x"], a["y"]) for a in hz]

    sn = []
    yi = None

    for i, s0 in enumerate(g["snakes"]):
        b = [(a["x"], a["y"]) for a in s0["body"]]
        s1 = {
            "id": s0["id"],
            "hp": s0.get("health", 100),
            "b": b,
            "alive": True,
        }
        sn.append(s1)

    yid = j["you"]["id"]
    for i, s1 in enumerate(sn):
        if s1["id"] == yid:
            yi = i
            break

    s = {
        "w": w,
        "h": h,
        "f": f,
        "z": z,
        "sn": sn,
        "yi": yi,
    }
    return s


# =========================
# STEP GAME ONE TURN (IN PLACE)
# =========================

def step_state_inplace(s, m):
    sn = s["sn"]
    f = s["f"]

    # 0) health tick
    for a in sn:
        if not a["alive"]:
            continue
        a["hp"] -= 1

    # 1) new heads
    nh = {}
    for a in sn:
        if not a["alive"]:
            continue
        sid = a["id"]
        if a["hp"] <= 0:
            a["alive"] = False
            continue
        b = a["b"]
        h0 = b[0]
        mv = m.get(sid)
        if mv is None:
            h1 = h0
        else:
            h1 = p_add(h0, dirm[mv])
        nh[sid] = h1

    # 2) who eats?
    eat = {}
    for a in sn:
        if not a["alive"]:
            continue
        sid = a["id"]
        if sid not in nh:
            eat[sid] = False
            continue
        h1 = nh[sid]
        eat[sid] = h1 in f

    # 3) compute new bodies (temporary)
    body_cells = {}
    for a in sn:
        if not a["alive"]:
            continue
        sid = a["id"]
        if sid not in nh:
            continue
        b = a["b"]
        h1 = nh[sid]
        if eat[sid]:
            nb = [h1] + b
        else:
            if len(b) > 1:
                nb = [h1] + b[:-1]
            else:
                nb = [h1]
        a["_nb"] = nb

    # fill body cells (skip heads)
    for a in sn:
        if not a["alive"]:
            continue
        sid = a["id"]
        if "_nb" not in a:
            continue
        nb = a["_nb"]
        for p in nb[1:]:
            body_cells.setdefault(p, []).append(sid)

    # 4) out-of-bounds + body collisions
    dead_heads = set()
    for a in sn:
        if not a["alive"]:
            continue
        sid = a["id"]
        if sid not in nh:
            continue
        h1 = nh[sid]
        if not in_bounds(s, h1) or h1 in body_cells:
            dead_heads.add(sid)

    # 5) head-to-head collisions
    hh = {}
    for a in sn:
        if not a["alive"]:
            continue
        sid = a["id"]
        if sid in dead_heads or sid not in nh:
            continue
        h1 = nh[sid]
        hh.setdefault(h1, []).append(a)

    for p, lst in hh.items():
        if len(lst) <= 1:
            continue
        mx = max(len(a["b"]) for a in lst)
        winners = [a for a in lst if len(a["b"]) == mx]
        if len(winners) >= 2:
            for a in lst:
                dead_heads.add(a["id"])
        else:
            w0 = winners[0]["id"]
            for a in lst:
                if a["id"] != w0:
                    dead_heads.add(a["id"])

    # 6) remove eaten food
    eaten_points = set()
    for a in sn:
        sid = a["id"]
        if not a["alive"] or sid in dead_heads or sid not in nh:
            continue
        if nh[sid] in f:
            eaten_points.add(nh[sid])
    s["f"] = [p for p in f if p not in eaten_points]

    # finalize snakes
    for a in sn:
        sid = a["id"]
        if not a["alive"]:
            continue
        if sid in dead_heads:
            a["alive"] = False
            continue
        if "_nb" in a:
            a["b"] = a["_nb"]
            del a["_nb"]
        if eat.get(sid, False) and a["alive"]:
            a["hp"] = 100

    return s


# =========================
# SAFE MOVES (WITH HEAD THREATS)
# =========================

def safe_moves(s, idx):
    """
    Return all non-suicidal moves for snake index idx.
    Non-suicidal:
      - stays on board
      - does NOT hit any body segment
      - does NOT step into a tile that any equal/bigger head
        could move to next turn (head-to-head threat)
    If no such moves exist, we relax to threatened-but-not-immediate-suicide moves.
    If even those don't exist, we return any in-bounds moves.
    """
    sn = s["sn"]
    me = sn[idx]
    if not me["alive"]:
        return []

    my_len = len(me["b"])
    my_head = me["b"][0]

    # build body set
    body = set()
    for a in sn:
        if not a["alive"]:
            continue
        for p in a["b"]:
            body.add(p)

    # find threat squares from bigger/equal heads
    head_threat = set()
    for j, a in enumerate(sn):
        if j == idx or not a["alive"]:
            continue
        opp_len = len(a["b"])
        opp_head = a["b"][0]
        if opp_len >= my_len:
            for mv in dirs:
                p = p_add(opp_head, dirm[mv])
                if in_bounds(s, p):
                    head_threat.add(p)

    good = []
    fallback = []

    for mv in dirs:
        new = p_add(my_head, dirm[mv])

        if not in_bounds(s, new):
            continue
        if new in body:
            continue

        if new in head_threat:
            fallback.append(mv)
        else:
            good.append(mv)

    if good:
        return good
    if fallback:
        return fallback

    # truly trapped: any in-bounds move
    legal = []
    for mv in dirs:
        new = p_add(my_head, dirm[mv])
        if in_bounds(s, new):
            legal.append(mv)
    return legal


# =========================
# OPPONENT PATH SAMPLER (VECTOR SPACE ELEMENT)
# =========================

def sample_paths(s0, d):
    """
    Sample one vector-space element:
    For each opponent and each step 0..d-1, choose a random non-suicidal move if possible.
    """
    s1 = copy.deepcopy(s0)
    sn = s1["sn"]
    yi = s1["yi"]

    w0 = {}
    for idx, a in enumerate(sn):
        if not a["alive"]:
            continue
        w0[a["id"]] = []

    for t in range(d):
        m = {}
        sn = s1["sn"]
        for idx, a in enumerate(sn):
            if idx == yi:
                continue
            if not a["alive"]:
                sid = a["id"]
                w0[sid].append(None)
                continue
            sid = a["id"]
            ms = safe_moves(s1, idx)
            if not ms:
                mv = random.choice(dirs)
            else:
                mv = random.choice(ms)
            w0[sid].append(mv)
            m[sid] = mv

        if m:
            step_state_inplace(s1, m)

    return w0


# =========================
# ONE SCENARIO SIMULATION (STATE VERSION)
# =========================

def simulate_scenario_state(s0, a0, w0, d):
    """
    Run one scenario simulation starting from state s0.

    a0: root move ("up"/"down"/"left"/"right")
    w0: sampled opponent paths (dict snake_id -> list of moves)
    d: depth

    Returns:
      died: True if you died in this scenario
      food_eaten: how many segments you grew (earlier food weighted more)
      win: True if at least one opponent died before you
    """
    s = copy.deepcopy(s0)
    sn = s["sn"]
    yi = s["yi"]
    yid = sn[yi]["id"]

    fe = 0
    w1 = False

    # turn 0
    sn = s["sn"]
    y = sn[yi]
    prev_len = len(y["b"])

    m = {yid: a0}
    for a in sn:
        sid = a["id"]
        if sid != yid and sid in w0:
            m[sid] = w0[sid][0]

    step_state_inplace(s, m)

    sn = s["sn"]
    y = sn[yi]
    new_len = len(y["b"])
    if y["alive"]:
        dl = new_len - prev_len
        if dl > 0:
            # earlier turns get higher weight
            fe += dl * (d - 0)

    if not y["alive"]:
        return True, fe, False

    # opponent died?
    for i, a in enumerate(sn):
        if i != yi and not a["alive"]:
            return False, fe, True

    # future turns
    for t in range(1, d):
        sn = s["sn"]
        y = sn[yi]
        if not y["alive"]:
            return True, fe, w1

        ms = safe_moves(s, yi)
        if not ms:
            a1 = random.choice(dirs)
        else:
            a1 = random.choice(ms)

        m = {yid: a1}
        for a in sn:
            sid = a["id"]
            if sid != yid and sid in w0 and t < len(w0[sid]):
                m[sid] = w0[sid][t]

        prev_len = len(s["sn"][yi]["b"])
        step_state_inplace(s, m)
        new_len = len(s["sn"][yi]["b"])

        if y["alive"]:
            dl = new_len - prev_len
            if dl > 0:
                # earlier turns get higher weight
                fe += dl * (d - t)

        if not y["alive"]:
            return True, fe, w1

        for i, a in enumerate(sn):
            if i != yi and not a["alive"]:
                w1 = True
                return False, fe, w1

    return False, fe, w1


# =========================
# ROOT MOVE SELECTION
# =========================

def choose_move_from_json(j, i_cnt=None, d_depth=None, debug=False):
    if i_cnt is None:
        i_cnt = I
    if d_depth is None:
        d_depth = D

    s0 = s_from_json(j)
    sn0 = s0["sn"]
    yi = s0["yi"]
    y0 = sn0[yi]
    yid = y0["id"]

    root_moves = safe_moves(s0, yi)
    if not root_moves:
        root_moves = dirs[:]

    deaths = {}
    wins = {}
    food_sum = {}

    for a0 in root_moves:
        deaths[a0] = 0
        wins[a0] = 0
        food_sum[a0] = 0

        for _ in range(i_cnt):
            w0 = sample_paths(s0, d_depth)
            died, fe, w1 = simulate_scenario_state(s0, a0, w0, d_depth)
            if died:
                deaths[a0] += 1
            if w1:
                wins[a0] += 1
            food_sum[a0] += fe

    # bucketed death probability (don't-die objective)
    alpha = 1.0
    bucket_size = 0.05  # changed from 0.1

    danger_bucket = {}
    for a0 in root_moves:
        p = (deaths[a0] + alpha) / (i_cnt + 2 * alpha)
        b = int(p / bucket_size)
        danger_bucket[a0] = b

    # step 1: safest bucket only
    mb = min(danger_bucket[a0] for a0 in root_moves)
    safe_set = [a0 for a0 in root_moves if danger_bucket[a0] == mb]

    # ---- food bias inside the safest bucket ----
    # score = win_rate + food_bias * avg_food
    food_bias = 0.1  # per your request
    scores = {}

    for a0 in safe_set:
        surv = max(1, i_cnt - deaths[a0])
        win_rate = wins[a0] / surv
        # average food score over survivals (earlier food already weighted in fe)
        avg_food = food_sum[a0] / surv
        scores[a0] = win_rate + food_bias * avg_food

    # If any move is a "guaranteed win" (wins == survivals), restrict to those
    gw = []
    for a0 in safe_set:
        surv = i_cnt - deaths[a0]
        if surv > 0 and wins[a0] == surv:
            gw.append(a0)

    if gw:
        candidate_set = gw
    else:
        candidate_set = safe_set

    # among candidates, pick the move with best (win + food) score
    best_score = None
    best_moves = []
    for a0 in candidate_set:
        sc = scores[a0]
        if (best_score is None) or (sc > best_score + 1e-12):
            best_score = sc
            best_moves = [a0]
        elif abs(sc - best_score) <= 1e-12:
            best_moves.append(a0)

    mv = random.choice(best_moves)

    if debug:
        print("\n=== DEBUG SCORES ===")
        print(f"iterations: {i_cnt}, depth: {d_depth}, food_bias: {food_bias}")
        print(f"{'move':<8}{'deaths':<8}{'bucket':<8}{'wins':<8}"
              f"{'win_rate':<12}{'food_avg':<10}{'score':<10}")
        print("-" * 80)
        for a0 in root_moves:
            surv = max(1, i_cnt - deaths[a0])
            wr = wins[a0] / surv
            fa = food_sum[a0] / surv
            sc = scores.get(a0, float('-inf')) if a0 in safe_set else float('-inf')
            print(f"{a0:<8}{deaths[a0]:<8}{danger_bucket[a0]:<8}{wins[a0]:<8}"
                  f"{wr:<12.4f}{fa:<10.4f}{sc:<10.4f}")
        print("chosen move:", mv)
        print("====================\n")

    return mv


def battlesnake_move_response(j_str, i_cnt=None, d_depth=None):
    j = json.loads(j_str)
    mv = choose_move_from_json(j, i_cnt=i_cnt, d_depth=d_depth)
    return {
        "move": mv,
        "shout": f"vector space survival: {mv}",
    }


# =========================
# OPTIONAL: FLASK SERVER FOR BATTLESNAKE
# =========================
# You can comment this whole section out if you just want offline testing.

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
    mv = choose_move_from_json(j)
    return jsonify({"move": mv, "shout": ""})

@app.post("/end")
def end():
    return jsonify({})

app.run(host="0.0.0.0", port=8000, debug=False)
