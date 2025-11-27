# import json
# import math
# import random
# import copy
# from collections import deque, defaultdict

# from flask import Flask, request, jsonify

# # =========================
# # Global config
# # =========================

# I_DEFAULT = 500    # number of simulations per turn
# D_DEFAULT = 10     # max depth per simulation

# # Risk-averse root scoring blend (mean vs worst)
# ALPHA_SAFE = 0.8      # when we're comfortable, lean on mean more
# ALPHA_DANGER = 0.45   # when in danger, still pessimistic but not paralyzed

# # Unvisited DecisionNodes handling at root
# USE_ONE_STEP_FALLBACK_FOR_UNVISITED = True

# # Head-to-head kill penalty baseline
# BASE_HEAD_KILL_PENALTY = 3e7

# # Random seed (None = fully random)
# r0 = None
# if r0 is not None:
#     random.seed(r0)

# DIRS = ["up", "down", "left", "right"]
# DELTA = {
#     "up":    (0, 1),
#     "down":  (0, -1),
#     "left":  (-1, 0),
#     "right": (1, 0),
# }

# # =========================
# # Basic helpers
# # =========================

# def in_bounds(x, y, w, h):
#     return 0 <= x < w and 0 <= y < h

# def manhattan(a, b):
#     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# def is_stacked_body(body):
#     """True if body has duplicate coords (stacked, tail won't move)."""
#     seen = set()
#     for x, y in body:
#         if (x, y) in seen:
#             return True
#         seen.add((x, y))
#     return False

# # =========================
# # State conversion
# # =========================

# def s_from_json(j):
#     board = j["board"]
#     you_id = j["you"]["id"]

#     w = board["width"]
#     h = board["height"]

#     snakes = []
#     yi = None

#     for idx, s_json in enumerate(board["snakes"]):
#         snake_id = s_json["id"]
#         body = [(seg["x"], seg["y"]) for seg in s_json["body"]]
#         hp = s_json["health"]
#         alive = True

#         snake = {
#             "id": snake_id,
#             "b": body,
#             "hp": hp,
#             "alive": alive,
#         }
#         snakes.append(snake)
#         if snake_id == you_id:
#             yi = idx

#     food = [(f["x"], f["y"]) for f in board["food"]]

#     s = {
#         "w": w,
#         "h": h,
#         "sn": snakes,
#         "yi": yi,
#         "f": food,
#     }
#     s["init_len"] = len(s["sn"][s["yi"]]["b"])
#     return s

# def state_copy(s):
#     return copy.deepcopy(s)

# def is_alive(s, idx):
#     return 0 <= idx < len(s["sn"]) and s["sn"][idx]["alive"]

# # =========================
# # Tail-aware occupancy & safe moves
# # =========================

# def compute_occupied_for_collision(s):
#     """
#     Occupied cells for body collisions:
#       - All body segments except tails are blocked.
#       - Tail of NON-stacked snake is vacating -> free.
#       - Tail of stacked snake stays -> blocked.
#     """
#     occ = set()
#     for snake in s["sn"]:
#         if not snake["alive"]:
#             continue
#         body = snake["b"]
#         if not body:
#             continue
#         tail = body[-1]
#         stacked = is_stacked_body(body)
#         for seg in body[:-1]:
#             occ.add(seg)
#         if stacked:
#             occ.add(tail)
#     return occ

# def safe_moves(s, idx):
#     """
#     Moves that don't immediately die by body/wall collision.
#     Head-to-head risk is evaluated separately.
#     """
#     snake = s["sn"][idx]
#     if not snake["alive"]:
#         return []

#     w, h = s["w"], s["h"]
#     head_x, head_y = snake["b"][0]
#     occ = compute_occupied_for_collision(s)

#     out = []
#     for mv in DIRS:
#         dx, dy = DELTA[mv]
#         nx, ny = head_x + dx, head_y + dy
#         if not in_bounds(nx, ny, w, h):
#             continue

#         if (nx, ny) in occ:
#             body = snake["b"]
#             if len(body) > 0 and (nx, ny) == body[-1] and not is_stacked_body(body):
#                 out.append(mv)  # own tail vacates
#             else:
#                 continue
#         else:
#             out.append(mv)
#     return out

# # =========================
# # Step function
# # =========================

# def step_state_inplace(s, moves):
#     w, h = s["w"], s["h"]
#     snakes = s["sn"]
#     food = s["f"]
#     food_set = set(food)

#     new_heads = {}
#     will_eat = {}
#     for snake in snakes:
#         if not snake["alive"]:
#             continue
#         sid = snake["id"]
#         body = snake["b"]
#         if not body:
#             snake["alive"] = False
#             continue

#         head_x, head_y = body[0]
#         mv = moves.get(sid, "up")
#         dx, dy = DELTA[mv]
#         nx, ny = head_x + dx, head_y + dy

#         new_heads[sid] = (nx, ny)
#         will_eat[sid] = (nx, ny) in food_set

#     new_bodies = {}
#     for snake in snakes:
#         if not snake["alive"]:
#             continue
#         sid = snake["id"]
#         body = snake["b"]
#         if sid not in new_heads:
#             snake["alive"] = False
#             continue
#         nx, ny = new_heads[sid]
#         eat = will_eat[sid]

#         if eat:
#             new_body = [(nx, ny)] + body
#         else:
#             new_body = [(nx, ny)] + body[:-1]
#         new_bodies[sid] = new_body

#     body_cells = set()
#     for snake in snakes:
#         if not snake["alive"]:
#             continue
#         nb = new_bodies.get(snake["id"])
#         if not nb:
#             continue
#         for (x, y) in nb[1:]:
#             body_cells.add((x, y))

#     for snake in snakes:
#         if not snake["alive"]:
#             continue
#         nb = new_bodies.get(snake["id"])
#         if not nb:
#             snake["alive"] = False
#             continue
#         hx, hy = nb[0]

#         if not in_bounds(hx, hy, w, h):
#             snake["alive"] = False
#             continue

#         if (hx, hy) in body_cells:
#             snake["alive"] = False
#             continue

#     heads_map = {}
#     for snake in snakes:
#         if not snake["alive"]:
#             continue
#         nb = new_bodies.get(snake["id"])
#         if not nb:
#             snake["alive"] = False
#             continue
#         hx, hy = nb[0]
#         heads_map.setdefault((hx, hy), []).append(snake)

#     for pos, snakes_here in heads_map.items():
#         if len(snakes_here) <= 1:
#             continue
#         max_len = max(len(s["b"]) for s in snakes_here)
#         longest = [s for s in snakes_here if len(s["b"]) == max_len]
#         if len(longest) == 1:
#             for s2 in snakes_here:
#                 if s2 is not longest[0]:
#                     s2["alive"] = False
#         else:
#             for s2 in snakes_here:
#                 s2["alive"] = False

#     eaten_food = set()
#     for snake in snakes:
#         if not snake["alive"]:
#             continue
#         sid = snake["id"]
#         nb = new_bodies.get(sid)
#         if not nb:
#             snake["alive"] = False
#             continue

#         snake["b"] = nb
#         snake["hp"] -= 1
#         if will_eat.get(sid, False):
#             snake["hp"] = 100
#             eaten_food.add(nb[0])

#         if snake["hp"] <= 0:
#             snake["alive"] = False

#     if eaten_food:
#         s["f"] = [f for f in food if f not in eaten_food]

# # =========================
# # Decaying-body flood fill
# # =========================

# def compute_walkable_at(s):
#     """
#     Return map (x,y) -> earliest time step when tile becomes walkable,
#     using dynamic tail decay (your suggested logic).
#     """
#     sn = s["sn"]
#     w, h = s["w"], s["h"]

#     walkable_at = {}

#     # default: empty tiles walkable at time 0
#     for x in range(w):
#         for y in range(h):
#             walkable_at[(x, y)] = 0

#     # overwrite body tiles with actual times
#     for snake in sn:
#         if not snake["alive"]:
#             continue

#         body = snake["b"]
#         L = len(body)
#         if L == 0:
#             continue

#         # head: blocked forever
#         hx, hy = body[0]
#         walkable_at[(hx, hy)] = math.inf

#         # inner segments open when tail eventually reaches them
#         # index k from head: opens at (L - 1 - k)
#         for k in range(1, L - 1):
#             x, y = body[k]
#             walkable_at[(x, y)] = (L - 1) - k

#         # tail becomes free next turn
#         if L > 1:
#             tx, ty = body[-1]
#             walkable_at[(tx, ty)] = 1

#     return walkable_at

# def compute_areas(s):
#     """
#     Dynamic territory floodfill with progressive tail-decay.
#     Uses your decaying-body BFS to get (my_area, max_enemy_area).
#     """
#     sn = s["sn"]
#     yi = s["yi"]
#     w, h = s["w"], s["h"]
#     dirs = list(DELTA.values())

#     walkable_at = compute_walkable_at(s)

#     dist = {}                # (x,y) -> earliest distance
#     claimants = defaultdict(set)
#     frontier = []

#     # Multi-source init from all heads
#     for i, snake in enumerate(sn):
#         if not snake["alive"]:
#             continue
#         body = snake["b"]
#         if not body:
#             continue
#         hx, hy = body[0]
#         dist[(hx, hy)] = 0
#         claimants[(hx, hy)].add(i)
#         frontier.append((hx, hy, i))

#     d = 0

#     # BFS with dynamic walkability
#     while frontier:
#         next_frontier = []

#         for (x, y, idx) in frontier:
#             for dx, dy in dirs:
#                 nx, ny = x + dx, y + dy

#                 if not in_bounds(nx, ny, w, h):
#                     continue

#                 if d + 1 < walkable_at[(nx, ny)]:
#                     continue

#                 if (nx, ny) not in dist:
#                     dist[(nx, ny)] = d + 1
#                     claimants[(nx, ny)].add(idx)
#                     next_frontier.append((nx, ny, idx))
#                 elif dist[(nx, ny)] == d + 1:
#                     claimants[(nx, ny)].add(idx)

#         if not next_frontier:
#             break

#         frontier = next_frontier
#         d += 1

#     lengths = {i: len(a["b"]) for i, a in enumerate(sn)}
#     area_count = defaultdict(int)

#     for cell, idxs in claimants.items():
#         if len(idxs) == 1:
#             area_count[next(iter(idxs))] += 1
#             continue

#         mx = max(lengths[i] for i in idxs)
#         best = [i for i in idxs if lengths[i] == mx]

#         if len(best) == 1:
#             area_count[best[0]] += 1
#         # ties of equal length: contested, ignored

#     my_area = area_count.get(yi, 0)

#     max_enemy_area = 0
#     for i, snake in enumerate(sn):
#         if i != yi and snake["alive"]:
#             max_enemy_area = max(max_enemy_area, area_count.get(i, 0))

#     return my_area, max_enemy_area

# def compute_local_space(s, idx, max_steps=30):
#     """
#     Local decaying-body floodfill from our head:
#     how many tiles we can reach before time max_steps.
#     This is what keeps us out of sticky self-trap situations.
#     """
#     w, h = s["w"], s["h"]
#     sn = s["sn"]
#     me = sn[idx]
#     head = me["b"][0]

#     walkable_at = compute_walkable_at(s)

#     visited = set([head])
#     q = deque([(head[0], head[1], 0)])
#     reachable = 0

#     while q:
#         x, y, d = q.popleft()
#         reachable += 1
#         if d >= max_steps:
#             continue
#         for dx, dy in DELTA.values():
#             nx, ny = x + dx, y + dy
#             if not in_bounds(nx, ny, w, h):
#                 continue
#             if (nx, ny) in visited:
#                 continue
#             if d + 1 < walkable_at[(nx, ny)]:
#                 continue
#             visited.add((nx, ny))
#             q.append((nx, ny, d + 1))
#     return reachable

# # =========================
# # Evaluation
# # =========================

# def evaluate_state(s):
#     """
#     Scalar eval, from our point of view.

#     Principles:
#       - Never die.
#       - Always value food (growth = threat).
#       - Shorter -> more food/health driven, shy about head-to-head.
#       - Longer -> still values food but leans more on space/control & kill pressure.
#       - Flood fill:
#           * global territory via decaying-body BFS
#           * local reachable area via decaying-body BFS
#     """
#     yi = s["yi"]
#     snakes = s["sn"]
#     me = snakes[yi]

#     if not me["alive"]:
#         return -1e9

#     alive_indices = [i for i, sn in enumerate(snakes) if sn["alive"]]
#     if len(alive_indices) == 1 and alive_indices[0] == yi:
#         return 1e8

#     my_body = me["b"]
#     my_len = len(my_body)
#     my_hp = me["hp"]
#     my_head = my_body[0]

#     # Length relations
#     enemy_lengths = [len(sn["b"]) for i, sn in enumerate(snakes) if sn["alive"] and i != yi]
#     max_enemy_len = max(enemy_lengths) if enemy_lengths else 0
#     length_diff = my_len - max_enemy_len  # >0: we're longer

#     # Territory & local space (both decaying-body based)
#     my_area, max_enemy_area = compute_areas(s)
#     local_space = compute_local_space(s, yi)

#     # Food: always valuable
#     food_bonus = 0.0
#     nearest_food_dist = None
#     for fx, fy in s["f"]:
#         d = manhattan(my_head, (fx, fy))
#         if nearest_food_dist is None or d < nearest_food_dist:
#             nearest_food_dist = d
#         if d == 0:
#             food_bonus += 12.0
#         elif d <= 8:
#             food_bonus += (10 - d)

#     # Base weights before personality adjustments
#     space_weight = 0.18
#     control_weight = 0.1
#     local_space_weight = 0.7  # still important, but not dominating

#     if my_hp <= 35:
#         base_health_w = 0.09
#         base_food_w = 0.8
#     else:
#         base_health_w = 0.05
#         base_food_w = 0.5

#     # Personality by relative length
#     if length_diff <= -3:
#         # Significantly shorter → eat & stay open
#         health_weight = base_health_w * 1.3
#         food_weight = base_food_w * 1.4
#         local_space_weight *= 1.1
#         space_weight *= 0.7
#         control_weight *= 0.7
#     elif length_diff >= 3:
#         # Significantly longer → still value food, but more on space/control
#         health_weight = base_health_w * 0.9
#         food_weight = base_food_w * 0.85
#         local_space_weight *= 1.0
#         space_weight *= 1.3
#         control_weight *= 1.3
#     else:
#         health_weight = base_health_w
#         food_weight = base_food_w

#     # Length gain since root (growth = threat)
#     init_len = s.get("init_len", my_len)
#     length_gain = my_len - init_len
#     if length_diff <= -2:
#         length_gain_weight = 0.35
#     elif length_diff >= 2:
#         length_gain_weight = 0.22
#     else:
#         length_gain_weight = 0.28

#     # Starvation / unreachable food
#     starvation_penalty = 0.0
#     if my_hp <= 25:
#         if not s["f"]:
#             starvation_penalty -= 80.0
#         elif nearest_food_dist is not None and nearest_food_dist > 8:
#             starvation_penalty -= 50.0

#     # Current adjacency head danger/opportunity
#     head_danger_term = 0.0
#     for i, sn in enumerate(snakes):
#         if i == yi or not sn["alive"]:
#             continue
#         other_head = sn["b"][0]
#         other_len = len(sn["b"])
#         d = manhattan(my_head, other_head)
#         if d == 1:
#             if other_len >= my_len + 1:
#                 head_danger_term -= 45.0
#             elif my_len >= other_len + 2:
#                 # bigger → mild reward for pressure
#                 head_danger_term += 10.0

#     space_term = space_weight * my_area
#     control_term = control_weight * (my_area - max_enemy_area)
#     health_term = health_weight * my_hp
#     food_term = food_weight * food_bonus
#     length_term = length_gain_weight * length_gain
#     local_space_term = local_space_weight * local_space

#     score = (
#         space_term +
#         control_term +
#         health_term +
#         food_term +
#         length_term +
#         starvation_penalty +
#         head_danger_term +
#         local_space_term
#     )
#     return score

# # =========================
# # Enemy model
# # =========================

# def random_enemy_moves(s, my_idx):
#     snakes = s["sn"]
#     moves = {}
#     for i, sn in enumerate(snakes):
#         if not sn["alive"]:
#             continue
#         if i == my_idx:
#             continue
#         sid = sn["id"]
#         sm = safe_moves(s, i)
#         if sm:
#             mv = random.choice(sm)
#         else:
#             mv = random.choice(DIRS)
#         moves[sid] = mv
#     return moves

# # =========================
# # Tree structures & MCTS
# # =========================

# def make_game_node(state):
#     return {
#         "type": "game",
#         "state": state,
#         "children": {},
#         "n": 0,
#         "sum_v": 0.0,
#         "min_v": float("inf")
#     }

# def make_decision_node(move, parent_game_node):
#     return {
#         "type": "decision",
#         "move": move,
#         "parent": parent_game_node,
#         "children": [],
#         "n": 0,
#         "sum_v": 0.0,
#         "min_v": float("inf"),
#     }

# def ucb_score(node, parent_visits, c=1.4):
#     if node["n"] == 0:
#         return float("inf")
#     mean = node["sum_v"] / node["n"]
#     return mean + c * math.sqrt(math.log(parent_visits + 1e-9) / node["n"])

# def simulate_once(root, max_depth=D_DEFAULT):
#     """
#     One MCTS rollout:
#       - selection
#       - expansion
#       - simulation
#       - backprop using evaluate_state
#     """
#     path = []
#     node = root
#     depth = 0
#     v = None

#     while True:
#         path.append(node)

#         if node["type"] == "game":
#             s = node["state"]
#             yi = s["yi"]

#             if not is_alive(s, yi):
#                 v = evaluate_state(s)
#                 break
#             alive_idxs = [i for i, sn in enumerate(s["sn"]) if sn["alive"]]
#             if len(alive_idxs) <= 1:
#                 v = evaluate_state(s)
#                 break
#             if depth >= max_depth:
#                 v = evaluate_state(s)
#                 break

#             my_idx = yi
#             my_moves = safe_moves(s, my_idx)
#             if not my_moves:
#                 v = evaluate_state(s)
#                 break

#             if not node["children"]:
#                 for mv in my_moves:
#                     node["children"][mv] = make_decision_node(mv, node)

#             parent_visits = node["n"] if node["n"] > 0 else 1
#             children = list(node["children"].values())
#             chosen = max(children, key=lambda cn: ucb_score(cn, parent_visits))
#             node = chosen
#             continue

#         else:
#             parent_game = node["parent"]
#             base_state = parent_game["state"]
#             my_idx = base_state["yi"]
#             my_id = base_state["sn"][my_idx]["id"]

#             if node["children"]:
#                 child = random.choice(node["children"])
#                 node = child
#                 depth += 1
#                 continue
#             else:
#                 new_state = state_copy(base_state)
#                 mv = node["move"]
#                 moves = {my_id: mv}
#                 moves.update(random_enemy_moves(new_state, my_idx))
#                 step_state_inplace(new_state, moves)

#                 child = make_game_node(new_state)
#                 node["children"].append(child)
#                 node = child
#                 depth += 1
#                 continue

#     # v computed at leaf / terminal
#     for n in path:
#         n["n"] += 1
#         n["sum_v"] += v
#         if v < n["min_v"]:
#             n["min_v"] = v

# # =========================
# # Risk profile + root fallback
# # =========================

# def danger_level(s):
#     """
#     Decide risk profile based on HP and relative size.
#     """
#     yi = s["yi"]
#     snakes = s["sn"]
#     me = snakes[yi]
#     my_len = len(me["b"])
#     my_hp = me["hp"]

#     enemy_lengths = [len(sn["b"]) for i, sn in enumerate(snakes) if sn["alive"] and i != yi]
#     max_enemy_len = max(enemy_lengths) if enemy_lengths else 0
#     length_diff = my_len - max_enemy_len

#     if my_hp <= 30:
#         return "danger"
#     if length_diff <= -3:
#         return "danger"
#     return "safe"

# def risk_averse_score(dnode, mode="safe"):
#     if dnode["n"] == 0:
#         return 0.0
#     mean_v = dnode["sum_v"] / dnode["n"]
#     worst_v = dnode["min_v"] if dnode["min_v"] != float("inf") else mean_v
#     alpha = ALPHA_DANGER if mode == "danger" else ALPHA_SAFE
#     return alpha * mean_v + (1.0 - alpha) * worst_v

# def fallback_value_for_unvisited(root_state, dnode, siblings, use_one_step=True):
#     mv = dnode["move"]

#     if use_one_step:
#         s1 = state_copy(root_state)
#         my_idx = s1["yi"]
#         my_id = s1["sn"][my_idx]["id"]

#         moves = {my_id: mv}
#         moves.update(random_enemy_moves(s1, my_idx))
#         step_state_inplace(s1, moves)
#         v = evaluate_state(s1)

#         dnode["n"] = 1
#         dnode["sum_v"] = v
#         dnode["min_v"] = v
#         return v
#     else:
#         visited_scores = []
#         mode = danger_level(root_state)
#         for sib in siblings:
#             if sib["n"] > 0:
#                 visited_scores.append(risk_averse_score(sib, mode=mode))

#         if visited_scores:
#             prior_mean = sum(visited_scores) / len(visited_scores)
#             prior_worst = min(visited_scores)
#             v = 0.5 * prior_mean + 0.5 * prior_worst
#         else:
#             v = evaluate_state(root_state)

#         dnode["n"] = 1
#         dnode["sum_v"] = v
#         dnode["min_v"] = v
#         return v

# # =========================
# # Root enemy head kill zones
# # =========================

# def compute_enemy_head_kill_zones(s):
#     """
#     Squares enemy heads can occupy next turn:
#       (x,y) -> max enemy length that can hit that square.
#     """
#     yi = s["yi"]
#     snakes = s["sn"]
#     w, h = s["w"], s["h"]

#     kill_map = {}
#     for i, sn in enumerate(snakes):
#         if not sn["alive"] or i == yi:
#             continue
#         body = sn["b"]
#         if not body:
#             continue
#         hx, hy = body[0]
#         enemy_len = len(sn["b"])
#         sm = safe_moves(s, i)
#         if not sm:
#             continue
#         for mv in sm:
#             dx, dy = DELTA[mv]
#             nx, ny = hx + dx, hy + dy
#             if not in_bounds(nx, ny, w, h):
#                 continue
#             prev = kill_map.get((nx, ny), 0)
#             if enemy_len > prev:
#                 kill_map[(nx, ny)] = enemy_len
#     return kill_map

# # =========================
# # Main move choice
# # =========================

# def choose_move_minimax_tree_from_json(
#     j,
#     I=I_DEFAULT,
#     max_depth=D_DEFAULT,
#     use_one_step_fallback=USE_ONE_STEP_FALLBACK_FOR_UNVISITED,
#     debug=False,
# ):
#     s0 = s_from_json(j)
#     root = make_game_node(state_copy(s0))

#     for _ in range(I):
#         simulate_once(root, max_depth=max_depth)

#     yi = s0["yi"]
#     my_snake = s0["sn"][yi]
#     my_len = len(my_snake["b"])
#     my_head_x, my_head_y = my_snake["b"][0]

#     if not root["children"]:
#         my_moves = safe_moves(s0, yi)
#         if not my_moves:
#             return "up"
#         return random.choice(my_moves)

#     mode = danger_level(s0)
#     move_scores = {}
#     dnodes = list(root["children"].values())

#     kill_map = compute_enemy_head_kill_zones(s0)

#     enemy_lengths_here = [v for v in kill_map.values()]
#     max_enemy_len_here = max(enemy_lengths_here) if enemy_lengths_here else 0
#     length_diff_global = my_len - max_enemy_len_here

#     for mv, dnode in root["children"].items():
#         if dnode["n"] == 0:
#             siblings = [sib for sib in dnodes if sib is not dnode]
#             _ = fallback_value_for_unvisited(s0, dnode, siblings, use_one_step=use_one_step_fallback)
#             score = risk_averse_score(dnode, mode=mode)
#         else:
#             score = risk_averse_score(dnode, mode=mode)

#         dx, dy = DELTA[mv]
#         nx, ny = my_head_x + dx, my_head_y + dy
#         enemy_len_here = kill_map.get((nx, ny), 0)

#         if enemy_len_here > 0:
#             if enemy_len_here >= my_len + 1:
#                 penalty_scale = 1.5 if length_diff_global <= -3 else 1.0
#                 score -= BASE_HEAD_KILL_PENALTY * penalty_scale
#             elif my_len >= enemy_len_here + 2:
#                 score += 800.0  # encourage kill squares we win

#         move_scores[mv] = score

#     best_move = max(move_scores.items(), key=lambda kv: kv[1])[0]

#     if debug:
#         print("Move scores:", move_scores)
#         print("Chosen move:", best_move)
#         print("Danger mode:", mode)

#     return best_move

# def handle_move_request(
#     request_body,
#     I=I_DEFAULT,
#     max_depth=D_DEFAULT,
#     use_one_step_fallback=USE_ONE_STEP_FALLBACK_FOR_UNVISITED,
#     debug=False,
# ):
#     mv = choose_move_minimax_tree_from_json(
#         request_body,
#         I=I,
#         max_depth=max_depth,
#         use_one_step_fallback=use_one_step_fallback,
#         debug=debug,
#     )
#     return {
#         "move": mv,
#         "shout": f"tree: {mv}",
#     }

# # =========================
# # Flask wiring
# # =========================

# app = Flask(__name__)

# @app.get("/")
# def index():
#     return jsonify({
#         "apiversion": "1",
#         "author": "geeked",
#         "color": "#00FFAA",
#         "head": "default",
#         "tail": "default"
#     })

# @app.post("/start")
# def start():
#     data = request.get_json()
#     return jsonify({"taunt": "glhf"})

# @app.post("/move")
# def move():
#     data = request.get_json()
#     move_resp = handle_move_request(
#         data,
#         I=I_DEFAULT,
#         max_depth=D_DEFAULT,
#         use_one_step_fallback=USE_ONE_STEP_FALLBACK_FOR_UNVISITED,
#         debug=False,
#     )
#     return jsonify(move_resp)

# @app.post("/end")
# def end():
#     data = request.get_json()
#     return jsonify({"status": "ok"})

# app.run(host="0.0.0.0", port=8000, debug=False)
