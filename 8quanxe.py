import heapq
import math
from queue import PriorityQueue
import time
import tkinter as tk
import random
from collections import deque
from tkinter import ttk

class EightRooks:
    def __init__(self):
        self.n = 8
        self.pos_rook = [[0]*self.n for _ in range(self.n)]
        self.rook_input = []
        self.rook_left = []
        self.rook_right = []
        self.solutions = {}
        self.alpha = 0.95
        self.T_min = 0.01
        self.is_running = False
        self.can_edit = True
        self._report_window = None
        self._report_figs = []
        self._report_canvases = []
        self.result_labels = {}
        self.algorithm_stats = {}
        self.create_gui()

    def run_all_algorithms(self, only_missing: bool = True):
        init_state = []
        for i in range(self.n):
            for j in range(self.n):
                if self.pos_rook[i][j] == 1:
                    init_state.append((i, j))

        algorithms = [
            ('bfs', lambda s: self.bfs(s)),
            ('dfs', lambda s: self.dfs(s)),
            ('ucs', lambda s: self.ucs(s)),
            ('ids', lambda s: self.IDS(s)),
            ('dls', lambda s: self.DLS(s)),
            ('a_star', lambda s: self.A_star(s)),
            ('greedy', lambda s: self.Greedy(s)),
            ('hill_climbing', lambda s: self.hill_climbing(s)),
            ('simulated_annealing', lambda s: self.simulated_annealing(s)),
            ('genetic_algorithm', lambda s: self.genetic_algorithm(s)),
            ('beam_search', lambda s: self.beam_search(s)),
            ('and_or', lambda s: self.and_or_search(s)),
            ('belief_search', lambda s: self.belief_search(s)),
            ('backtracking', lambda s: self.backtracking(s)),
            ('forward_checking', lambda s: self.forward_checking(s)),
            ('ac3', lambda s: self.ac3(s)),
            ('partially_observable', lambda s: self.partially_observable_search(s)),
        ]

        prev_running = self.is_running
        self.is_running = False

        for key, func in algorithms:
            if only_missing:
                st = self.algorithm_stats.get(key, {})
                if isinstance(st, dict) and ('time' in st):
                    continue
            start = time.time()
            try:
                func(init_state)
            except Exception:
                pass
            end = time.time()
            if key not in self.algorithm_stats:
                self.algorithm_stats[key] = {}
            self.algorithm_stats[key]['time'] = end - start

            if key == 'and_or':
                self.algorithm_stats[key].setdefault('nodes_expanded', '')
                self.algorithm_stats[key].setdefault('max_frontier', '')

        self.is_running = prev_running

    def bfs(self, state):
        nodes_expanded = 0
        max_frontier = 0
        
        q = deque()
        q.append(state)

        while q:
            max_frontier = max(max_frontier, len(q))
            state = q.popleft()
            nodes_expanded += 1
            
            if len(state) == self.n:
                self.solutions['bfs'] = state.copy()
                self.algorithm_stats['bfs'] = {
                    'nodes_expanded': nodes_expanded,
                    'max_frontier': max_frontier,
                }
                return state.copy()

            for row in range(self.n):
                if all(r != row for r, c in state):
                    for col in range(self.n):
                        if all(c != col for r, c in state):
                            q.append(state + [(row, col)])
                    break

    def dfs(self, state):
        nodes_expanded = 0
        max_frontier = 0
        
        stack = [state]

        while stack:
            max_frontier = max(max_frontier, len(stack))
            state = stack.pop()
            nodes_expanded += 1
            
            if len(state) == self.n:
                self.solutions['dfs'] = state.copy()
                self.algorithm_stats['dfs'] = {
                    'nodes_expanded': nodes_expanded,
                    'max_frontier': max_frontier,
                }
                return state.copy()
            for row in range(self.n):
                if all(r != row for r, c in state):
                    for col in range(self.n):
                        if all(c != col for r, c in state):
                            stack.append(state + [(row, col)])
                    break

    def ucs(self, state):
        frontier = PriorityQueue()
        frontier.put((0, state))
        explored = set()

        nodes_expanded = 0
        max_frontier = 1

        while not frontier.empty():
            cost, curr_state = frontier.get()

            nodes_expanded += 1
            max_frontier = max(max_frontier, frontier.qsize())

            if len(curr_state) == self.n:
                self.solutions['ucs'] = curr_state.copy()
                self.algorithm_stats['ucs'] = {
                    "nodes_expanded": nodes_expanded,
                    "max_frontier": max_frontier,
                    "cost_g": cost,
                }
                return curr_state.copy()

            explored.add(tuple(curr_state))

            used_rows = {r for r, _ in curr_state}
            used_cols = {c for _, c in curr_state}

            for row in range(self.n):
                if row in used_rows:
                    continue
                for col in range(self.n):
                    if col in used_cols:
                        continue
                    new_state = curr_state + [(row, col)]
                    new_cost = cost + 1
                    child = (new_cost, new_state)
                    if tuple(new_state) not in explored and child not in frontier.queue:
                        frontier.put(child)
                    else:
                        for index, node in enumerate(frontier.queue):
                            if node[1] == new_state and node[0] > new_cost:
                                del frontier.queue[index]
                                frontier.put(child)
                                break
                    break
                break

    def IDS(self, state):
        start = state
        total_nodes_expanded = 0
        total_max_frontier = 0

        for depth in range(1, self.n * 2):
            result = self.DLS(start, depth)

            if 'DLS' in self.algorithm_stats:
                total_nodes_expanded += self.algorithm_stats['DLS']['nodes_expanded']
                total_max_frontier = max(total_max_frontier, self.algorithm_stats['DLS']['max_frontier'])

            if result != 'cutoff' and result:
                self.solutions['ids'] = result
                self.algorithm_stats['ids'] = {
                    "nodes_expanded": total_nodes_expanded,
                    "max_frontier": total_max_frontier,
                }
                return result
            
    def DLS(self, state, limit=7):
        stack = [(state, limit)]
        nodes_expanded = 0
        max_frontier = 1

        while stack:
            current_state, limit = stack.pop()
            nodes_expanded += 1
            max_frontier = max(max_frontier, len(stack))

            if self.goal_test(current_state):
                self.solutions['dls'] = current_state.copy()
                self.algorithm_stats['dls'] = {
                    "nodes_expanded": nodes_expanded,
                    "max_frontier": max_frontier,
                }
                return current_state.copy()

            if not limit:
                continue

            used_rows = {r for r, _ in current_state}
            used_cols = {c for _, c in current_state}

            for row in range(self.n):
                if row in used_rows:
                    continue
                for col in range(self.n):
                    if col in used_cols:
                        continue
                    stack.append((current_state + [(row, col)], limit - 1))
                break
        return 'cutoff'
        
    def Greedy(self, state):
        nodes_expanded = 0
        max_frontier = 0
        
        q = []
        heapq.heappush(q, (self.h(state), state))
        
        while q:
            max_frontier = max(max_frontier, len(q))
            _, state = heapq.heappop(q)
            nodes_expanded += 1
            
            if self.goal_test(state): 
                self.solutions['greedy'] = state.copy()
                self.algorithm_stats['greedy'] = {
                    'nodes_expanded': nodes_expanded,
                    'max_frontier': max_frontier,
                    'heuristic': self.h(state),
                }
                return state.copy()

            for row in range(self.n):
                if all(r != row for r, c in state):
                    for col in range(self.n):
                        if all(c != col for r, c in state):
                            new_state = state + [(row, col)]
                            heapq.heappush(q, (self.h(new_state), new_state))
                    break
        
    def h(self, state):
        occupied_rows = {r for r, c in state}
        occupied_cols = {c for r, c in state}
        empty_rows = self.n - len(occupied_rows)
        empty_cols = self.n - len(occupied_cols)
        return max(empty_rows, empty_cols)

    def A_star(self, state):
        nodes_expanded = 0
        max_frontier = 0
        
        q = []
        g = len(state)
        h = self.h(state)
        f = g + h
        heapq.heappush(q, (f, g, state))

        visited = set()

        while q:
            max_frontier = max(max_frontier, len(q))
            f, g, state = heapq.heappop(q)
            nodes_expanded += 1
            
            if self.goal_test(state):
                self.solutions['a_star'] = state.copy()
                self.algorithm_stats['a_star'] = {
                    'nodes_expanded': nodes_expanded,
                    'max_frontier': max_frontier,
                    'heuristic': self.h(state),
                    'cost_g': g,
                    'total_cost_f': f
                }
                return state.copy()

            state_key = tuple(state)
            if state_key in visited:
                continue
            visited.add(state_key)

            for row in range(self.n):
                if all(r != row for r, c in state):
                    for col in range(self.n):
                        if all(c != col for r, c in state):
                            new_state = state + [(row, col)]
                            g_new = g + 1
                            h_new = self.h(new_state)
                            f_new = g_new + h_new
                            heapq.heappush(q, (f_new, g_new, new_state))
                    break

    def get_neighbors(self, state):
        neighbors = []
        used_rows = {r for r, c in state}
        used_cols = {c for r, c in state}

        # thêm một quân xe mới ở hàng hoặc cột chưa có
        for row in range(self.n):
            if row not in used_rows:
                for col in range(self.n):
                    if col not in used_cols:
                        new_state = state + [(row, col)]
                        neighbors.append(new_state)
        return neighbors

    def hill_climbing(self, state):
        current = state
        nodes_expanded = 0
        max_frontier = 1

        while True:
            neighbors = self.get_neighbors(current)
            nodes_expanded += 1
            max_frontier = max(max_frontier, len(neighbors))
            if not neighbors:
                break
            neighbor = min(neighbors, key=self.h)
            if self.h(neighbor) >= self.h(current):
                break
            current = neighbor
            if self.goal_test(current):
                heuristic = self.h(current)
                self.solutions['hill_climbing'] = current.copy()
                self.algorithm_stats['hill_climbing'] = {
                    "nodes_expanded": nodes_expanded,
                    "max_frontier": max_frontier,
                    "heuristic": heuristic,
                }
                return current.copy()
        self.status_label.config(text="Hill Climbing dừng lại (kẹt local minima)")
        return current.copy()

    def simulated_annealing(self, state):
        self.T = 100
        current = state
        nodes_expanded = 0
        max_frontier = 1

        while self.T > self.T_min:
            nodes_expanded += 1

            if self.goal_test(current):
                heuristic = self.h(current)
                self.solutions['simulated_annealing'] = current.copy()
                self.algorithm_stats['simulated_annealing'] = {
                    "nodes_expanded": nodes_expanded,
                    "max_frontier": max_frontier,
                    "heuristic": heuristic,
                }
                return current.copy()

            neighbors = []
            for row in range(self.n):
                if all(r != row for r, c in current):
                    for col in range(self.n):
                        if all(c != col for r, c in current):
                            new_state = current + [(row, col)]
                            neighbors.append(new_state)

            if not neighbors:
                break

            max_frontier = max(max_frontier, len(neighbors))
            next_state = random.choice(neighbors)

            delta = self.h(current) - self.h(next_state)

            if delta > 0:
                current = next_state
            else:
                p = math.exp(delta / self.T)
                if random.random() < p:
                    current = next_state

            self.T *= self.alpha

        self.status_label.config(text="Simulated Annealing kết thúc (hết nhiệt độ)")
        return current.copy()

    def beam_search(self, state, k=2):
        beam = [(self.h(state), state)]
        nodes_expanded = 0
        max_frontier = 1

        while beam:
            new_beam = []
            nodes_expanded += 1
            max_frontier = max(max_frontier, len(beam))

            for _, curr_state in beam:
                if self.goal_test(curr_state):
                    heuristic = self.h(curr_state)
                    self.solutions['beam_search'] = curr_state.copy()
                    self.algorithm_stats['beam_search'] = {
                        "nodes_expanded": nodes_expanded,
                        "max_frontier": max_frontier,
                        "heuristic": heuristic,
                    }
                    return curr_state.copy()

                used_rows = {r for r, _ in curr_state}
                used_cols = {c for _, c in curr_state}

                for row in range(self.n):
                    if row in used_rows:
                        continue
                    for col in range(self.n):
                        if col in used_cols:
                            continue
                        new_state = curr_state + [(row, col)]
                        heapq.heappush(new_beam, (self.h(new_state), new_state))
                    break 
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:k]
        return None

    def genetic_algorithm(self, state, pop_size=50, generations=200, mutation_rate=0.1):
        def random_individual():
            cols = list(range(self.n))
            random.shuffle(cols)
            return [(i, cols[i]) for i in range(self.n)]

        def fitness(ind):
            return -self.h(ind)

        def select(population):
            weights = [max(fitness(ind), 0) + 1 for ind in population]
            total = sum(weights)
            probs = [w / total for w in weights]
            return random.choices(population, weights=probs, k=1)[0]

        def reproduce(x, y):
            n = self.n
            c = random.randint(1, n - 1)
            child_cols = [col for (_, col) in x[:c]]
            for (_, col) in y:
                if col not in child_cols:
                    child_cols.append(col)
            return [(i, child_cols[i]) for i in range(n)]

        def mutate(ind):
            ind = ind[:]
            i, j = random.sample(range(self.n), 2)
            ind[i], ind[j] = (ind[i][0], ind[j][1]), (ind[j][0], ind[i][1])
            return ind

        population = [random_individual() for _ in range(pop_size - 1)] + [state]

        nodes_expanded = 0
        max_frontier = pop_size
        best = None

        for gen in range(generations):
            nodes_expanded += pop_size
            max_frontier = max(max_frontier, len(population))

            population.sort(key=lambda ind: self.h(ind))
            best = population[0]

            if self.goal_test(best):
                heuristic = self.h(best)
                self.solutions['genetic_algorithm'] = best.copy()
                self.algorithm_stats['genetic_algorithm'] = {
                    "nodes_expanded": nodes_expanded,
                    "max_frontier": max_frontier,
                    "heuristic": heuristic,
                }
                return best.copy()

            new_population = []
            for _ in range(pop_size):
                x = select(population)
                y = select(population)
                child = reproduce(x, y)
                if random.random() < mutation_rate:
                    child = mutate(child)
                new_population.append(child)
            population = new_population

        best = min(population, key=lambda ind: self.h(ind))
        self.solutions['genetic_algorithm'] = best.copy()
        return best.copy()

    def and_or_search(self, state):
        self.or_search(state, [])

    def or_search(self, state, path):
        current_state = list(state)

        if self.goal_test_and_or(current_state):
            self.solutions['and_or'] = current_state.copy()
            return []

        state_signature = tuple(current_state)
        if state_signature in path:
            return None

        for action in self.get_actions(current_state):
            results = self.get_results(current_state, action)
            subplan = self.and_search(results, path + [state_signature])
            if subplan is not None:
                if isinstance(subplan, list):
                    return [action] + subplan
                else:
                    return [action, subplan]
        return None

    def and_search(self, states, path):
        plans = []
        for next_state in states:
            plan = self.or_search(next_state, path)
            if plan is None:
                return None
            plans.append(plan)
        if not plans:
            return []
        if len(plans) == 1:
            return plans[0]
        return plans
    
    def get_results(self, state, action):
        return [list(state) + [action]]
    
    def belief_search(self, state):
        self.algorithm_stats['belief_search'] = {
            "nodes_expanded": 0,
            "max_frontier": 0,
            "heuristic": 0,
        }

        queue = [(state, 1)]
        visited = set()
        max_frontier_size = 1

        try:
            while queue:
                max_frontier_size = max(max_frontier_size, len(queue))
                self.algorithm_stats['belief_search']["max_frontier"] = max_frontier_size
                queue.sort(key=lambda x: x[1], reverse=True)
                state, _ = queue.pop(0)
                state_tuple = tuple(map(tuple, state))
                self.algorithm_stats['belief_search']["nodes_expanded"] += 1
                if state_tuple in visited:
                    continue
                visited.add(state_tuple)
                if self.goal_test(state):
                    self.solutions['belief_search'] = state.copy()
                    heuristic = self.h(state)
                    self.algorithm_stats['belief_search'].update({
                        "heuristic": heuristic,
                    })
                    return state.copy()

                if len(state) >= self.n:
                    continue
                actions = self.get_actions(state)

                for action in actions:
                    next_state = self.results(state, action)
                    num_rooks = len(next_state)
                    rows_used = len(set(r for r, _ in next_state))
                    cols_used = len(set(c for _, c in next_state))
                    score = num_rooks + (rows_used == num_rooks) + (cols_used == num_rooks)
                    next_state_tuple = tuple(map(tuple, next_state))
                    if next_state_tuple not in visited:
                        queue.append((next_state, score))

        except Exception as e:
            raise e

    def get_actions(self, state):
        occupied_rows = {r for r, _ in state}
        occupied_cols = {c for _, c in state}
        actions = []
        for row in range(self.n):
            if row in occupied_rows:
                continue
            for col in range(self.n):
                if col in occupied_cols:
                    continue
                actions.append((row, col))
        return actions

    def results(self, state, action):
        return state + [action]

    def goal_test(self, state):
        if len(state) != self.n:
            return False
        rows = {r for r, c in state}
        cols = {c for r, c in state}
        return len(rows) == self.n and len(cols) == self.n
    
    def goal_test_and_or(self, state):
        if len(state) != self.n:
            return False
        rows = set()
        cols = set()
        for r, c in state:
            if r in rows or c in cols:
                return False
            rows.add(r)
            cols.add(c)
        return True
    
    def is_valid(self, state, action):
        row, col = action
        for r, c in state:
            if r == row or c == col:
                return False
        return True

    def partially_observable_search(self, state):
        nodes_expanded = 0
        max_frontier = 1
        
        # Start with initial belief state - we know the placed rooks but have uncertainty about the rest
        belief_state = [state.copy()]
        
        # For tracking the search process
        visited_beliefs = set()
        
        while belief_state:
            max_frontier = max(max_frontier, len(belief_state))
            current_belief = belief_state.pop(0)
            nodes_expanded += 1
            
            # Convert belief to tuple for hashing
            belief_key = tuple(sorted(current_belief))
            if belief_key in visited_beliefs:
                continue
            visited_beliefs.add(belief_key)
            
            # If this belief state contains a complete solution, return it
            if self.goal_test(current_belief):
                self.solutions['partially_observable'] = current_belief.copy()
                self.algorithm_stats['partially_observable'] = {
                    'nodes_expanded': nodes_expanded,
                    'max_frontier': max_frontier,
                    'heuristic': self.h(current_belief),
                }
                return current_belief.copy()

            observable_rows = self.get_observable_rows(current_belief)

            for row in observable_rows:
                if any(r == row for r, _ in current_belief):
                    continue
                for col in range(self.n):
                    if any(c == col for _, c in current_belief):
                        continue
                    new_belief = current_belief + [(row, col)]
                    if self.is_belief_consistent(new_belief):
                        belief_state.append(new_belief)
                    break
                break
        
        best_solution = max(visited_beliefs, key=lambda x: len(x))
        self.solutions['partially_observable'] = list(best_solution)
        return list(best_solution)

    def get_observable_rows(self, state):
        placed_rows = {r for r, _ in state}

        if len(state) <= 2:
            return [r for r in range(min(4, self.n)) if r not in placed_rows]
        elif len(state) <= 4:
            return [r for r in range(min(6, self.n)) if r not in placed_rows][:3]
        else:
            return [r for r in range(self.n) if r not in placed_rows][:2]

    def is_belief_consistent(self, belief):
        rows = set()
        cols = set()
        for r, c in belief:
            if r in rows or c in cols:
                return False
            rows.add(r)
            cols.add(c)
        return True
    
    def backtracking(self, state):
        nodes_expanded = 0
        max_frontier = 1
        def _bt(curr, depth):
            nonlocal nodes_expanded, max_frontier
            nodes_expanded += 1
            max_frontier = max(max_frontier, depth)

            if len(curr) == self.n:
                return curr.copy()
            for action in self.get_actions(curr):
                if self.is_valid(curr, action):
                    curr.append(action)
                    res = _bt(curr, depth + 1)
                    if res:
                        return res
                    curr.pop()
            return None

        result = _bt(list(state), max(1, len(state)))
        if result:
            self.solutions["backtracking"] = result.copy()

        self.algorithm_stats["backtracking"] = {
            "nodes_expanded": nodes_expanded,
            "max_frontier": max_frontier,
        }
        return result
    
    def forward_checking(self, state, domains=None):
        nodes_expanded = 0
        max_frontier = 1

        if domains is None:
            domains = {r: [c for c in range(self.n)] for r in range(self.n)}
        for r, c in state:
            domains[r] = [c]
            for rr in range(self.n):
                if rr != r and c in domains[rr]:
                    domains[rr].remove(c)

        def _fc(curr, doms, depth):
            nonlocal nodes_expanded, max_frontier
            nodes_expanded += 1
            max_frontier = max(max_frontier, depth)
            if len(curr) == self.n:
                return curr.copy()
            used_rows = {r for r, _ in curr}
            next_row = next(r for r in range(self.n) if r not in used_rows)
            for col in doms[next_row]:
                action = (next_row, col)
                if not self.is_valid(curr, action):
                    continue
                new_domains = {r: doms[r].copy() for r in doms}
                for rr in range(self.n):
                    if rr != next_row and col in new_domains[rr]:
                        new_domains[rr].remove(col)
                if any(len(new_domains[r]) == 0 for r in new_domains if r not in used_rows and r != next_row):
                    continue
                curr.append(action)
                res = _fc(curr, new_domains, depth + 1)
                if res:
                    return res
                curr.pop()
            return None
        
        result = _fc(list(state), domains, max(1, len(state)))
        if result:
            self.solutions["forward_checking"] = result.copy()
        self.algorithm_stats["forward_checking"] = {
            "nodes_expanded": nodes_expanded,
            "max_frontier": max_frontier,
        }
        return result

    def revise(self, domains, xi, xj):
        revised = False
        to_remove = []
        for x in domains[xi]:
            if all(x == y for y in domains[xj]):
                to_remove.append(x)
                revised = True
        for x in to_remove:
            domains[xi].remove(x)
        return revised

    def ac3(self, state, domains=None):
        nodes_expanded = 0
        max_frontier = 1
        if domains is None:
            domains = {r: [c for c in range(self.n)] for r in range(self.n)}
        for r, c in state:
            domains[r] = [c]
            for rr in range(self.n):
                if rr != r and c in domains[rr]:
                    domains[rr].remove(c)

        def _ac3(curr, doms, depth):
            nonlocal nodes_expanded, max_frontier
            nodes_expanded += 1
            max_frontier = max(max_frontier, depth)
            if len(curr) == self.n:
                return curr.copy()
            used_rows = {r for r, _ in curr}
            next_row = next(r for r in range(self.n) if r not in used_rows)
            queue = deque((xi, xj) for xi in range(self.n) for xj in range(self.n) if xi != xj)
            while queue:
                max_frontier = max(max_frontier, len(queue))
                xi, xj = queue.popleft()
                if self.revise(doms, xi, xj):
                    if len(doms[xi]) == 0:
                        return None
                    for xk in range(self.n):
                        if xk != xi and xk != xj:
                            queue.append((xk, xi))

            for col in doms[next_row]:
                action = (next_row, col)
                if not self.is_valid(curr, action):
                    continue
                new_domains = {r: doms[r].copy() for r in doms}
                for rr in range(self.n):
                    if rr != next_row and col in new_domains[rr]:
                        new_domains[rr].remove(col)
                curr.append(action)
                res = _ac3(curr, new_domains, depth + 1)
                if res:
                    return res
                curr.pop()
            return None

        result = _ac3(list(state), domains, max(1, len(state)))
        if result:
            self.solutions["ac3"] = result.copy()

        self.algorithm_stats["ac3"] = {
            "nodes_expanded": nodes_expanded,
            "max_frontier": max_frontier
        }
        return result

    def on_root_close(self):
        self.is_running = False
        try:
            if self._report_window and self._report_window.winfo_exists():
                self._close_report_window()
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except Exception:
                pass
        finally:
            self.root.destroy()

    def _close_report_window(self):
        # destroy canvases and figures to free all matplotlib resources
        try:
            for canvas in getattr(self, "_report_canvases", []):
                try:
                    widget = canvas.get_tk_widget()
                    widget.destroy()
                except Exception:
                    pass
            # close figures
            try:
                import matplotlib.pyplot as plt
                for fig in getattr(self, "_report_figs", []):
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
            except Exception:
                pass
        finally:
            if self._report_window and self._report_window.winfo_exists():
                self._report_window.destroy()
            self._report_window = None
            self._report_figs = []
            self._report_canvases = []

    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("8 Rooks Problem")
        self.root.geometry("1100x830")
        self.root.protocol("WM_DELETE_WINDOW", self.on_root_close)
        self.root.resizable(False, False)
        self.root.configure(bg='#000000')

        try:
            from PIL import Image, ImageTk
            rook_img = Image.open("rook.png")
            rook_img = rook_img.resize((40, 40))
            self.xe = ImageTk.PhotoImage(rook_img)
            self.img_null = ImageTk.PhotoImage(Image.new('RGBA', (40, 40), (0, 0, 0, 0)))
        except Exception as e:
            print(f"Lỗi khi load ảnh: {e}")
            self.xe = None
            self.img_null = None

        main_frame = tk.Frame(self.root, bg="#FFFED3")
        main_frame.pack(fill=tk.BOTH, expand=True)

        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_columnconfigure(2, weight=1)

        result_frame = tk.LabelFrame(main_frame, text="Kết quả", bg="#FFFED3", fg='black', font=('Arial', 11, 'bold'), padx=20, pady=5)
        result_frame.grid(column=2, rowspan=1, padx=10, pady=5, sticky='nw')

        tk.Label(result_frame, text="Tham số", font=("Segoe UI", 10, "bold"), bg="#FFFED3").grid(row=0, column=0, padx=10, sticky="w")
        tk.Label(result_frame, text="Giá trị", font=("Segoe UI", 10, "bold"), bg="#FFFED3").grid(row=0, column=1, padx=10)

        params = {
            "Nodes Expanded": "",
            "Max Frontier Size": "",
            "Time (seconds)": "",
            "Heuristic h(x)": "",
        }
        for i, (key, value) in enumerate(params.items(), start=1):
            tk.Label(result_frame, text=key, bg="#FFFED3").grid(row=i, column=0, sticky="w", padx=10, pady=2)
            lbl = tk.Label(result_frame, text=value, bg="#FFFED3")
            lbl.grid(row=i, column=1, padx=10, pady=2)
            self.result_labels[key] = lbl

        board1_frame = tk.LabelFrame(main_frame, text="Bàn cờ 1 - Đặt quân xe (nhấn để đặt hoặc xóa)", 
                                     bg='#FFFED3', fg='black', font=('Arial', 11, 'bold'),
                                     padx=10, pady=10)
        board1_frame.grid(row=0, column=0, padx=10, pady=5, sticky='nw')

        for i in range(self.n):
            row_buttons = []
            for j in range(self.n):
                color = '#f0d9b5' if (i + j) % 2 == 0 else '#b58863'
                btn = tk.Button(board1_frame, image=self.img_null, width=40, height=40, bg=color, borderwidth=0, relief="flat", highlightthickness=0,
                               command=lambda r=i, c=j: self.place_rook(r, c))
                btn.grid(row=i, column=j, padx=1, pady=1)
                row_buttons.append(btn)
            self.rook_input.append(row_buttons)

        algorithm_frame = tk.LabelFrame(main_frame, text="Lựa chọn thuật toán", 
                                        bg="#FFFED3", fg='black', font=('Arial', 11, 'bold'),
                                        padx=10, pady=5, height=370, width=360)
        algorithm_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=5, sticky='nw')
        algorithm_frame.grid_propagate(False)

        control_frame = tk.LabelFrame(main_frame, 
                                      text="Điều khiển",
                                        bg="#FFFED3", fg='black', font=('Arial', 11, 'bold'),
                                        padx=10, pady=1, height=150)
        control_frame.grid(row=2, columnspan=2, padx=10, pady=1, sticky='sw')
        control_frame.grid_propagate(False)

        self.selected_algorithm = tk.StringVar(value="BFS")

        left_col = tk.Frame(algorithm_frame, bg="#FFFED3")
        right_col = tk.Frame(algorithm_frame, bg="#FFFED3")
        left_col.pack(side="left", padx=5)
        right_col.pack(side="left", padx=5)

        def make_group(parent, title, options):
            f = tk.LabelFrame(parent, text=title, bg="#FFFED3", fg='black', font=('Arial', 10, 'bold'), pady=5)
            f.pack(fill="x", pady=5, anchor="nw")
            for text in options:
                tk.Radiobutton(f, text=text, value=text, variable=self.selected_algorithm, bg="#FFFED3", command=lambda t=text: self.status_label.config(text=f"Thuật toán đã chọn: {t}")).pack(anchor="nw")

        make_group(left_col, "Uninformed search", ["BFS", "DFS", "UCS", "DLS", "IDS"])
        make_group(left_col, "Local search", ["Hill Climbing", "Simulated Annealing", "Genetic Algorithm", "Beam Search"])

        make_group(right_col, "Informed search", ["A*", "Greedy"])
        make_group(right_col, "CSP Algorithms", ["Backtracking", "Forward Checking", "AC-3"])
        make_group(right_col, "Complex Environment", ["AND-OR Tree Search", "Belief State Search", "Partially Observable Search"])
        
        button_frame = tk.Frame(control_frame, bg='#FFFED3')
        button_frame.pack(pady=5, padx=5, fill='x')
        
        self.btn_run = tk.Button(button_frame, text="Chạy", width=8, height=1,
                                bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                                command=self.run_algorithm)
        self.btn_run.pack(side='left', padx=2)
        
        self.btn_simulate = tk.Button(button_frame, text="Mô phỏng lại", width=12, height=1,
                                      bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                                      command=self.simulate_algorithm, state='disabled')
        self.btn_simulate.pack(side='left', padx=2)
        
        self.btn_stop = tk.Button(button_frame, text="Dừng", width=8, height=1,
                                 bg='#f44336', fg='white', font=('Arial', 10, 'bold'),
                                 command=self.stop_simulation)
        self.btn_stop.pack(side='left', padx=2)
        
        self.btn_reset = tk.Button(button_frame, text="Reset", width=8, height=1,
                                   bg='#FF9800', fg='white', font=('Arial', 10, 'bold'),
                                   command=self.reset_board)
        self.btn_reset.pack(side='left', padx=2)

        self.btn_report = tk.Button(button_frame, text="So sánh", width=12, height=1,
                                    bg='#9C27B0', fg='white', font=('Arial', 10, 'bold'),
                                    command=self.show_report)
        self.btn_report.pack(side='left', padx=2)

        self.status_label = tk.Label(main_frame, text="Chọn một ô trên bàn cờ 1 để đặt quân xe", 
                                     bg="#FFFED3", fg='black', font=('Arial', 10))
        self.status_label.grid(row=2, columnspan=4, sticky='e', padx=50, pady=5)

        board2_frame = tk.LabelFrame(main_frame, text="Bàn cờ 2 - Mô phỏng", 
                                     bg="#FFFED3", fg='black', font=('Arial', 11, 'bold'),
                                     padx=10, pady=5)
        board2_frame.grid(row=1, column=0, padx=10, pady=5, sticky='w')

        for i in range(self.n):
            row_buttons = []
            for j in range(self.n):
                color = '#f0d9b5' if (i + j) % 2 == 0 else '#b58863'
                btn = tk.Button(board2_frame, image=self.img_null, width=40, height=40, bg=color, borderwidth=0, relief="flat", highlightthickness=0)
                btn.grid(row=i, column=j, padx=1, pady=1)
                row_buttons.append(btn)
            self.rook_left.append(row_buttons)

        board3_frame = tk.LabelFrame(main_frame, text="Bàn cờ 3 - Kết quả", 
                                     bg="#FFFED3", fg='black', font=('Arial', 11, 'bold'),
                                     padx=10, pady=5)
        board3_frame.grid(row=1, column=1, padx=10, pady=5, sticky='w')

        for i in range(self.n):
            row_buttons = []
            for j in range(self.n):
                color = '#f0d9b5' if (i + j) % 2 == 0 else '#b58863'
                btn = tk.Button(board3_frame, image=self.img_null, width=40, height=40, bg=color, borderwidth=0, relief="flat", highlightthickness=0)
                btn.grid(row=i, column=j, padx=1, pady=1)
                row_buttons.append(btn)
            self.rook_right.append(row_buttons)
        
        self.buttons_r = self.rook_right

        self.root.mainloop()

    def show_report(self):
        # Bổ sung dữ liệu còn thiếu để có thể hiển thị báo cáo ngay
        try:
            self.run_all_algorithms(only_missing=True)
        except Exception as e:
            print("run_all_algorithms error:", e)
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            report_window = tk.Toplevel(self.root)
            report_window.title("Báo cáo so sánh thuật toán")
            report_window.geometry("1000x600")
            self._report_window = report_window
            self._report_figs = []
            self._report_canvases = []

            # ensure proper cleanup when closing the report window
            report_window.protocol("WM_DELETE_WINDOW", self._close_report_window)
            notebook = ttk.Notebook(report_window)
            notebook.pack(fill='both', expand=True)
            
            time_frame = ttk.Frame(notebook)
            notebook.add(time_frame, text='Thời gian chạy')
            
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            algorithms = [a for a in self.algorithm_stats.keys() if 'time' in self.algorithm_stats.get(a, {})]
            times = [float(self.algorithm_stats[alg]['time']) for alg in algorithms]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
            
            bars = ax1.bar(algorithms, times, color=colors[:len(algorithms)])
            ax1.set_xlabel('Thuật toán', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Thời gian (giây)', fontsize=12, fontweight='bold')
            ax1.set_title('So sánh thời gian chạy của các thuật toán', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}s',
                        ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            canvas1 = FigureCanvasTkAgg(fig1, time_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill='both', expand=True)
            
            # Tab 2: Biểu đồ Nodes Expanded
            nodes_frame = ttk.Frame(notebook)
            notebook.add(nodes_frame, text='Nodes Expanded')
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            nodes = [int(self.algorithm_stats[alg].get('nodes_expanded', 0) or 0) for alg in algorithms]
            bars2 = ax2.bar(algorithms, nodes, color=colors[:len(algorithms)])
            ax2.set_xlabel('Thuật toán', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Số lượng nodes', fontsize=12, fontweight='bold')
            ax2.set_title('So sánh số lượng nodes được mở rộng', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            canvas2 = FigureCanvasTkAgg(fig2, nodes_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill='both', expand=True)
            
            # Tab 3: Bảng so sánh chi tiết
            table_frame = ttk.Frame(notebook)
            notebook.add(table_frame, text='Bảng so sánh')
            
            # Tạo bảng
            tree = ttk.Treeview(table_frame, columns=('Algorithm', 'Time', 'Nodes', 'Max Frontier', 'h(x)', 'g(x)', 'f(x)'), show='headings')
            tree.heading('Algorithm', text='Thuật toán')
            tree.heading('Time', text='Thời gian (s)')
            tree.heading('Nodes', text='Nodes Expanded')
            tree.heading('Max Frontier', text='Max Frontier')
            tree.heading('h(x)', text='Heuristic h(x)')
            
            tree.column('Algorithm', width=150)
            tree.column('Time', width=120)
            tree.column('Nodes', width=120)
            tree.column('Max Frontier', width=120)
            tree.column('h(x)', width=100)
            
            for alg in algorithms:
                stats = self.algorithm_stats[alg]
                tree.insert('', 'end', values=(
                    alg,
                    f"{stats.get('time', 0):.4f}",
                    stats.get('nodes_expanded', ''),
                    stats.get('max_frontier', ''),
                    stats.get('heuristic', ''),
                ))
            
            tree.pack(fill='both', expand=True, padx=10, pady=10)
            
            self.status_label.config(text="Đã hiển thị báo cáo so sánh thuật toán")
            
        except ImportError:
            self.status_label.config(text="Cần cài đặt matplotlib: pip install matplotlib")
        except Exception as e:
            self.status_label.config(text=f"Lỗi khi tạo báo cáo: {str(e)}")
            print(f"Lỗi: {e}")

    def place_rook(self, row, col):
        if not self.can_edit:
            self.status_label.config(text="Không thể chỉnh sửa sau khi đã chạy thuật toán, vui lòng reset bàn cờ")
            return
        
         # Kiểm tra trùng cột và trùng hàng
        for r in range(self.n):
            if self.pos_rook[r][col] == 1 and r != row:
                self.status_label.config(text=f"Không thể đặt quân xe tại ({row}, {col}): trùng cột với ({r}, {col})")
                return
        for c in range(self.n):
            if self.pos_rook[row][c] == 1 and c != col:
                self.status_label.config(text=f"Không thể đặt quân xe tại ({row}, {col}): trùng hàng với ({row}, {c})")
                return
        
        if not self.pos_rook[row][col]:
            if self.xe:
                self.rook_input[row][col].config(image=self.xe)
            else:
                self.rook_input[row][col].config(text='♜', font=('Arial', 24))
            self.pos_rook[row][col] = 1
            self.status_label.config(text=f"Đã đặt quân xe tại vị trí ({row}, {col})")
        else:
            self.rook_input[row][col].config(image=self.img_null, text='')
            self.pos_rook[row][col] = 0
            self.status_label.config(text=f"Đã xóa quân xe tại vị trí ({row}, {col})")

    def draw_xe(self, state, board, is_delay=False, delay=0.5):
        import time
        
        for r, c in state:
            if not self.is_running:
                break
            if self.xe:
                board[r][c].config(image=self.xe)
                if is_delay:
                    self.root.update()
                    time.sleep(delay)
            else:
                board[r][c].config(text='♜', font=('Arial', 24))

    def run_algorithm(self):
        self.can_edit = False
        algorithm = self.selected_algorithm.get()
        self.reset_result_boards()
        self.status_label.config(text=f"Đang chạy thuật toán: {algorithm}")

        init_state = []
        
        for i in range(self.n):
            for j in range(self.n):
                if self.pos_rook[i][j] == 1:
                    init_state.append((i, j))
        
        try:
            if algorithm == "BFS":
                start = time.time()
                self.bfs(init_state)
                end = time.time()
                self.algorithm_stats['bfs']['time'] = end - start
                if self.solutions['bfs']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['bfs'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('bfs')
            elif algorithm == "DFS":
                start = time.time()
                self.dfs(init_state)
                end = time.time()
                self.algorithm_stats['dfs']['time'] = end - start
                if self.solutions['dfs']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['dfs'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('dfs')
            elif algorithm == "UCS":
                start = time.time()
                self.ucs(init_state)
                end = time.time()
                self.algorithm_stats['ucs']['time'] = end - start
                if self.solutions['ucs']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['ucs'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('ucs')
            elif algorithm == "IDS":
                start = time.time()
                self.IDS(init_state)
                end = time.time()
                self.algorithm_stats['ids']['time'] = end - start
                if self.solutions['ids']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['ids'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('ids')
            elif algorithm == "A*":
                start = time.time()
                self.A_star(init_state)
                end = time.time()
                self.algorithm_stats['a_star']['time'] = end - start
                if self.solutions['a_star']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['a_star'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('a_star')
            elif algorithm == "Greedy":
                start = time.time() 
                self.Greedy(init_state)
                end = time.time()
                self.algorithm_stats['greedy']['time'] = end - start
                if self.solutions['greedy']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['greedy'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('greedy')
            elif algorithm == "Hill Climbing":
                start = time.time()
                self.hill_climbing(init_state)
                end = time.time()
                self.algorithm_stats['hill_climbing']['time'] = end - start
                if self.solutions['hill_climbing']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['hill_climbing'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('hill_climbing')
            elif algorithm == "Simulated Annealing":
                start = time.time()
                self.simulated_annealing(init_state)
                end = time.time()
                self.algorithm_stats['simulated_annealing']['time'] = end - start
                if self.solutions['simulated_annealing']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['simulated_annealing'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('simulated_annealing')
            elif algorithm == "Genetic Algorithm":
                start = time.time()
                self.genetic_algorithm(init_state)
                end = time.time()
                self.algorithm_stats['genetic_algorithm']['time'] = end - start
                if self.solutions.get('genetic_algorithm'):
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['genetic_algorithm'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('genetic_algorithm')
            elif algorithm == "Beam Search":
                start = time.time()
                self.beam_search(init_state)
                end = time.time()
                self.algorithm_stats['beam_search']['time'] = end - start
                if self.solutions['beam_search']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['beam_search'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('beam_search')
            elif algorithm == "AND-OR Tree Search":
                start = time.time()
                self.and_or_search(init_state)
                end = time.time()
                self.algorithm_stats['and_or'] = {
                    'time': end - start,
                    'nodes_expanded': '',
                    'max_frontier':  '',
                    'heuristic': '',
                }
                if self.solutions['and_or']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['and_or'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('and_or')
            elif algorithm == "Belief State Search":
                start = time.time()
                self.belief_search(init_state)
                end = time.time()
                self.algorithm_stats['belief_search']['time'] = end - start
                if self.solutions['belief_search']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['belief_search'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('belief_search')
            elif algorithm == "Partially Observable Search":
                start = time.time()
                self.partially_observable_search(init_state)
                end = time.time()
                self.algorithm_stats['partially_observable']['time'] = end - start
                if self.solutions['partially_observable']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['partially_observable'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('partially_observable')
            elif algorithm == "DLS":
                start = time.time()
                self.DLS(init_state)
                end = time.time()
                self.algorithm_stats['dls']['time'] = end - start
                if self.solutions['dls']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['dls'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('dls')
            elif algorithm == "Backtracking":
                start = time.time()
                self.backtracking(init_state)
                end = time.time()
                self.algorithm_stats['backtracking']['time'] = end - start
                if self.solutions['backtracking']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['backtracking'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('backtracking')
            elif algorithm == "Forward Checking":
                start = time.time()
                self.forward_checking(init_state)
                end = time.time()
                self.algorithm_stats['forward_checking']['time'] = end - start
                if self.solutions['forward_checking']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['forward_checking'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('forward_checking')
            elif algorithm == "AC-3":
                start = time.time()
                self.ac3(init_state)
                end = time.time()
                self.algorithm_stats['ac3']['time'] = end - start
                if self.solutions['ac3']:
                    self.simulate_algorithm()
                    self.draw_xe(self.solutions['ac3'], self.buttons_r)
                    self.btn_simulate.config(state='normal')
                    self.update_result_display('ac3')
            else:
                self.status_label.config(text=f"Thuật toán {algorithm} chưa được triển khai")
        except Exception as e:
            self.status_label.config(text=f"Lỗi: {str(e)}")
            print(f"Lỗi: {e}")

    def simulate_algorithm(self):
        algorithm = self.selected_algorithm.get()
        for i in range(self.n):
            for j in range(self.n):
                if self.img_null:
                    self.rook_left[i][j].config(image=self.img_null)
                else:
                    self.rook_left[i][j].config(text='')
        self.is_running = True
        self.status_label.config(text=f"Đang mô phỏng thuật toán: {algorithm}")
        
        try:
            if algorithm == "BFS":
                if not self.solutions.get('bfs'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['bfs'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "DFS":
                if not self.solutions.get('dfs'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['dfs'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "UCS":
                if not self.solutions.get('ucs'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['ucs'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "IDS":
                if not self.solutions.get('ids'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['ids'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "A*":
                if not self.solutions.get('a_star'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['a_star'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "Greedy":
                if not self.solutions.get('greedy'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['greedy'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "Hill Climbing":
                if not self.solutions.get('hill_climbing'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['hill_climbing'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "Simulated Annealing":
                if not self.solutions.get('simulated_annealing'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['simulated_annealing'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "Genetic Algorithm":
                if not self.solutions.get('genetic_algorithm'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['genetic_algorithm'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "Beam Search":
                if not self.solutions.get('beam_search'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['beam_search'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "AND-OR Tree Search":
                if not self.solutions.get('and_or'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['and_or'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "Belief State Search":
                if not self.solutions.get('belief_search'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['belief_search'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "Partially Observable Search":
                if not self.solutions.get('partially_observable'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['partially_observable'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "DLS":
                if not self.solutions.get('dls'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['dls'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "Backtracking":
                if not self.solutions.get('backtracking'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['backtracking'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "Forward Checking":
                if not self.solutions.get('forward_checking'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['forward_checking'], self.rook_left, is_delay=True, delay=0.5)
            elif algorithm == "AC-3":
                if not self.solutions.get('ac3'):
                    self.status_label.config(text="Chưa có kết quả để mô phỏng. Vui lòng chạy thuật toán trước.")
                    self.reset_result_boards()
                    return
                self.draw_xe(self.solutions['ac3'], self.rook_left, is_delay=True, delay=0.5)
            else:
                self.status_label.config(text=f"Mô phỏng cho {algorithm} chưa được hỗ trợ. Sử dụng nút 'Chạy'")
            self.status_label.config(text=f"Mô phỏng thuật toán {algorithm} hoàn tất")
        except Exception as e:
            self.status_label.config(text=f"Lỗi: {str(e)}")
            print(f"Lỗi: {e}")

    def stop_simulation(self):
        self.is_running = False
        self.status_label.config(text="Đã dừng mô phỏng")

    def reset_board(self):
        for i in range(self.n):
            for j in range(self.n):
                self.pos_rook[i][j] = 0
                if self.img_null:
                    self.rook_input[i][j].config(image=self.img_null)
                else:
                    self.rook_input[i][j].config(text='')
        
        self.reset_result_boards()
        self.solutions = {}
        self.algorithm_stats = {}
        self.is_running = False
        self.can_edit = True
        self.btn_simulate.config(state='disabled')
        
        self.status_label.config(text="Đã reset tất cả bàn cờ")

    def update_result_display(self, algorithm_name):
        if algorithm_name in self.algorithm_stats:
            stats = self.algorithm_stats[algorithm_name]
            self.result_labels["Nodes Expanded"].config(text=str(stats.get('nodes_expanded', '')))
            self.result_labels["Max Frontier Size"].config(text=str(stats.get('max_frontier', '')))
            self.result_labels["Time (seconds)"].config(text=f"{stats.get('time', 0):.4f}")
            self.result_labels["Heuristic h(x)"].config(text=str(stats.get('heuristic', '')))
        else:
            # Xóa các giá trị nếu không có thống kê
            for key in self.result_labels:
                self.result_labels[key].config(text='')

    def reset_result_boards(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.img_null:
                    self.rook_left[i][j].config(image=self.img_null)
                    self.rook_right[i][j].config(image=self.img_null)
                else:
                    self.rook_left[i][j].config(text='')
                    self.rook_right[i][j].config(text='')

if __name__ == "__main__":
    game = EightRooks()