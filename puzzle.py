import heapq
from typing import List, Optional, Tuple, Dict

# Board is stored as tuple so each state can be hashed easily
# 0 means blank tile
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
GOAL_POS = {value: (idx // 3, idx % 3) for idx, value in enumerate(GOAL)}


class PuzzleNode:
    def __init__(self, state: Tuple[int, ...], parent: Optional["PuzzleNode"] = None,
                 move: Optional[str] = None, g_cost: int = 0) -> None:
        self.state = state
        self.parent = parent
        self.move = move
        self.g = g_cost
        self.h = grid_distance(state)
        self.f = self.g + self.h

    def __lt__(self, other: "PuzzleNode") -> bool:
        # In priority queue, node with smaller f is expanded first
        # If f is same, use smaller h as tie-break
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f


def grid_distance(state: Tuple[int, ...]) -> int:
    """h(n): sum of tile distances from their goal positions in 3x3 grid."""
    # This heuristic is admissible, so A* can still find optimal path
    dist = 0
    for i, tile in enumerate(state):
        if tile == 0:
            continue
        r1, c1 = i // 3, i % 3
        r2, c2 = GOAL_POS[tile]
        dist += abs(r1 - r2) + abs(c1 - c2)
    return dist


def is_solvable(state: Tuple[int, ...]) -> bool:
    """8-puzzle solvability check using inversion count."""
    # Quick check to avoid running search on impossible puzzle
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv % 2 == 0


def get_successors(state: Tuple[int, ...]) -> List[Tuple[Tuple[int, ...], str]]:
    """Generate all valid successor states by moving blank in legal directions."""
    # Generate all legal next states from current state
    successors = []
    blank_idx = state.index(0)
    row, col = blank_idx // 3, blank_idx % 3

    # 4 possible moves of blank tile
    directions = [
        (-1, 0, "UP"),
        (1, 0, "DOWN"),
        (0, -1, "LEFT"),
        (0, 1, "RIGHT"),
    ]

    for dr, dc, move_name in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            # If move is inside grid, create next state by swap
            new_idx = nr * 3 + nc
            board = list(state)
            board[blank_idx], board[new_idx] = board[new_idx], board[blank_idx]
            successors.append((tuple(board), move_name))

    return successors


def reconstruct_path(goal_node: PuzzleNode) -> List[PuzzleNode]:
    # Follow parent pointers to rebuild full path from start to goal
    path = []
    current = goal_node
    while current is not None:
        path.append(current)
        current = current.parent
    path.reverse()
    return path


def print_board(state: Tuple[int, ...]) -> None:
    for i in range(0, 9, 3):
        row = state[i:i + 3]
        print(" ".join("_" if x == 0 else str(x) for x in row))


def solve_8_puzzle_a_star(initial_state: Tuple[int, ...]) -> Optional[PuzzleNode]:
    print("\nA* Goal-Based Agent for 8-Puzzle")
    print("Initial state:")
    print_board(initial_state)

    if initial_state == GOAL:
        print("\nAlready at goal state. Cost = 0")
        return PuzzleNode(initial_state)

    if not is_solvable(initial_state):
        print("\nThis puzzle is not solvable.")
        return None

    # Open list is maintained as a min-heap
    open_heap: List[PuzzleNode] = []
    start = PuzzleNode(initial_state)
    heapq.heappush(open_heap, start)

    # Store best known g(n) for each state to avoid worse duplicate paths
    best_g: Dict[Tuple[int, ...], int] = {initial_state: 0}
    expanded = 0

    while open_heap:
        current = heapq.heappop(open_heap)

        # Ignore outdated node if a better route is already known
        if current.g > best_g.get(current.state, float("inf")):
            continue

        if current.state == GOAL:
            print("\nGoal found!")
            print(f"Nodes expanded: {expanded}")
            return current

        expanded += 1

        for next_state, move in get_successors(current.state):
            new_g = current.g + 1
            if new_g < best_g.get(next_state, float("inf")):
                best_g[next_state] = new_g
                child = PuzzleNode(next_state, parent=current, move=move, g_cost=new_g)
                heapq.heappush(open_heap, child)

    print("No solution found.")
    return None


def show_full_solution(goal_node: PuzzleNode) -> None:
    # Print complete path with g, h, f values at each step
    path = reconstruct_path(goal_node)
    total_moves = len(path) - 1

    print("\nFull solution path:")
    for step_index, node in enumerate(path):
        if step_index == 0:
            print(f"\nStep {step_index} (Initial)")
        else:
            print(f"\nStep {step_index} (Move: {node.move})")
        print(f"g(n) = {node.g}, h(n) = {node.h}, f(n) = {node.f}")
        print_board(node.state)

    print("\nMove sequence:")
    moves = [node.move for node in path if node.move is not None]
    print(" -> ".join(moves))

    print(f"\nTotal moves (cost): {total_moves}")


def print_analysis_note() -> None:
    print("\nHow the agent makes optimal decisions:")
    print("1. It generates all valid successor states from the blank tile moves.")
    print("2. It computes h(n) with grid distance for each state.")
    print("3. It computes f(n) = g(n) + h(n) and expands the smallest f(n) first.")
    print("4. It tracks best g(n) for each state so worse paths are ignored.")
    print("5. With admissible heuristic, first goal reached is optimal.")


if __name__ == "__main__":
    city_test_cases = {
        "Islamabad": (1, 2, 3,
                       4, 0, 5,
                       7, 8, 6),
        "Lahore": (1, 2, 3,
                    5, 0, 6,
                    4, 7, 8),
        "Karachi": (8, 1, 3,
                     4, 0, 2,
                     7, 6, 5),
    }

    print("8-PUZZLE A* TEST RUNS (Pakistan Major Cities)")
    print("=" * 55)

    for city_name, initial in city_test_cases.items():
        print(f"\nTest Case: {city_name}")
        print("-" * 55)
        result = solve_8_puzzle_a_star(initial)
        if result is not None:
            show_full_solution(result)
        else:
            print("No path available for this test case.")

        print_analysis_note()
        print("=" * 55)
