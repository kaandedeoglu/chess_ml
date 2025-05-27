import chess
from chessml.move_encoding import move_to_index
from chessml.board_encoding import encode_board
import math
import torch

class MCTSNode:
    def __init__(self, board: chess.Board, parent=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = 0.0 # from policy head
        self.mean_value = 0.0
    
    def is_expanded(self):
        return len(self.children) > 0
    
    def expand(self, policy_probs):
        """
        Add children nodes for all legal moves, with priors from the policy head.
        """

        for move in self.board.legal_moves:
            next_board = self.board.copy()
            next_board.push(move)

            idx = move_to_index(move)
            prob = policy_probs[idx].item()

            child_node = MCTSNode(next_board, parent=self)
            child_node.prior = prob
            self.children[move] = child_node

    def select_child(self, c_puct=1.0):
        if not self.children:
            return None, None
        
        best_score = -float("inf")
        best_move = None
        best_child = None

        for move, child in self.children.items():
            Q = child.mean_value
            P = child.prior
            N = self.visit_count
            n = child.visit_count

            ucb_score = Q + c_puct * P * math.sqrt(N) / (1 + n)

            if ucb_score > best_score:
                best_score = ucb_score
                best_move = move
                best_child = child

        return best_move, best_child

    def update(self, value):
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count

def run_mcts(model, root_board, num_simulations=100, c_puct=1.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    root = MCTSNode(root_board)

    x = torch.tensor(encode_board(root.board), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    import time
    start_time = time.time()

    with torch.no_grad():
        policy_logits, value = model(x)
        policy_probs = torch.softmax(policy_logits, dim=1)[0]
    root.expand(policy_probs)
    root.update(value.item())

    for sim in range(num_simulations):
        sim_start = time.time()

        node = root
        search_path = [node]

        while node is not None and node.is_expanded() and not node.board.is_game_over():
            _, next_node = node.select_child(c_puct)
            if next_node is None:
                break
            node = next_node
            search_path.append(node)
        
        x = torch.tensor(encode_board(root.board), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_logits, value = model(x)
            policy_probs = torch.softmax(policy_logits, dim=1)[0]
        if node is not None and not node.board.is_game_over():
            node.expand(policy_probs)

        leaf_value = value.item()
        for n in reversed(search_path):
            n.update(leaf_value if n.board.turn == root.board.turn else -leaf_value)

        if sim == 0 or sim == num_simulations - 1:
            print(f"    🔁 MCTS sim {sim+1}/{num_simulations} took {time.time() - sim_start:.2f}s")


    print(f"    ⏳ MCTS total time: {time.time() - start_time:.2f}s")
    move_counts = {move: child.visit_count for move, child in root.children.items()}
    return move_counts