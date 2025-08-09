import torch

from AI五子棋.mcts.MCTS_Node import MCTSNode
from AI五子棋.net.GomokuNet import PolicyValueNet


class MCTS():
    def __init__(self, model: PolicyValueNet):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def run(self):
        pass

    def select_child(self, node: MCTSNode):
        pass

    def expand_node(self, node: MCTSNode):
        pass
