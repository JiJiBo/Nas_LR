import torch


class MCTS():
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def run(self):
        pass

    def select_child(self, node):
        pass

    def expand_node(self, node):
        pass
