import numpy as np

class GomokuBoard:
    def __init__(self, size=15, count_win=5):
        self.size = size
        self.count_win = count_win
        self.board = np.zeros((size, size), dtype=np.int8)  # -1, 0, +1
        self.move_count = 0

    def reset(self):
        self.board.fill(0)
        self.move_count = 0

    def legal_moves(self):
        # 返回所有空位 (y, x)
        ys, xs = np.where(self.board == 0)
        return list(zip(ys.tolist(), xs.tolist()))

    def step(self, move, player_flag):
        y, x = move
        if player_flag not in (-1, 1):
            raise ValueError("player_flag must be +1 (Black) or -1 (White).")
        if not (0 <= y < self.size and 0 <= x < self.size):
            raise ValueError("move out of board.")
        if self.board[y, x] != 0:
            raise ValueError("illegal move: cell occupied.")
        self.board[y, x] = player_flag
        self.move_count += 1
        return self.board, player_flag

    def is_terminal(self):
        # 终局：一方胜，或棋满
        return self.winner() != 0 or self.move_count == self.size * self.size

    def play_count(self):
        return self.move_count

    def winner(self):
        """
        返回 1（先手/黑胜），-1（后手/白胜），0（未分胜负或和棋）。
        注意：这里不做禁手判定；需要禁手的话，放到 legal_moves 过滤或单独规则里。
        """
        s = self.size
        b = self.board
        K = self.count_win

        # 四个方向：水平(0,1)，竖直(1,0)，斜下(1,1)，斜上(-1,1)
        dirs = [(0, 1), (1, 0), (1, 1), (-1, 1)]

        for y in range(s):
            for x in range(s):
                p = b[y, x]
                if p == 0:
                    continue
                for dy, dx in dirs:
                    # 起点往反方向退一步避免重复计数（只从“线段最左/最上端”开始数）
                    py, px = y - dy, x - dx
                    if 0 <= py < s and 0 <= px < s and b[py, px] == p:
                        continue

                    # 向 (dy, dx) 数连续同色子
                    cnt = 0
                    ny, nx = y, x
                    while 0 <= ny < s and 0 <= nx < s and b[ny, nx] == p:
                        cnt += 1
                        if cnt >= K:
                            return int(p)  # 胜者
                        ny += dy
                        nx += dx
        # 没有连五即未分胜负（也可能和棋，但和棋由外部 “棋满” 判定）
        return 0
