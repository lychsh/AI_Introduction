#coding:utf-8
'''
    Lab 2
    井字棋(Tic tac toe)Python语言实现, 带有Alpha-Beta剪枝的Minimax算法.
'''
import random

from numpy.core.numeric import infty
from sympy import false

# 棋盘位置表示（0-8）:
# 0  1  2
# 3  4  5
# 6  7  8

# 设定获胜的组合方式(横、竖、斜)
WINNING_TRIADS = ((0, 1, 2), (3, 4, 5), (6, 7, 8),
                  (0, 3, 6), (1, 4, 7),(2, 5, 8),
                  (0, 4, 8), (2, 4, 6))
# 设定棋盘按一行三个打印
PRINTING_TRIADS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
# 用一维列表表示棋盘:
SLOTS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
# -1表示X玩家 0表示空位 1表示O玩家.
X_token = -1
Open_token = 0
O_token = 1
# MAX代表先手,MIN代表后手
MAX = True
MIN = False

MARKERS = ['_', 'O', 'X']
END_PHRASE = ('平局', '胜利', '失败')


def print_board(board):
    """打印当前棋盘"""
    for row in PRINTING_TRIADS:
        r = ' '
        for hole in row:
            r += MARKERS[board[hole]] + ' '
        print(r)


def legal_move_left(board):
    """ 判断棋盘上是否还有空位 """
    for slot in SLOTS:
        if board[slot] == Open_token:
            return True
    return False


def winner(board):
    """ 判断局面的胜者,返回值-1表示X获胜,1表示O获胜,0表示平局或者未结束"""
    for triad in WINNING_TRIADS:
        triad_sum = board[triad[0]] + board[triad[1]] + board[triad[2]]
        if triad_sum == 3 or triad_sum == -3:
            return board[triad[0]]  # 表示棋子的数值恰好也是-1:X,1:O
    return 0


def alpha_beta(board, alpha, beta, player, depth):
    """
        alphabeta剪枝算法
    """
    score = winner(board)
    if score != 0 or not legal_move_left(board):  # 胜负知晓或者平局返回
        return score
    # 轮到先手,找赢面最大的
    if player:
        for slot in SLOTS:
            if board[slot] == Open_token:
                board[slot] = O_token
                value = alpha_beta(board, alpha, beta, MIN, depth + 1)
                board[slot] = Open_token
                alpha = max(alpha, value)
                if beta <= alpha:
                    break     # α剪枝
        return alpha
    # 轮到后手，找使先手赢面最小的
    else:
        for slot in SLOTS:
            if board[slot] == Open_token:
                board[slot] = X_token
                value = alpha_beta(board, alpha, beta, MAX, depth + 1)
                board[slot] = Open_token
                beta = min(beta, value)
                if beta <= alpha:
                    break  # β剪枝
        return beta


def determine_move(board):
    """
        决定电脑(玩家O)的下一步棋(使用Alpha-beta 剪枝优化搜索效率)
        Args:
            board (list):井字棋盘

        Returns:
            next_move(int): 电脑(玩家O) 下一步棋的位置
    """
    # α，β初值分别设为负无穷和正无穷
    alpha = float('-inf')
    beta = float('inf')
    next_move = None
    maxvalue = float('-inf')
    for slot in SLOTS:
        if board[slot] == Open_token:   # 找到空白格
            board[slot] = O_token      # 落子
            value = alpha_beta(board, alpha, beta, MIN, 0)
            board[slot] = Open_token   # 撤销落子
            alpha = max(alpha, value)
            if maxvalue < value:   # 找到赢面更大的slot，更新maxvalue和nextmove
                maxvalue = value
                next_move = slot
    return next_move


HUMAN = 1
COMPUTER = 0


def main():
    """主函数,先决定谁是X(先手方),再开始下棋"""
    next_move = HUMAN
    opt = input("请选择先手方，输入X表示玩家先手，输入O表示电脑先手：")
    if opt == "X" or opt == "x":
        next_move = HUMAN
    elif opt == "O" or opt == "o":
        next_move = COMPUTER
    else:
        print("输入有误，默认玩家先手")

    # 初始化空棋盘
    board = [Open_token for i in range(9)]
    print_board(board)
    # 开始下棋
    while legal_move_left(board) and winner(board) == Open_token:
        if next_move == HUMAN and legal_move_left(board):
            try:
                print("\n")
                humanmv = int(input("请输入你要落子的位置(0-8)："))
                if board[humanmv] != Open_token:
                    continue
                board[humanmv] = X_token
                next_move = COMPUTER
                print_board(board)
            except:
                print("输入有误，请重试")
                continue
        if next_move == COMPUTER and legal_move_left(board):
            mymv = determine_move(board)
            if mymv is None:
                break
            print("Computer最终决定下在", mymv)
            board[mymv] = O_token
            next_move = HUMAN
            print_board(board)

    # 输出结果
    # print_board(board)
    print(["平局", "Computer赢了", "你赢了"][winner(board)])


if __name__ == '__main__':
    main()
