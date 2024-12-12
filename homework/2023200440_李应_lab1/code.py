import heapq
from heapq import heappush, heappop

import numpy as np
import random
import time

from setuptools.namespaces import flatten
from sympy import Float
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from sympy.codegen.cnodes import static


class PuzzleBoardState(object):
    """ 华容道棋盘类
    """

    def __init__(self, dim=3, random_seed=2024, data=None, parent=None):
        """ 根据给定随机数 随机初始化一个可解的华容道问题
            dim         :   int, 华容道棋盘维度(阶数)
            random_seed :   int, 创建随机棋盘的随机种子
                            可尝试不同的随机数 创建不同的棋盘
            data        :   numpy.ndarray (dim*dim), 创建棋盘的数据
                            可给定一个有解的初始棋盘 (程序未设置对给定数据的可解性检查)
            parent      :   PuzzleBoardState, 设定棋盘状态的父节点状态
                            根节点/初始节点的父节点为空
            default_dst_data : 华容道的目标棋盘状态
                            numpy.array (dim*dim)
            g:          :   初始状态到当前状态的耗散值
            f:          :   初始状态到目标状态的估计值
        """
        self.dim = dim
        self.parent = parent
        self.default_dst_data = np.array([j for j in range(1, self.dim ** 2)] + [0]).reshape(self.dim, self.dim)  #目标状态
        self.data = self._init_data(random_seed, data)  # 随机初始状态
        self.piece_x, self.piece_y = self._get_piece_index()
        self.g = self._get_g()          # g值
        self.f = self.g + self._get_h1()    # f值
        # print(self.piece_x, self.piece_y)

    def _init_data(self, random_seed, data):
        if data is not None:    # data是否有初始值
            if self._if_solvable(data, self.default_dst_data):  # data是否可解
                return data
        # data没有初始值,随机化初始值
        init_cnt = 0
        while init_cnt < 500:
            init_data = self._get_random_data(random_seed=random_seed + init_cnt)
            init_solvable = self._if_solvable(init_data, self.default_dst_data)
            if init_solvable:    # 返回第一个随机化的可解初始状态
                return init_data
            init_cnt += 1
        return None

    def _get_random_data(self, random_seed):
        """ 根据random_seed 生成一个dim*dim的华容道棋盘数据
            random_seed :   int, 随机数
            return      :   numpy.ndarray (dim*dim), 华容道棋盘数据
        """
        random.seed(random_seed)
        init_data = [i for i in range(self.dim ** 2)]
        random.shuffle(init_data)
        init_data = np.array(init_data).reshape((self.dim, self.dim))

        return init_data

    def _get_h1(self):
        """
        计算h(n)的值，即当前状态到目标状态的估计代价，
        h(n)=不在位的数字个数
        """
        h = 0
        flatten_data = self.data.flatten()
        flatten_dst = self.default_dst_data.flatten()
        for j in range(self.dim ** 2):
            if flatten_data[j] == 0:   # 0视作空格
                continue
            if flatten_data[j] != flatten_dst[j]:
                h += 1
        return h

    def _get_h2(self):
        """
                计算h(n)的值，即当前状态到目标状态的估计代价，
                h(n)=不在位的数字到目标位置的距离和
                """
        h = 0
        flatten_data = self.data.flatten()
        flatten_dst = self.default_dst_data.flatten()
        for j in range(self.dim ** 2):
            if flatten_data[j] == 0:  # 0视作空格
                continue
            num = flatten_data[j]
            if num != flatten_dst[j]:
                h += abs(j // self.dim - num // self.dim) + abs(j % self.dim - num % self.dim)
        return h

    def _get_g(self):
        """
        计算初始状态到当前状态的耗散值
        根节点耗散值为0/ 子节点耗散值=父节点耗散值+1
        """
        if self.parent:
            g = self.parent.g + 1
        else:
            g = 0
        return g

    def _get_piece_index(self):
        """ 返回当前将牌(空格)位置
            return :    int, 将牌横坐标 (axis=0)
                        int, 将牌纵坐标 (axis=0)
        """
        index = np.argsort(self.data.flatten())[0]

        return index // self.dim, index % self.dim

    def _inverse_num(self, puzzle_board_data):
        """
        计算总逆序数
        """
        flatten_data = puzzle_board_data.flatten()
        res = 0
        for i in range(len(flatten_data)):
            if flatten_data[i] == 0:
                if self.dim % 2 == 0:
                    res += self.dim - 1 - i // self.dim   # 偶数阶加上0所在行数与目标状态0行数差
                continue
            for j in range(i):
                if flatten_data[j] > flatten_data[i]:
                    res += 1

        return res

    def _if_solvable(self, src_data, dst_data):
        """ 判断一个(src_data => dst_data)的华容道问题是否可解
            src_data : numpy.ndarray (dim*dim), 作判断的棋盘初始状态数据
            src_data : numpy.ndarray (dim*dim), 作判断的棋盘终止状态数据
            return :    boolean, True可解 False不可解
        """
        assert src_data.shape == dst_data.shape, "src_data and dst_data should share same shape."
        inverse_num_sum = self._inverse_num(src_data)
        # 逆序数奇偶性相同有解
        return inverse_num_sum % 2 == 0

    def is_final(self):
        """ 判断棋盘当前状态是否为目标终止状态
            return :    boolean, True终止 False未终止
        """
        flatten_data = self.data.flatten()
        if flatten_data[-1] != 0:
            return False
        for i in range(self.dim ** 2 - 1):
            if flatten_data[i] != (i + 1):
                return False
        return True

    def next_states(self):
        """ 返回当前状态的相邻状态
            return :    list, 当前状态的相邻状态，构成的PuzzleBoardState对象列表
        """
        res = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            x2, y2 = self.piece_x + dx, self.piece_y + dy
            if 0 <= x2 < self.dim and 0 <= y2 < self.dim:   # 没有超出棋盘范围
                new_data = self.data.copy()
                new_data[self.piece_x][self.piece_y] = new_data[x2][y2]
                new_data[x2][y2] = 0
                res.append(PuzzleBoardState(data=new_data, parent=self, dim=self.dim))
                # print(new_data)


        return res

    def get_data(self):
        """ 返回当前棋盘状态数据
            return :    numpy.ndarray (dim*dim), 当前棋盘的数据
        """
        return self.data

    def get_data_hash(self):
        """ 返回基于当前状态数据的哈希值 存储在set中 供判断相同状态使用
            return :    int, 当前状态哈希值
        """
        return hash(tuple(self.data.flatten()))

    def get_parent(self):
        """ 返回当前状态的父节点状态
            return :    PuzzleBoardState, 当前状态的父节点
        """
        return self.parent

    def __lt__(self, other):   # 根据f值比较状态大小
        return self.f < other.f


def bfs(puzzle_board_state):
    """ 已实现的华容道广度优先算法 供参考 """
    visited = set()

    from collections import deque
    queue = deque()
    queue.append((0, puzzle_board_state))
    visited.add(puzzle_board_state.get_data_hash())

    ans = []
    while queue:
        (now, cur_state) = queue.popleft()
        if cur_state.is_final():
            while cur_state.get_parent() is not None:
                ans.append(cur_state)
                cur_state = cur_state.get_parent()
            ans.append(cur_state)
            break

        next_states = cur_state.next_states()
        for next_state in next_states:
            if next_state.get_data_hash() in visited:
                continue
            visited.add(next_state.get_data_hash())
            queue.append((now + 1, next_state))

    return ans


def astar(puzzle_board_state, k=3):
    # TODO
    # A*算法实现
    open_list = []  # open列表存储待访问节点
    heapq.heapify(open_list)   # 初始化最小堆
    closed_set = set()  # closed列表存储已访问过的节点

    # 将初始节点放入open表
    heapq.heappush(open_list, puzzle_board_state)

    ans = []
    count = 0
    while open_list:  # 防止递归深度过大，栈空间已满
        cur_state = heapq.heappop(open_list)  # 弹出f值最小的节点，计算哈希值并放入clos表
        closed_set.add(cur_state.get_data_hash())

        # 判断是否结束
        if cur_state.is_final():
            while cur_state:   # 开始回溯父节点
                ans.append(cur_state)
                cur_state = cur_state.get_parent()
            return ans[::-1]
        # 获取相邻状态
        next_states = cur_state.next_states()

        for next_state in next_states:
            if next_state.get_data_hash() in closed_set:
                continue
            heappush(open_list, next_state)

    return None


def plot_puzzle(state, diff=None, dim=3):
    plt.cla()  # 清除旧的图形

    plt.imshow(state, cmap='gray_r', interpolation='nearest')
    for i in range(dim):
        for j in range(dim):
            # 添加木块纹理和阴影效果
            color = 'saddlebrown' if state[i, j] != 0 else 'white'
            edgecolor = 'black' if state[i, j] != 0 else 'white'
            if diff and i == diff[0] and j == diff[1]:
                color = 'tan'
            if state[i, j] != 0:
                rect = Rectangle((j - 0.5, 2 - i - 0.5), 1, 1, facecolor=color, edgecolor=edgecolor, lw=2)
                plt.gca().add_patch(rect)
            else:
                rect = Rectangle((j - 0.5 + 0.05, 2 - i - 0.5 + 0.05), 0.9, 0.9, facecolor='white', edgecolor=edgecolor,
                                 lw=2)
                plt.gca().add_patch(rect)

            # 添加立体效果：内阴影
            rect_shadow = Rectangle((j - 0.5 + 0.025, 2 - i - 0.5 + 0.025), 0.95, 0.95, facecolor=edgecolor, alpha=0.5)
            plt.gca().add_patch(rect_shadow)

            # 在单元格上绘制数字
            plt.text(j, 2 - i, str(state[i, j]), ha='center', va='center', color='black', fontsize=18)

        plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
        plt.grid(False)  # 不显示网格线


# 定义动画更新函数
def update(frame, path, dim=3):
    if frame < len(path) - 1:  # 只更新前两次动画
        # 获取当前状态和下一个状态
        current_state = path[frame]
        next_state = path[frame + 1]

        # 找到移动的位置
        diff = np.where(np.logical_and(current_state != next_state, current_state != 0))

        # 绘制当前状态
        plot_puzzle(current_state, diff, dim)
    else:
        plot_puzzle(cpath[frame], None, dim)


def visilize_path(path, dim):
    # 创建动画
    fig, ax = plt.subplots()
    # 使用lambda函数捕获states变量
    ani = FuncAnimation(fig, lambda frame: update(frame, path, dim), frames=len(path), interval=1500, repeat=False)
    plt.show()


if __name__ == "__main__":
    # t1 = time.perf_counter()
    test_data = np.array([[5, 3, 6], [1, 8, 4], [7, 2, 0]])
    test_board = PuzzleBoardState(random_seed=11, dim=3)
    res = astar(test_board)
    for i in res:
        print(i.data)
    # 可视化
    # if res is not None:
    #     visilize_path([r.data for r in res], test_board.dim)

    # print(time.perf_counter() - t1)
