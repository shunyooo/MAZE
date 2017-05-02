# coding:utf-8
import numpy as np
import random
from GridMDP import *

# -1が壁、0が道、1がゴール
MAZE = [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
        [-1,-1,-1,-1,-1, 0,-1,-1, 0,-1],
        [-1,-1, 0, 0, 0, 0, 0,-1, 0,-1],
        [-1, 0,-1,-1,-1,-1,-1,-1, 0,-1],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
        [-1, 0,-1,-1,-1,-1,-1,-1, 0,-1],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
        [-1,-1, 0,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1, 0, 0, 0, 0, 0, 0, 0,-1],
        [-1,-1,-1,-1, 0,-1,-1,-1,-1,-1],
        [-1,-1, 0, 0, 0, 0, 0, 0, 1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]

START  = (1, 1)
ACTION = [(-1, 0), (1, 0), (0, -1), (0, 1)] # [上, 下, 左, 右]
EPOCH  = 1000 #何世代やるか
ALPH   = 0.1 #
EPSIL  = 0.1 #
RESULT = []

mdp = GridMDP(MAZE, terminals = (len(MAZE)-2, len(MAZE[0])-2), init=START, gamma=ALPH)



# 二つのテキストファイルを出力。
# maze.txt 迷路情報の出力

# action.txt 行動履歴を出力
# (Action y) (Action x) (ゲーム状況)
# を一行ごとに。
# ゲーム状況:スタート位置3 ゴール2 それ以外0
# 0,0,3が初期値となる


# スタートは(1,1)
# ゴールは(y, x) = (len(MAZE)-2, len(MAZE[0])-2)
# に固定
