from  MDP import MDP
from utils import argmax, vector_add, print_table  # noqa
from grid import orientations, turn_right, turn_left
from pprint import pprint
import numpy as np, pandas as pd

import random

class GridMDP(MDP):

    # grid引数を受け取る点が差分。
    # grid: 各状態での報酬が格納されている。Mapデータみたいなもの。
    # GridMDP([[-0.04, -0.04, -0.04, +1],
    #          [-0.04, None,  -0.04, -1],
    #          [-0.04, -0.04, -0.04, -0.04]],
    #           terminals=[(3, 2), (3, 1)])
    # のように記述。Noneは壁。
    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):                                                                              
        MDP.__init__(self, init, actlist=orientations,
                     terminals=terminals, gamma=gamma)

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        # print(self.rows,self.cols)

        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[y, x] = grid[y][x]
                if self.state_check((y,x)):
                    self.states.add((y, x))

    
    def state_check(self,state):
           """ その状態が存在するかどうかを判定。"""
           y,x = state
           if y < 0 or x < 0 or self.rows-1 < y or self.cols-1 < x :
                return False
           if self.grid[y][x] is None:# Noneだったら壁。
                return False
           return True

    # 状態stateで行動actionを取った時に、
    # 次状態への遷移確率と次状態のタプル(probability,s')
    # のリストを返す。 
    def T(self, state, action):
        # print("state:{0},a:{1},ra:{2},la:{3}".format(state,action,turn_right(action),turn_left(action)))
        if action is None:
            #アクションが取られなかった時、そのまま。
            return [(0.0, state)]
        else:
            # アクションが取られた時、
            # 行きたい方向に0.8、その左右に0.1の確率で遷移する。
            list1 = []
            acts = [action,turn_right(action),turn_left(action)]
            pros = [0.8,0.1,0.1]
            for (a,p) in zip(acts,pros):
                if self.state_check([x+y for (x,y) in zip(state,a)]):
                    list1.append((p, self.go(state, a)))
            return list1

            # return [(0.8, self.go(state, action)),
            #         (0.1, self.go(state, turn_right(action))),
            #         (0.1, self.go(state, turn_left(action)))]

    # 指定した方向に移動した場合の状態を返す。
    # 移動後の状態がMDPになければ、元の状態を返す(移動しない)。
    def go(self, state, direction):
        state1 = vector_add(state, direction)
        #print("state:{0}, direction:{1} -> state'{2}".format(state,direction,state1))
        return state1 if state1 in self.states else state

    # 状態における報酬メソッドを変更。
    # 状態と行動を引数に取ることにし、移動しない時の報酬を0にする。
    # いままでは、その状態の価値を返していた。
    def R(self, state, action):
        n_state = self.go(state,action)
        if n_state == state:
            return 0.0
        return self.reward[n_state]


    # 状態stateでとれる行動のリストを返す。
    # 問題用にオーバーライド。壁に向かう行動はできないようにする。
    def actions(self, state):
        if state not in self.states:
            print("状態値が不正です。:{0}\n{1}にあるべきです".format(state,self.states))
            raise

        if state in self.terminals:
            return [None]
        else:# 壁への行動を抜く
            return [a for a in self.actlist if self.state_check([x+y for (x,y) in zip(state,a)])]
# ________________________________________________________________________________

# ______________________________________________________________________________

def printGrid(grid,mark=None):
    """gridを受け取って迷路の出力"""
    wall = "■"
    road = "□"
    star = "☆"
    for row in grid:
        for i in row:
            if i == None:
                print(wall,end="")
            # elif i == mark:
            #     print(star,end="")
            else:
                print(road,end="")
        print()


def printGridByStates(tuples,rows,cols,mark=None):
    """状態を受け取って、迷路の出力"""
    wall = "■"
    road = "□"
    star_road = "☆"
    star_wall = "★"
    for y in range(rows):
        for x in range(cols):
            if (y,x) in tuples:
                if (y,x) == mark:
                    print(star_road,end="")
                    continue
                print(road,end="")
            else:
                if (y,x) == mark:
                    print(wall_road,end="")
                    continue
                print(wall,end="")
        print()

def toArrow(action):
    if action == (1,0):
        return "↓"
    elif action == (0,1):
        return "→"
    elif action == (-1,0):
        return "↑"
    elif action == (0,-1):
        return "←"
    elif action == None:
        return "☆"

def printPi(pi,rows = 13,cols = 10):
    wall = "■"
    road = "□"

    # print("   ",end = "")
    # for x in range(cols):
    #     print("{0:2d}".format(x),end = "")
    # print()

    for y in range(rows):
        # print("{0:2d}".format(y),end = "")
        for x in range(cols):
            if (y,x) in pi.keys():
                print(toArrow(pi[(y,x)]),end="")
            else:
                print(wall,end="")
        print()





