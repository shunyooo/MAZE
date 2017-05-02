from  MDP import MDP
from utils import argmax, vector_add, print_table  # noqa
from grid import orientations, turn_right, turn_left
from pprint import pprint

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


    # エピソードの終わりの判定
    # 次状態を引数として枠外なら0を
    def state_check(self,state):
           """エピソードの終わりの判定
           次状態を引数として枠外なら0を返す。
           """
           y,x = state
           if y < 0 or x < 0 or self.rows-1 < y or self.cols-1 < x :
                return False
           if self.reward[y,x] < 0:#報酬が負だったら壁
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
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]

    # 指定した方向に移動した場合の状態を返す。
    # 移動後の状態がMDPになければ、元の状態を返す(移動しない)。
    def go(self, state, direction):
        state1 = vector_add(state, direction)
        #print("state:{0}, direction:{1} -> state'{2}".format(state,direction,state1))
        return state1 if state1 in self.states else state
# ________________________________________________________________________________


def value_iteration(mdp, epsilon=0.001):
    """
    入力としてGridMDPとepsilon:収束判定に用いる小さな値を受け取る。
    出力として各状態における価値U(s)を返す。
    
    価値反復法
    状態sから最適な行動をとり続けた時の期待利得を計算する。
    得られる利得の期待値を考えるのは確率的な遷移を行われるため
    1.全ての状態sについてU(s)を適当な値(ゼロなど)に初期化
    2.全てのU(s)について以下の式を計算して値を更新
      U(s) <- R(s) + r max sig[s'](P(s'|s,a)U(s'))
    3.ステップ2が収束するまで繰り返す。
    """

    # 全ての状態sについてU(s)を0に初期化
    U1 = {s: 0 for s in mdp.states}

    # R(s):報酬関数,T(s,a):次状態への遷移確率と次状態のタプルを返す関数,gamma:割引係数
    R, T, gamma = mdp.R, mdp.T, mdp.gamma

    while True:
        U = U1.copy()
        delta = 0
        print("-"*40)
        print(mdp.states)
        printGrid(mdp.states,mdp.rows,mdp.cols)
        #全ての状態sに関して
        for s in mdp.states:
            # 価値更新。

            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])# (p:次状態への遷移確率,s1:次状態)
                                        for a in mdp.actions(s)])# 状態sにおける、可能なaction全てに関して

            print("s{0}".format(s))
            printGrid(mdp.states,mdp.rows,mdp.cols,mark = s)
            #この状態における全てのactionについて、
            for a in mdp.actions(s):
                print("----- a:{0}".format(toArrow(a)))
                #遷移確率と、その状態
                for p,s1 in T(s,a):
                    print("---------- R[{0}]:{5} + {6} * U{2} p{3} U[s1]{4}".format(s,a,s1,p,U[s1],R(s),gamma))



            print(" U[{0}] = {1}".format(s,U[s]))
            print("U1[{0}] = {1}".format(s,U1[s]))


            delta = max(delta, abs(U1[s] - U[s]))# 変化値の最大値を取っておく
            print("delta = {0}".format(delta))
        # もし、変化値の最大値が閾値を超えなかったら->収束したら、この価値をreturn
        if delta < epsilon * (1 - gamma) / gamma:
            return U

def best_policy(mdp, U):
    """与えられたMDPと価値関数から最適な方策を決定する。
    具体的には、全ての状態に関して{状態s:するべき行動a}の辞書を追加して返す。"""

    pi = {}
    for s in mdp.states:
        # keyで比較を行なっている。つまり、sにおいて全ての(s,a)の価値を計算して比較。
        # 最大のaを、{s:a}として登録。
        pi[s] = argmax(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
    return pi

# 与えられたMDPと価値関数Uから、s,aでの次状態s1での価値の期待値p*U(s1)の和
# つまり、(s,a)における価値を表すっぽい。
def expected_utility(a, s, U, mdp):
    """The expected utility of doing a in state s, according to the MDP and U."""
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])

# ______________________________________________________________________________


def printGrid(tuples,rows,cols,mark=None):
    """迷路の出力"""
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





