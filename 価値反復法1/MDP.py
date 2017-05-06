# マルコフ決定過程
# これを継承してGridMDPクラスを作る。
# 参考：http://qiita.com/Hironsan/items/56f6c0b2f4cfd28dd906
class MDP:

    # init: 初期状態
    # actlist: 各状態でとれる行動
    # terminals: 終了状態(終点)のリスト
    # gamma: 割引係数
    # states: 状態の取りうる値
    # reward: 各状態における報酬を持つ
    def __init__(self, init, actlist, terminals, gamma=.9):
        self.init = init
        self.actlist = actlist
        self.terminals = terminals
        if not (0 <= gamma < 1):
            raise ValueError("An MDP must have 0 <= gamma < 1")
        self.gamma = gamma
        self.states = set()
        self.reward = {}

    # 各状態での報酬を返す
    def R(self, state):
        return self.reward[state]

    # stateにおいてactionした時の報酬を返す。
    def R1(self,state,action):
        s1 = state + action
        return self.reward[s1]


    # 遷移モデル。ここでは抽象メソッドだが、
    # 状態stateで行動actionを取った時に、
    # 次状態への遷移確率と次状態のタプル(probability,s')
    # のリストを返す。P(s',a,s)
    # ここはGridMDPで実装していく。
    def T(self, state, action):
        raise NotImplementedError

    # 状態stateでとれる行動のリストを返す。
    def actions(self, state):
        if state not in self.states:
            print("状態値が不正です。:{0}\n{1}にあるべきです".format(state,self.states))
            raise

        if state in self.terminals:
            return [None]
        else:
            return self.actlist