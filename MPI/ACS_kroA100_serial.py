import csv
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
import numpy as np
import seaborn

class node:
    def __init__(self, pid, x, y):
        self.pid = pid
        self.x = x
        self.y = y

fr = open('kroA100.txt', 'r')
nodeList = []
nodeDict={}
xList = []
yList = []

for row in fr:
    entry = row.split()
    pid = entry[0]
    tmp_x = int(entry[1])
    tmp_y = int(entry[2])
    newNode = node(pid, tmp_x, tmp_y)
    xList.append(tmp_x)
    yList.append(tmp_y)
    nodeList.append(newNode)
    nodeDict[pid] = newNode
fr.close()
n = len(nodeDict)# 100，节点个数
# plt.plot(xList,yList,'*')
# plt.show()

# 距离矩阵，constant
distanceMat = {}
for na in nodeDict.keys():
    distanceMat[na] = {}
    xa = nodeDict[na].x
    ya = nodeDict[na].y
    for nb in nodeDict.keys():
        xb = nodeDict[nb].x
        yb = nodeDict[nb].y
        distanceMat[na][nb] = round(math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2))
print('distanceMat:',distanceMat)

# 长度为nn的最近邻列表矩阵
nn_list = {}
nn = 40
for na in nodeDict.keys():
    # nn_list[na] = []
    tmpDict = dict(sorted(distanceMat[na].items(),key=lambda x:x[1]))
    nn_list[na] = list(tmpDict.keys())[1:nn+1]# 去掉na本身
print('nn_list:', nn_list)

# 信息素矩阵
pheromone = {}
# 合并信息素和启发式信息的矩阵
choice_info = {}
# 启发式信息矩阵，constant
heuristic = {}

# start用最近邻方法求Cnn
tmp_visited = {}
for na in nodeDict.keys():
    tmp_visited[na] = 0
curNode = '1'
Cnn = 0
tmp_path = []
flag = False
while not flag:
    # print(curNode)
    tmp_visited[curNode] = 1
    tmp_path.append(curNode)
    flag = True
    for na in tmp_visited.keys():
        if tmp_visited[na] == 0:
            flag = False
            break
    if flag:
        break
    for i in range(nn):
        tmpNode = nn_list[curNode][i]
        if tmp_visited[tmpNode] == 0:
            Cnn += distanceMat[curNode][tmpNode]
            curNode = tmpNode
            break
        if i == nn-1:
            cc = math.inf
            nextNode = 0
            for nextN in nodeDict.keys():
                if tmp_visited[nextN] == 0:
                    if distanceMat[curNode][nextN] < cc:
                        cc = distanceMat[curNode][nextN]
                        nextNode = nextN
            curNode = nextNode
            Cnn += distanceMat[curNode][nextNode]

tmp_path.append('1')
tmpNode = '1'
Cnn += distanceMat[curNode][tmpNode]
print('Cnn:',Cnn)# 27772
print('Cnn_path:',tmp_path)
print('Cnn_path_length:',len(tmp_path))

tmp_path_x = []
tmp_path_y = []
for na in tmp_path:
    tmp_path_x.append(nodeDict[na].x)
    tmp_path_y.append(nodeDict[na].y)
# plt.plot(tmp_path_x,tmp_path_y,'*-')
# plt.show()
# end用最近邻方法求Cnn

q0 = 0.9# 按照choice_info的最优来选择的概率
# 定义蚂蚁结构体
class single_ant:
    def __init__(self,k):
        self.k = k
        self.tour_length = 0# 蚂蚁的路径长度
        self.tour = []# 蚂蚁保存（部分）路径的记忆存储
        self.visited = {}# 已经访问过的城市
        # 全部初始化为未访问
        for na in nodeDict.keys():
            self.visited[na] = 0

    def reInitialization(self):
        self.tour_length = 0  # 蚂蚁的路径长度
        self.tour = []  # 蚂蚁保存（部分）路径的记忆存储
        self.visited = {}  # 已经访问过的城市
        # 全部初始化为未访问
        for na in nodeDict.keys():
            self.visited[na] = 0

    def computeTourLength(self):
        for ii in range(len(self.tour)-1):
            self.tour_length += distanceMat[self.tour[ii]][self.tour[ii+1]]

    def ACSDecisionRule(self, i):
        curN = self.tour[i-1]
        selection_probability = {}
        sum_probabilities = 0.0
        nextN = ''
        # 将已被访问过的节点的选择可能性设置为0
        for na in nodeDict.keys():
            if self.visited[na] == 1:
                selection_probability[na] = 0.0
            else:
                selection_probability[na] = choice_info[curN][na]
                sum_probabilities += selection_probability[na]
        r1 = random.random()
        # 以q0的概率，直接选择最优的下一节点（初次迭代的时候，如果q0=1，则算法退化成了最近邻；如果q0=0，则算法退化成普通的AS）
        if r1 <= q0:
            sorted_selection_probability = dict(sorted(selection_probability.items(),key=lambda x: x[1]))
            nextN = list(sorted_selection_probability.keys())[-1]
        # 以1-q0的概率，进行轮盘赌选择
        else:
            r2 = random.random()*sum_probabilities
            p = 0.0
            for na in selection_probability.keys():
                p += selection_probability[na]
                if p > r2:
                    nextN = na
                    break
                # elif na == list(selection_probability.keys())[-1]:
                #     print(curN,sum_probabilities,selection_probability)
        self.tour.append(nextN)
        LocalPheromoneUpdate(curN,nextN)
        self.visited[nextN] = 1

    # 有候选城市列表的选下一城市的规则
    def NeighborListACSDecisionRule(self,i):
        curN = self.tour[i - 1]
        selection_probability = {}
        sum_probabilities = 0.0
        nextN = ''
        # 将已被访问过的节点的选择可能性设置为0
        for na in nn_list[curN]:
            if self.visited[na] == 1:
                selection_probability[na] = 0.0
            else:
                selection_probability[na] = choice_info[curN][na]
                sum_probabilities += selection_probability[na]
        if sum_probabilities == 0.0:
            self.ChooseBestNext(i)
        else:
            r1 = random.random()
            # 以q0的概率，直接选择最优的下一节点（初次迭代的时候，如果q0=1，则算法退化成了最近邻；如果q0=0，则算法退化成普通的AS）
            if r1 <= q0:
                sorted_selection_probability = dict(sorted(selection_probability.items(), key=lambda x: x[1]))
                nextN = list(sorted_selection_probability.keys())[-1]
            else:
                r2 = random.random() * sum_probabilities
                p = 0.0
                for na in selection_probability.keys():
                    p += selection_probability[na]
                    if p > r2:
                        nextN = na
                        break
                    # elif na == list(selection_probability.keys())[-1]:
                    #     print(curN,sum_probabilities,selection_probability)
            self.tour.append(nextN)
            LocalPheromoneUpdate(curN, nextN)
            self.visited[nextN] = 1

    def ChooseBestNext(self,i):
        curN = self.tour[i - 1]
        v=0.0
        for na in nodeDict.keys():
            if self.visited[na] == 0:
                if choice_info[curN][na]>v:
                    nextN = na
                    v = choice_info[curN][na]
        self.tour.append(nextN)
        LocalPheromoneUpdate(curN, nextN)
        self.visited[nextN] = 1

    def DepositPheromone(self,best_k,hbt,hbtl):
        # # start AS的全局信息素更新
        # delta_tao = 1/self.tour_length
        # for ii in range(len(self.tour)-1):
        #     curN = self.tour[ii]
        #     nextN = self.tour[ii+1]
        #     pheromone[curN][nextN] += delta_tao
        #     pheromone[nextN][curN] = pheromone[curN][nextN]
        # # end AS的全局信息素更新
        rou = 0.1
        # best_tour = antDict[best_k].tour
        # best_tour_length = antDict[best_k].tour_length
        best_tour = hbt
        best_tour_length = hbtl
        delta_tao_best_tour = 1 / best_tour_length
        for i1 in range(len(self.tour) - 1):
            curN = self.tour[i1]
            nextN = self.tour[i1 + 1]
            for i2 in range(len(best_tour)):
                if best_tour[i2] == curN:
                    curInd = i2
                    break
            for i3 in range(len(best_tour)):
                if best_tour[i3] == nextN:
                    nextInd = i3
                    break
            if abs(curInd - nextInd) == 1 or abs(curInd - nextInd) == n-1:
                pheromone[curN][nextN] += rou * delta_tao_best_tour
                pheromone[nextN][curN] = pheromone[curN][nextN]

history_maxminpher = []
history_maxminheu = []
history_tour_length = []
m = 10# 蚂蚁个数
antDict = {}# 字典长度为m，蚂蚁个数

def InitializeData():
    # 蚂蚁群初始化
    for k in range(1,m+1):
        tmp_ant = single_ant(k)
        antDict[k] = tmp_ant

    # 信息素矩阵初始化
    tao0 = 1/(n*Cnn)
    for na in nodeDict.keys():
        pheromone[na] = {}
        for nb in nodeDict.keys():
            pheromone[na][nb] = tao0
    print(pheromone)
    # pher_matrix = np.array([list(pheromone[pk].values()) for pk in pheromone.keys()])
    # plt.figure(10)
    # ax1 = seaborn.heatmap(pher_matrix, cmap="YlGnBu")
    # ax1.set_title('pheromone matrix in generation: {}'.format(0))

    # 启发式信息矩阵初始化
    beta = 3
    for na in nodeDict.keys():
        heuristic[na] = {}
        for nb in nodeDict.keys():
            if na == nb:
                heuristic[na][nb] = 0
            else:
                heuristic[na][nb] = 1 / (distanceMat[na][nb] ** beta)
    print(heuristic)

    max_heu = 0
    min_heu = math.inf
    for na in nodeDict.keys():
        if max(heuristic[na].values()) ** (1/beta) > max_heu:
            max_heu = max(heuristic[na].values()) ** (1/beta)
        for nb in heuristic[na].keys():
            if heuristic[na][nb] != 0:
                if heuristic[na][nb] ** (1/beta) < min_heu:
                    min_heu = heuristic[na][nb] ** (1/beta)
    for i in range(max_generation):
        history_maxminheu.append(max_heu / min_heu)

    # 信息素、启发式信息结合的矩阵初始化
    for na in nodeDict.keys():
        choice_info[na] = {}
        for nb in nodeDict.keys():
            choice_info[na][nb] = pheromone[na][nb]*heuristic[na][nb]

def update_choice_info():
    for na in nodeDict.keys():
        for nb in nodeDict.keys():
            choice_info[na][nb] = pheromone[na][nb] * heuristic[na][nb]

def ConstructSolutions():
    step = 1
    for k in antDict.keys():
        antDict[k].reInitialization()
        rand_node = random.choice(list(nodeDict.keys()))
        antDict[k].tour.append(rand_node)
        antDict[k].visited[rand_node] = 1
    while step < n:
        for k in antDict.keys():
            # antDict[k].ACSDecisionRule(step)
            antDict[k].NeighborListACSDecisionRule(step)
        step += 1
    # 确保是走一个闭环，而不是非闭环
    for k in antDict.keys():
        antDict[k].tour.append(antDict[k].tour[0])
    for k in antDict.keys():
        antDict[k].computeTourLength()


def SaveInfo(iiii, hbtl, hbt, hbg):
    # 将历次迭代的最短路径蚂蚁的路径长度储存
    min_tour_length = math.inf
    min_k = 0
    for k in antDict.keys():
        if antDict[k].tour_length < min_tour_length:
            min_tour_length = antDict[k].tour_length
            min_k = k
    history_tour_length.append(min_tour_length)
    if min_tour_length < hbtl:
        hbtl = min_tour_length
        hbt = antDict[min_k].tour
        hbg = iiii + 1
    if iiii == max_generation-1:
        # # start输出最后一次迭代最短路径蚂蚁的信息
        # ant_path_x = []
        # ant_path_y = []
        # for i in range(len(antDict[min_k].tour)):
        #     ant_path_x.append(nodeDict[antDict[min_k].tour[i]].x)
        #     ant_path_y.append(nodeDict[antDict[min_k].tour[i]].y)
        # plt.plot(ant_path_x,ant_path_y,'*-')
        # plt.show()
        # print('ant min_k first tour:',antDict[min_k].tour)
        # print('ant min_k first tour length:',antDict[min_k].tour_length)
        # # end输出最后一次迭代最短路径蚂蚁的信息
        best_path_x = []
        best_path_y = []
        for i in range(len(hbt)):
            best_path_x.append(nodeDict[hbt[i]].x)
            best_path_y.append(nodeDict[hbt[i]].y)
        plt.plot(best_path_x, best_path_y, '*-')
        print('history best tour:', hbt)
        print('history best tour length:', hbtl)
        print('History best generation: ',hbg)
    return min_k, hbtl, hbt, hbg


def ACSPheromoneUpdate(best_k,hbt,hbtl):
    # # start AS的全局信息素更新
    # Evaporate()
    # for k in antDict.keys():
    #     antDict[k].DepositPheromone()
    # update_choice_info()
    # # end AS的全局信息素更新
    # start ACS的全局信息素更新

    # Evaporate(best_k,hbt)
    # # antDict[best_k].DepositPheromone(best_k,hbt,hbtl)# 只有那只至今最优蚂蚁会释放信息素
    # for k in antDict.keys():
    #     antDict[k].DepositPheromone(best_k,hbt,hbtl)
    EvaporateAndDepositPheromone(hbt, hbtl)
    update_choice_info()
    # max_pher = 0
    # min_pher = math.inf
    # for nn in pheromone.keys():
    #     if max(pheromone[nn].values()) > max_pher:
    #         max_pher = max(pheromone[nn].values())
    #     if min(pheromone[nn].values()) < min_pher:
    #         min_pher = min(pheromone[nn].values())
    # history_maxminpher.append(max_pher / min_pher)
    # if (len(history_tour_length)) % 1000 == 0:
    #     pher_matrix = np.array([list(pheromone[pk].values()) for pk in pheromone.keys()])
    #     plt.figure((len(history_tour_length)) / 1000 + 11)
    #     ax1 = seaborn.heatmap(pher_matrix, cmap="YlGnBu")
    #     ax1.set_title('pheromone matrix in generation: {}'.format(len(history_tour_length)))
    # end ACS的全局信息素更新

def Evaporate(best_k,hbt):
    rou = 0.1
    # best_tour = antDict[best_k].tour
    best_tour = hbt
    for ii in range(len(best_tour)-1):
        curN = best_tour[ii]
        nextN = best_tour[ii+1]
        pheromone[curN][nextN] = (1 - rou) * pheromone[curN][nextN]
        pheromone[nextN][curN] = pheromone[curN][nextN]
    # for na in nodeDict.keys():
    #     for nb in nodeDict.keys():
    #         pheromone[na][nb] = (1-rou) * pheromone[na][nb]
    #         pheromone[nb][na] = pheromone[na][nb]

def EvaporateAndDepositPheromone(hbt,hbtl):
    rou = 0.1
    for ii in range(len(hbt) - 1):
        curN = hbt[ii]
        nextN = hbt[ii + 1]
        pheromone[curN][nextN] = (1 - rou) * pheromone[curN][nextN] + rou * (1 / hbtl)
        pheromone[nextN][curN] = pheromone[curN][nextN]

def LocalPheromoneUpdate(na,nb):
    tao0 = 1 / (n * Cnn)
    ep = 0.1
    pheromone[na][nb] = (1 - ep) * pheromone[na][nb] + ep * tao0
    pheromone[nb][na] = pheromone[na][nb]
    choice_info[na][nb] = pheromone[na][nb] * heuristic[na][nb]
    choice_info[nb][na] = choice_info[na][nb]

max_generation = 2000
def main():
    InitializeData()
    history_best_tour_length = math.inf
    history_best_tour = []
    history_best_generation = 0
    for i in tqdm(range(max_generation)):
        ConstructSolutions()
        best_k, history_best_tour_length, history_best_tour, history_best_generation = SaveInfo(i, history_best_tour_length, history_best_tour, history_best_generation)
        ACSPheromoneUpdate(best_k,history_best_tour,history_best_tour_length)
    plt.figure(2)
    plt.plot(range(1,max_generation + 1), history_tour_length)
    # plt.figure(3)
    # plt.plot(range(1,max_generation + 1), history_maxminpher, label = 'pheromone')
    # plt.plot(range(1, max_generation + 1), history_maxminheu, label='heuristic')
    # plt.legend()
    plt.show()


main()