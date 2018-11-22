# 运行
# mpiexec -np 12 python3 D_SAGA_COVTYPE.py > out

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import re
import scipy.sparse
import scipy.sparse.linalg
import sys
import time

from mpi4py import MPI

# dataSetConfig = {
#   'name': 'covtype',
#   'dataSet' : 'COVTYPE_D_SET',
#   'dataPerNode': 50,
#   'honestNodeSize': 9,
#   'byzantineNodeSize': 1,
#   'maxFeature' : 54,
#   'findingType' : '1',
#   'epoch': 10
# }
dataSetConfig = {
  'name': 'covtype',
  'dataSet' : 'COVTYPE_D_SET_50',
  'dataPerNode': 50,
  'honestNodeSize': 3,
  'byzantineNodeSize': 0,
  'maxFeature' : 54,
  'findingType' : '1',
  'epoch': 5
}
DATA_PER_NODE = dataSetConfig['dataPerNode']
NODE_SIZE = 1 + dataSetConfig['honestNodeSize'] + dataSetConfig['byzantineNodeSize']
SET_SIZE = DATA_PER_NODE * dataSetConfig['honestNodeSize']
maxFeature = dataSetConfig['maxFeature']
findingType = dataSetConfig['findingType']

# ====================================================
# 初始化MPI
comm = MPI.COMM_WORLD
NODE_RANK = comm.Get_rank()
if NODE_SIZE != comm.Get_size():
  print('ERROR, NODE_SIZE != PROCESS_SIZE')
  exit(1)

# ====================================================
# 报告函数
def log(*k, **kw):
  timeStamp = time.strftime('[%H:%M:%S] ', time.localtime())
  print(timeStamp, end='')
  print(*k, **kw)
  sys.stdout.flush()

# ====================================================
# 构建矩阵
def loadXY(file, size):
  X = scipy.sparse.lil_matrix((size, maxFeature))
  Y = []
  __dir__ = os.path.dirname(os.path.abspath(__file__))
  dataFile = __dir__ + '/' + dataSetConfig['dataSet'] + '/' + file
  try:
    f = open(dataFile, 'r')
    for (line, vector) in enumerate(f):
      (cat, data) = vector.split(' ', 1)
      if cat == findingType:
        Y.append(1)
      else:
        Y.append(-1)
      for piece in data.strip().split(' '):
        match = re.search(r'(\S+):(\S+)', piece)
        feature = int(match.group(1)) - 1 # 数据集从1开始
        value = float(match.group(2))
        # 插入矩阵
        X.rows[line].append(feature)
        X.data[line].append(value)
    f.close()
    log('节点{}: 加载数据集完成({})'.format(NODE_RANK, file))
    return X, Y
  except FileNotFoundError:
    return None, None

# ====================================================
def loadL(X):
  if NODE_RANK == 0:
    L = 0
  elif NODE_RANK <= dataSetConfig['honestNodeSize']:
    assert(DATA_PER_NODE == X.shape[0])
    L = np.sum([(scipy.sparse.linalg.norm(X[i,:]) + 1)**2 for i in range(X.shape[0])])
  else:
    L = 0
  L = comm.allreduce(L, MPI.SUM)
  # 强凸系数
  Lambda = 1/SET_SIZE
  L = Lambda + 1/(4*SET_SIZE) * L
  return L

# ====================================================
def loadFunc(X, Y):
  Lambda = 1/SET_SIZE
  ERR = math.log(1e30)
  def Product(x, w):
    product = x.dot(w[0:-1].transpose()) + w[-1]
    product = product[0]
    return product

  # ====================================================
  # 梯度
  def G_Fi(i, w):
    product = X[i, :].dot(w[0:-1].transpose()) + w[-1]
    product = product[0]
    # product = Product(X[i, :], w)
    ext_X = np.hstack((X[i,:].toarray()[0], 1)).transpose()
    if -Y[i]*product < ERR:
      return Lambda * w - (1-1/(1+math.exp(-Y[i]*product)))*Y[i]*ext_X
    else:
      return Lambda * w - Y[i]*ext_X
  # ====================================================
  # 计算函数值
  if NODE_RANK != 0:
    return None, G_Fi
  else:
    X, Y = loadXY("full", SET_SIZE)
    def F(w_path):
      res = [0] * len(w_path)
      for i in range(X.shape[0]):
        for (k, w) in enumerate(w_path):
          product = Product(X[i, :], w)
          if -Y[i]*product < ERR:
            res[k] += math.log(1+math.exp(-Y[i]*product))
          else:
            res[k] += -Y[i]*product
        if i % 10 == 0:
          log('[计算函数值] 已计算{}个数据'.format(i))
      for (k, w) in enumerate(w_path):
        res[k] = Lambda/2 * np.linalg.norm(w)**2 + res[k]/X.shape[0]
      return res
    return F, G_Fi

def loadMasterFunc(**kw):
  def init():
    pass
  def update(w):
    return w
  return init, update

def loadWorkerFunc(x0, dataPerNode, gamma, G_Fi, **kw):
  store = [x0] * dataPerNode
  G_avg = np.zeros(x0.shape)
  def init():
    nonlocal store, G_avg
    for i in range(dataPerNode):
      store[i] = G_Fi(i, x0)
      G_avg += store[i]
    G_avg /= dataPerNode
  def update(w):
    nonlocal store, G_avg, gamma
    i = random.randint(0, dataPerNode-1)
    # 更新梯度表
    (old_G, new_G) = (store[i], G_Fi(i, w))
    store[i] = new_G
    gradient = new_G - old_G + G_avg
    G_avg += (new_G - old_G) / dataPerNode
    # 梯度下降
    w = w - gamma * gradient
    return w
  return init, update

def loadByzantineFunc(**kw):
  def init():
    pass
  def update(w):
    for i, _ in enumerate(w):
      w[i] = 4*random.random()*w[i]-2
    return w
  return init, update

# ====================================================
def loadFedSAGA():
  def SAGA_aggregate(wList):
    res = np.mean(wList, axis=0)
    return res
  def FedSAGA(G_Fi, setSize, dataPerNode, x0, gamma, node_init, node_update, epoch=1, aggregate=SAGA_aggregate, **kw):
    # 初始化存储
    node_init()
    w = x0
    path = [x0]
    
    for e in range(epoch):
      for _ in range(setSize):
        # 同步状态
        comm.Bcast(w, 0)
        # 节点开始计算
        w = node_update(w)
        # 收集新位置信息
        wList = comm.gather(w, root=0)
        if NODE_RANK == 0:
          # 去除master的梯度
          wList.pop(0)
          # 聚合
          w = aggregate(wList)
      if NODE_RANK == 0:
        path.append(w)
        log('已完成{}趟迭代'.format(e+1))
    return w, path
  return FedSAGA
def CentralSAGA(G_Fi, setSize, x0, gamma, epoch=1, innerLoop=0, **kw):
  store = [x0] * setSize
  G_avg = np.zeros(x0.shape)
  for i in range(setSize):
    store[i] = G_Fi(i, x0)
    G_avg = G_avg + store[i]
  G_avg /= setSize
  x = x0
  path = [x0]
  for e in range(epoch):
    for _ in range(setSize):
      # 更新梯度表
      i = random.randint(0, setSize-1)
      old_G = store[i]
      new_G = G_Fi(i, x)
      store[i] = new_G
      gradient = new_G - old_G + G_avg
      G_avg = G_avg + (new_G - old_G) / setSize
      x = x - gamma * gradient
    path.append(x)
    print('[SAGA]已迭代{0:.0f}趟'.format(e+1))
  return x, path

# ====================================================
# 聚合函数
def aggregate_line(wList):
  res = np.mean(wList, axis=0)
  return res
def aggregate_geometric(wList):
  max_iter = 1000
  tol = 1e-7
  guess = np.mean(wList, axis=0)
  for _ in range(max_iter):
    dist_li = [np.linalg.norm(w - guess) for w in wList]
    dist_li = [d if d != 0 else 1 for d in dist_li]
    temp1 = np.sum([w / dist for w, dist in zip(wList, dist_li)], axis=0)
    temp2 = np.sum([1.0 / dist for dist in dist_li])
    guess_next = temp1 / temp2
    guess_movement = np.linalg.norm(guess - guess_next)
    guess = guess_next
    if guess_movement <= tol:
      break
  return guess
def aggregate_order_three(wList):
  res = np.mean([w**3 for w in wList], axis=0)
  res = np.cbrt(res)
  return res
def aggregate_hyperbolic_sin(wList):
  res = np.mean([np.sinh(w) for w in wList], axis=0)
  res = np.arcsinh(res)
  return res
def aggregate_sigmoid(wList):
  res = np.mean([1/(1+np.exp(-w)) for w in wList], axis=0)
  res = np.log(1/(1-res)-1)
  return res
# ====================================================
if NODE_RANK <= dataSetConfig['honestNodeSize']:
  X, Y = loadXY(str(NODE_RANK), DATA_PER_NODE)
  F, G_Fi = loadFunc(X, Y)
else:
  X, Y = 0, 1
  F, G_Fi = lambda x: x, lambda x: x
L = loadL(X)
Lambda = 1/SET_SIZE
FedSAGA = loadFedSAGA()
w0 = np.array([0.0] * (maxFeature+1))

VRConfig = {
  'G_Fi': G_Fi,
  'setSize': SET_SIZE,
  'dataPerNode': DATA_PER_NODE,
  'x0': w0,
  'gamma': 1/L,
  'epoch': dataSetConfig['epoch'],
  'node_init': None,
  'node_update': None,
}

# 加载SAGA的初始化和更新函数
if NODE_RANK == 0:
  (VRConfig['node_init'], VRConfig['node_update']) = loadMasterFunc(**VRConfig)
elif NODE_RANK <= dataSetConfig['honestNodeSize']:
  (VRConfig['node_init'], VRConfig['node_update']) = loadWorkerFunc(**VRConfig)
else:
  (VRConfig['node_init'], VRConfig['node_update']) = loadByzantineFunc(**VRConfig)

# 不同的任务
jobList = [
  {'label': 'line aggregation', 'aggregate': aggregate_line},
  {'label': 'geometric aggregation', 'aggregate': aggregate_geometric},
  {'label': 'sigmoid aggregation', 'aggregate': aggregate_sigmoid},
  # {'label': 'order_three aggregation', 'aggregate': aggregate_order_three},
  # {'label': 'hyperbolic_sin aggregation', 'aggregate': aggregate_hyperbolic_sin},
]
for job in jobList:
  _VRConfig = VRConfig.copy()
  _VRConfig['aggregate'] = job['aggregate']
  _, path = FedSAGA(**_VRConfig)
  job['path'] = path
del X, Y

if NODE_RANK == 0:
  # 作图的长度
  SHOW_LENGTH = dataSetConfig['epoch']
  
  for job in jobList:
    F_path = F(job['path'])
    plt.plot(range(SHOW_LENGTH+1), F_path, label=job['label'])

  # 中心化SAGA的收敛曲线
  centralVRConfig = VRConfig.copy()
  centralVRConfig['epoch'] *= 3
  wmin, cp = CentralSAGA(**VRConfig)
  F_c = F(cp[0:SHOW_LENGTH+1])
  F_min = F([wmin])[0]
  plt.plot(range(SHOW_LENGTH+1), F_c, label='Central SAGA')
  
  timeStamp = time.strftime('%m%d%H%M%S', time.localtime())
  __dir__ = os.path.dirname(os.path.abspath(__file__))
  figName = __dir__ + '/result/' + dataSetConfig['name'] + '_' + timeStamp

  plt.title(dataSetConfig['name'])
  plt.xlabel(r'$epoch$')
  plt.ylabel(r'$f(w)$')
  plt.ylim(ymax=F([w0])[0]*1.1)
  plt.legend()

  # plt.savefig(figName+'_accuracy')
  plt.show()
