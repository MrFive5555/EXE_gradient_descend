# 运行
# mpiexec -np 4 python3 D_SGD_COVTYPE.py

import linecache
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import re
import scipy.sparse
import scipy.sparse.linalg

from mpi4py import MPI

dataSet = 'COVTYPE_D_SET'

findingType = '1'
# SET_SIZE = 581012
NODE_SIZE = 4
maxFeature = 54

# ====================================================
# 初始化MPI
comm = MPI.COMM_WORLD
NODE_RANK = comm.Get_rank()
if NODE_SIZE != comm.Get_size():
  print("ERROR, NODE_SIZE != PROCESS_SIZE")
  exit(1)

DATA_PER_NODE = 50
SET_SIZE = DATA_PER_NODE * (NODE_SIZE - 1)

# ====================================================
# 构建矩阵
X = scipy.sparse.lil_matrix((DATA_PER_NODE, maxFeature))
Y = []

if NODE_RANK == 0:
  print('[COVTYPE]正在导入数据集')
else:
  __dir__ = os.path.dirname(os.path.abspath(__file__))
  dataFile = __dir__ + '/' + dataSet + '/' + str(NODE_RANK)
  with open(dataFile, 'r') as f:
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

# ====================================================
# 定义函数
Lambda = 1/SET_SIZE
# Lipchitz常数
L = np.sum([(scipy.sparse.linalg.norm(X[i,:]) + 1)**2 for i in range(X.shape[0])])
comm.reduce(L, MPI.SUM, 0)
L = Lambda + 1/(4*SET_SIZE) * L
if NODE_RANK == 0:
  print('L: {}, Lambda: {}'.format(L, Lambda))

def Product(x, w):
  product = x.dot(w[0:-1].transpose()) + w[-1]
  product = product[0]
  return product
ERR = math.log(1e30)

def F(w):
  sum = 0
  for i in range(DATA_PER_NODE):
    product = Product(X[i, :], w)
    if -Y[i]*product < ERR:
      sum += math.log(1+math.exp(-Y[i]*product))
    else:
      sum += -Y[i]*product
  return Lambda/2 * np.linalg.norm(w)**2 + sum / DATA_PER_NODE

def G_Fi(i, w):
  product = Product(X[i, :], w)
  ext_X = np.hstack((X[i,:].toarray()[0], 1)).transpose()
  if -Y[i]*product < ERR:
    return Lambda * w - (1-1/(1+math.exp(-Y[i]*product)))*Y[i]*ext_X
  else:
    return Lambda * w - Y[i]*ext_X

def G_F(w):
  sum = np.zeros(w.shape)
  for i in range(X.shape[0]):
    product = Product(X[i, :], w)
    ext_X = np.hstack((X[i,:].toarray()[0], 1)).transpose()
    sum = sum + (1-1/(1+math.exp(-Y[i]*product)))*Y[i]*ext_X
  return Lambda * w - sum / X.shape[0]

w0 = np.array([0.0] * (maxFeature+1))

MAX_PASS = 0.1

cList = [1e-2, 1e-3, 1e-4]
for c in cList:
  w = w0.copy()
  t = 1
  w_path = []
  while t < MAX_PASS * SET_SIZE:
    w_path.append(w.copy())
    comm.Bcast(w, 0)
    gradient = np.zeros(w.shape)
    gradient_part = np.zeros(w.shape)
    if NODE_RANK != 0:
      i = random.randint(0, DATA_PER_NODE-1)
      gradient_part = G_Fi(i, w)
    comm.Reduce(gradient_part, gradient, MPI.SUM, 0)
    if NODE_RANK == 0:
      gradient = gradient / (NODE_SIZE - 1)
      stepSize = c / (Lambda * t)
      w = w - stepSize * gradient
    t += 1
    if NODE_RANK == 0 and t % SET_SIZE == 0:
      print('已完成{}趟迭代'.format(t))

  F_path = np.array([0.0] * len(w_path))
  F_path_part = np.array([0.0] * len(w_path))
  if NODE_RANK != 0:
    F_path_part = np.array([F(w) for w in w_path])
  comm.Reduce(F_path_part, F_path, MPI.SUM, 0)
  F_path = F_path / NODE_SIZE

  if NODE_RANK == 0:
    F_path_log = [math.log10(f) for f in F_path]
    plt.plot(range(1, len(F_path)+1), F_path_log, label='c={:.2e}'.format(c))
    import CheckResult
    print('[c:{:.2e}] 正确率:{}'.format(c, CheckResult.check(w)))
    
if NODE_RANK == 0:
  plt.xlabel(r'$T$')
  plt.ylabel(r'$log_{10}(f(w))$')
  plt.legend()
  plt.show()