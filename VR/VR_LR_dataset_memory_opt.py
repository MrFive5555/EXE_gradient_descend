import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import os
import re
import random
import time

# ====================================================
# 加载数据集的函数
def loadDataSet(dataSet, setSize, maxFeature, findingType, **kw):
  print('[{}]正在加载数据集'.format(dataSet))
  X = scipy.sparse.lil_matrix((setSize, maxFeature))
  Y = []
  __dir__ = os.path.dirname(os.path.abspath(__file__))
  dataFile = __dir__ + '/source/' + dataSet
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
  print('[{}]数据集导入完毕，共{}个数据'.format(dataSet, setSize))
  return X, Y

# ====================================================
# 加载目标函数和梯度的函数
def loadFunc(X, Y):
  def _LC(i, w):
    product = X[i:, ].dot(w[0:-1].transpose()) + w[-1]
    product = product[0]
    return product

  ERR = math.log(1e30)
  
  Lambda = 1/X.shape[0]

  def F(w):
    _F_sum = 0
    for i in range(X.shape[0]):
      product = _LC(i, w)
      if -Y[i]*product < ERR:
        _F_sum += math.log(1+math.exp(-Y[i]*product))
      else:
        _F_sum += -Y[i]*product
    return Lambda/2 * np.linalg.norm(w)**2 + _F_sum/X.shape[0]
  # linear core
  def G_Fi_LC(i, LC):
    ext_X = np.hstack((X[i,:].toarray()[0], 1)).transpose()
    if -Y[i]*LC < ERR:
      return - (1-1/(1+math.exp(-Y[i]*LC)))*Y[i]*ext_X
    else:
      return - Y[i]*ext_X
  def check_F_and_accuracy(w):
    correct = 0
    _F_sum = 0
    for i in range(X.shape[0]):
      product = _LC(i, w)
      # 计算正确率
      if Y[i] * product > 0:
        correct += 1
      # 计算函数值
      if -Y[i]*product < ERR:
        _F_sum += math.log(1+math.exp(-Y[i]*product))
      else:
        _F_sum += -Y[i]*product
    _F = Lambda/2 * np.linalg.norm(w)**2 + _F_sum/X.shape[0]
    _accuracy = correct / X.shape[0]
    return _F, _accuracy
  return F, G_Fi_LC, check_F_and_accuracy, _LC

# ====================================================
# 加载函数相关参数的函数
def loadArg(X):
  # 强凸系数
  Lambda = 1/X.shape[0]
  # Lipchitz常数
  L = np.sum([(scipy.sparse.linalg.norm(X[i,:]) + 1)**2 for i in range(X.shape[0])])
  return Lambda, L

# ====================================================
# 加载VR方法的函数
def loadVR():
  # 调用格式
  # dataSetConfig = {
  #   'name': 'covtype',
  #   'dataSet' : 'covtype.libsvm.binary.scale',
  #   'setSize' : 581012,
  #   'maxFeature' : 54,
  #   'findingType' : '1',
  #   'epoch': 1
  # }
  # SAG(**VRConfig)
  def SAG(G_Fi_LC, LC, n, x0, gamma, Lambda, epoch=1, **kw):
    dec_fac = (1-Lambda*gamma)
    store = [0] * n
    G_avg = np.zeros(x0.shape)
    for i in range(n):
      store[i] = LC(i, x0)
      G_avg = G_avg + G_Fi_LC(i, store[i])
    G_avg /= n
    x = x0
    t = 1
    path = []
    path.append(x)
    while True:
      # 迭代
      x = dec_fac*x - gamma * G_avg
      # 更新梯度表
      i = random.randint(0, n-1)
      oldGradient = G_Fi_LC(i, store[i])
      store[i] = LC(i, x)
      G_avg = G_avg + (G_Fi_LC(i, store[i]) - oldGradient) / n
      if t % n == 0:
        path.append(x)
        print('[SAG]已迭代{0:.0f}趟，||x||={1}'.format(t/n, np.linalg.norm(x)))
      if t == epoch * n:
        break
      t += 1
    solution = x
    return solution, path

  def SVRG(G_Fi_LC, LC, n, x0, gamma, Lambda, epoch=1, innerLoop=0, **kw):
    dec_fac = (1-Lambda*gamma)
    if innerLoop == 0:
      innerLoop = 2 * n
    else:
      innerLoop = innerLoop * n
    x = x0
    t = 1
    path = []
    path.append(x)
    end = False
    # 外层迭代
    while not end:
      store = [0] * n
      G_avg = np.zeros(x0.shape)
      for i in range(n):
        store[i] = LC(i, x0)
        G_avg = G_avg + G_Fi_LC(i, store[i])
      G_avg = G_avg / n
      for _ in range(innerLoop): 
        # 内层迭代
        i = random.randint(0, n-1)
        x = dec_fac*x - gamma * (G_Fi_LC(i, LC(i, x)) - store[i] + G_avg)
        # 显示进度
        if t % n == 0:
          path.append(x)
          print('[SVRG]已迭代{0:.0f}趟，||x||={1}'.format(t/n, np.linalg.norm(x)))
        if t == epoch * n:
          end = True
          break
        t += 1
    solution = x
    return solution, path

  def SAGA(G_Fi_LC, LC, n, x0, gamma, Lambda, epoch=1, **kw):
    dec_fac = (1-Lambda*gamma)
    store = [0] * n
    G_avg = np.zeros(x0.shape)
    for i in range(n):
      store[i] = LC(i, x0)
      G_avg = G_avg + G_Fi_LC(i, store[i])
    G_avg /= n
    x = x0
    t = 1
    path = []
    path.append(x)
    while True:
      # 更新梯度表
      i = random.randint(0, n-1)
      oldGradient = G_Fi_LC(i, store[i])
      store[i] = LC(i, x)
      newGradient = G_Fi_LC(i, store[i])
      # 迭代
      x = dec_fac*x - gamma*(newGradient - oldGradient + G_avg)
      G_avg = G_avg + (newGradient - oldGradient) / n
      if t % n == 0:
        path.append(x)
        print('[SAGA]已迭代{0:.0f}趟，||x||={1}'.format(t/n, np.linalg.norm(x)))
      if t == epoch * n:
        break
      t += 1
    return x, path
  
  def calMin(G_Fi_LC, LC, n, x0, gamma, Lambda, epoch=1, **kw):
    print('开始计算函数最小值')
    # SGD
    x = x0
    for t in range(1, n+1):
      step = gamma / t
      i = random.randint(0, n-1)
      SG_F = G_Fi_LC(i, LC(i, x))
      x = (1-gamma*Lambda/t)*x - step * SG_F

    # SAGA
    dec_fac = (1-Lambda*gamma)
    store = [0] * n
    G_avg = np.zeros(x0.shape)
    for i in range(n):
      store[i] = LC(i, x0)
      G_avg = G_avg + G_Fi_LC(i, store[i])
    G_avg /= n
    iterPass = 0
    while True:
      print('正在计算函数最小值，已迭代{0}趟'.format(iterPass))
      if iterPass == epoch:
        break
      iterPass += 1
      for _ in range(n):
        # 更新梯度表
        i = random.randint(0, n-1)
        oldGradient = G_Fi_LC(i, store[i])
        store[i] = LC(i, x)
        newGradient = G_Fi_LC(i, store[i])
        # 迭代
        x = dec_fac*x - gamma * (newGradient - oldGradient + G_avg)
        G_avg = G_avg + (newGradient - oldGradient) / n

    # SVRG
    while True:
      print('正在计算函数最小值，已迭代{0}趟'.format(iterPass))
      if iterPass == 2*epoch:
        break
      iterPass += 1
      for i in range(n):
        store[i] = LC(i, x0)
        G_avg = G_avg + G_Fi_LC(i, store[i])
      G_avg = G_avg / n
      for _ in range(n): 
        # 内层迭代
        i = random.randint(0, n-1)
        x = dec_fac*x - gamma * (G_Fi_LC(i, LC(i, x)) - store[i] + G_avg)
    return x
  return SAG, SVRG, SAGA, calMin

# ====================================================
# 数据集配置
# dataSetConfig = {
#   'name': 'covtype',
#   'dataSet' : 'covtype.libsvm.binary.scale',
#   'setSize' : 581012,
#   'maxFeature' : 54,
#   'findingType' : '1',
#   'epoch': 50
# }
# dataSetConfig = {
#   'name': 'covtype_1000',
#   'dataSet' : 'covtype.libsvm.binary.scale.train1000',
#   'setSize' : 1000,
#   'maxFeature' : 54,
#   'findingType' : '1',
#   'epoch': 50
# }
# dataSetConfig = {
#   'name': 'a1a',
#   'dataSet' : 'a1a',
#   'setSize' : 1605,
#   'maxFeature' : 123,
#   'findingType' : '+1',
#   'epoch': 50
# }
dataSetConfig = {
  'name': 'RVC',
  'dataSet' : 'RVC_Index_EN-EN',
  'setSize' : 18758,
  'maxFeature' : 21540,
  'findingType' : 'CCAT',
  'epoch': 17
}
# ====================================================

def main():
  # 加载数据集
  X, Y = loadDataSet(**dataSetConfig)
  # 加载函数
  _, G_Fi_LC, check_F_and_accuracy, LC = loadFunc(X, Y)
  # 计算强凸系数和Lipchitz常数
  Lambda, L = loadArg(X)
  # 加载VR方法
  SAG, SVRG, SAGA, calMin = loadVR()

  w0 = np.array([0.0] * (dataSetConfig['maxFeature']+1))

  VRConfig = {
    'G_Fi_LC': G_Fi_LC,
    'LC': LC,
    'n': dataSetConfig['setSize'],
    'x0': w0,
    'gamma': 1/L,
    'Lambda': Lambda,
    'epoch': dataSetConfig['epoch'],
  }

  # 求最小值
  wmin = calMin(**VRConfig)
  (Fmin, accuracyMin) = check_F_and_accuracy(wmin)

  VRConfig['epoch'] = int(VRConfig['epoch']*0.6)

  runConfigs = [
    (SAG, r'SAG'),
    (SVRG, r'SVRG'),
    (SAGA, r'SAGA'),
  ]
  for (VR, label) in runConfigs:
    SHOW_PASS = VRConfig['epoch']
    _, path = VR(**VRConfig)

    VR_err = []
    VR_accuracy = []
    for (i, p) in enumerate(path):
      (f, acc) = check_F_and_accuracy(p)
      VR_err.append(math.log10(f-Fmin))
      VR_accuracy.append(acc)
      if i % 10 == 0:
        print('已计算{0}个函数值'.format(i))
    plt.figure(1)
    plt.plot(range(SHOW_PASS), VR_err[:SHOW_PASS], label=label)
    plt.figure(2)
    plt.plot(range(SHOW_PASS), VR_accuracy[:SHOW_PASS], label=label)

  timeStamp = time.strftime('%m%d%H%M%S', time.localtime())
  __dir__ = os.path.dirname(os.path.abspath(__file__))
  figName = __dir__ + '/result/' + dataSetConfig['name'] + '_' + timeStamp

  plt.figure(1)
  plt.title(dataSetConfig['name'] + r'_convergence_curve')
  plt.xlabel(r'$T$')
  plt.ylabel(r'$log_{10}(f(w) - f(w^*))$')
  plt.legend()
  plt.savefig(figName+'converge')

  plt.figure(2)
  plt.plot([0, SHOW_PASS-1], [accuracyMin]*2, '--')
  plt.title(dataSetConfig['name'] + r'_accuracy')
  plt.xlabel(r'$T$')
  plt.ylabel(r'$classification_accuracy$')
  plt.legend()
  plt.savefig(figName+'accuracy')

  plt.show()

if __name__ == '__main__':
  main()