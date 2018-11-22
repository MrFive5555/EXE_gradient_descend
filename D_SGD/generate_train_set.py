# 用于生成多个分布式的数据

import os
import random

FILE_NAME = 'COVTYPE_D_SET'
if not os.path.exists(FILE_NAME):
  os.makedirs(FILE_NAME)

# 计算节点数(不包含master)
NODE_CNT = 9
# 每个节点样本数
SAMPLE_PER_NODE = 50
# 总样本大小
SAMPLE_SIZE = NODE_CNT * SAMPLE_PER_NODE
# 数据集总大小
SET_SIZE = 581012

# 分出每个文件应该取的数据的标号
l = random.sample(range(SET_SIZE), SAMPLE_SIZE)
sampleIndex = []
while len(l) != 0:
  sampleIndex.append(l[0:SAMPLE_PER_NODE])
  sampleIndex[-1].sort()
  l = l[SAMPLE_PER_NODE:]

try:
  f = open('covtype.libsvm.binary.scale')
  full_f = open(FILE_NAME+'/full', 'w')
  node = 1
  line = 0
  outputs = []
  for i in range(1, NODE_CNT+1):
    outputs.append(open(FILE_NAME + '/{0}'.format(i), 'w'))
  for (line, vector) in enumerate(f):
    for (i, index) in enumerate(sampleIndex):
      if(len(index) != 0 and line == index[0]):
        outputs[i].write(vector)
        index.pop(0)
        full_f.write(vector)
        break
except FileNotFoundError:
  pass