import numpy as np
import matplotlib.pyplot as plt
import mnist_utils as mu
import maths_utils as matu
import random
import pdb

neuron = 'relu'
if neuron == 'relu':
    import relu_utils as alg
elif neuron == 'sig':
    import sigmoid_utils as alg
elif neuron == 'krelu':
    import kbrelu_utils as alg
    
train_x, train_y = mu.get_train_data()
random.seed(0)
label_list = np.array(train_y).astype(int)
index_digit = np.where(label_list>=0)[0]
train_num = len(index_digit)-1
index_train = index_digit[0:train_num]
Data_v = np.array(train_x[index_train]).astype(float)
Data_v = Data_v/255.
Labels = np.array(train_y[index_train]).astype(int)
Data_l = np.zeros((train_num, 10))
for i in range(train_num):
    Data_l[i, Labels[i]] = 1.
'''
nodes = [784, 500, 500, 2000, 10]
bsize = 10
iteration = 50
'''

nodes = [784, 100, 100, 400, 10]
bsize = 10
iteration = 1

dbnet = alg.init_label_dbn(Data_v, Data_l, nodes, eta=1e-3, batch_size=bsize, epoc=iteration)
dbnet = alg.greedy_train(dbnet)
#dbnet['train_x']=[]
#dbnet['train_y']=[]
dbnet['train_x']= Data_v
dbnet['train_y']= Data_l
predict, recon = alg.greedy_recon(dbnet, Data_v[0])


test_x, test_y = mu.get_test_data()
index_digit = np.where(test_y>=0)[0]
train_num = len(index_digit)-1
index_train = index_digit[0:train_num]
test_v = np.array(test_x[index_train]).astype(float)
test_v = test_v/255.
test_l = np.array(test_y[index_train]).astype(int)
#dbnet = alg.test_label_data(dbnet, test_v, test_l)
dbnet['test_x'] = test_v
dbnet['test_y'] = test_l
pdb.set_trace()
predict, result = alg.dbn_greedy_test(dbnet)
print np.where(result==False)[0].shape, np.where(result==1)[0].shape, np.where(result==-1)[0].shape

dbn_file = '%s_greedy_b%d_epoc%d'%(neuron, bsize, iteration)
alg.save_dict(dbnet, dbn_file)

'''
#fine training
dbnet['train_x'] = Data_v
dbnet['train_y'] = Data_l
dbnet = alg.fine_train(dbnet)
predict, recon = alg.dbn_recon(dbnet, Data_v[0])
dbn_file = '%s_fine_b%d_epoc%d'%(neuron, bsize, iteration)
alg.save_dict(dbnet, dbn_file)
predict, result = alg.dbn_test(dbnet)
print np.where(result==False)[0].shape, np.where(result==1)[0].shape, np.where(result==-1)[0].shape
'''

