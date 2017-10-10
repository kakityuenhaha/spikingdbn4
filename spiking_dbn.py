import matplotlib.pyplot as plt
import numpy as np
import pyNN.spiNNaker as p
import relu_utils as alg
import spiking_relu as sr
import random
import mnist_utils as mu
import os.path
#import sys

#USAGE: spiking_dbn.py scaled_weight b10_epoc5

#w_listf = sys.argv[1]
#dbn_f = sys.argv[2]
w_listf = 'scaled_weight'
#dbn_f = 'special'
#dbnet = alg.load_dict(dbn_f)
import scipy.io as sio
tmp_x=np.double(sio.loadmat('mnist_uint8.mat')['test_x'])
#tmp_x=np.transpose(tmp_x,(2,0,1))
#tmp_x=np.reshape(tmp_x,(tmp_x.shape[0],28*28,),order='F')


tmp_y=np.double(sio.loadmat('mnist_uint8.mat')['test_y'])
tmp_y=np.argmax(tmp_y,axis=1)
cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 1.,
                   'tau_syn_E': 1.0,
                   'tau_syn_I': 1.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }
                   
if os.path.isfile('%s.pkl'%w_listf):
    scaled_w = alg.load_dict(w_listf)
    w = scaled_w['w']
    k = scaled_w['k']
    x0 = scaled_w['x0']
    y0 = scaled_w['y0']
    print 'found w_list file'
#else:
#    w, k, x0, y0 = sr.w_adjust(dbnet, cell_params_lif)
#    scaled_w = {}
#    scaled_w['w'] = w
#    scaled_w['k'] = k
#    scaled_w['x0'] = x0
#    scaled_w['y0'] = y0
#    alg.save_dict(scaled_w, w_listf)

num_test = 100
random.seed(0)
dur_test = 1000
silence = 200
#test_x = dbnet['test_x']
#result_list = np.zeros((test_x.shape[0], 2))
count = 0
for offset in range(0, tmp_x.shape[0], num_test):
#for offset in range(0, 1000, num_test):
    print offset
    test = tmp_x[offset:(offset+num_test), :]
    test=test*60.
    spike_source_data = sr.gen_spike_source(test)                
    spikes = sr.run_test(w, cell_params_lif, spike_source_data)
    spike_count = list()

    for i in range(w[-1].shape[1]):
        index_i = np.where(spikes[:,0] == i)
        spike_train = spikes[index_i, 1]
        temp = sr.counter(spike_train, range(0, (dur_test+silence)*num_test,dur_test+silence), dur_test)
        spike_count.append(temp)
    spike_count = np.array(spike_count)/(dur_test / 1000.)
    r = np.argmax(spike_count, axis=0)
    correct = np.sum(r == tmp_y[offset:offset+num_test]).astype(int) #- len(np.where(spike_count.max(axis=0)==0)[0])
    print 'correct number'#
    print correct
    count = count + correct
print 'count'
print count
    #result_list[offset:offset+num_test, 0] = r
    #result_list[offset:offset+num_test, 1] = (result_list[offset:offset+num_test, 0] == dbnet['test_y'][offset:offset+num_test]).astype(int)
    #index = np.where(spike_count.max(axis=0)==0)[0]
    #result_list[offset+index, 1] = -1
    #print spike_count, np.argmax(spike_count, axis=0), dbnet['test_y'][:10]   
    #np.save('result_list1', result_list)


