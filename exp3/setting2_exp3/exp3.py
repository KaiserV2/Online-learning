"""
Exp3 on 10 arms bandit problem in TCP
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import collections

plt.rcParams['font.size'] = 16

random.seed(2)
np.random.seed(2)
K = 10
T = 20000

def read_data_static(file_name):
    with open(file_name, "r") as input:
        lines = []
        for line in input:
            fs = [float(f) for f in line.split(", ")]
            lines.append(fs)
        return lines

def read_data_dynamic(file_name):
    with open(file_name, "r") as input:
        lines = []
        for line in input:
            line = line.rstrip('\n')
            fs = [float(f) for f in line[0:-1].split(' ')]
            lines.append(fs)
        return lines


def calculate_freq(counter_log):
    plot_freq = []
    counter_freq = collections.Counter( counter_log )
    for i in range( 10 ):

        if i in counter_freq:
            plot_freq.append( counter_freq[i] / 5000 )
        else:
            plot_freq.append( 0 )
    return plot_freq

loss_data_static = read_data_static("dataset1.txt")
loss_data_dynamic = read_data_dynamic("dataset2.txt")


def Exp3(loss_data, T ,K ,etha):
    P = np.ones( 10 ) / 10
    estimated_loss = np.ones( 10 ) / 10
    cumulative_loss = 0
    avg_cum_loss = []

    counter_log = []

    for t in range( T ):

        # ArmLossVector = [x/4 for x in loss_data[t]]
        ArmLossVector = loss_data[t]
        # Choose arm with P
        arm_index = np.random.choice( list( range( K ) ) ,p=P )
        cumulative_loss += abs( ArmLossVector[arm_index] )
        avg_cum_loss.append( cumulative_loss / (t + 1) )


        # exp3 update
        for i in range( 10 ):
            if i == arm_index:
                estimated_loss[i] = ArmLossVector[i] / P[i]
            else:
                estimated_loss[i] = 0
            P[i] *= np.exp( -etha * estimated_loss[i] )

        P /= np.sum( P )

        counter_log.append(np.argmax(P))

        # if t == 4999:     #4
        #     plot_freq = calculate_freq(counter_log)
        #     freq_arms = [0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0]
        #     plt.plot( range( 10 ) ,freq_arms ,color='black' ,linestyle='--' ,label='Ground Truth' )
        #     plt.plot( range( 10 ) ,plot_freq ,color='red' ,label='exp3' )
        #     plt.legend()
        #     plt.title( "xt_exp3 5000 iteration" )
        #     plt.tight_layout()
        #     plt.savefig('../figs/exp3/exp3_dynamic_5000.png', dpi=250)
        #     plt.show()
        #
        # if t == 9999:   #0
        #     plot_freq = calculate_freq(counter_log[4999:9999])
        #     freq_arms = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #     plt.plot( range( 10 ) ,freq_arms ,color='black' ,linestyle='--' ,label='Ground Truth' )
        #     plt.plot( range( 10 ) ,plot_freq ,color='red' ,label='exp3' )
        #     plt.legend()
        #     plt.title( "xt_exp3 10000 iteration" )
        #     plt.tight_layout()
        #     plt.savefig( '../figs/exp3/exp3_dynamic_10000.png' ,dpi=250 )
        #     plt.show()
        #
        # if t == 14999:   #0, 7
        #     plot_freq = calculate_freq(counter_log[9999:14999])
        #     freq_arms = [0.5 ,0 ,0 ,0 ,0 ,0 ,0 ,0.5 ,0 ,0]
        #     plt.plot( range( 10 ) ,freq_arms ,color='black' ,linestyle='--' ,label='Ground Truth' )
        #     plt.plot( range( 10 ) ,plot_freq ,color='red' ,label='exp3' )
        #     plt.legend()
        #     plt.title( "xt_exp3 15000 iteration" )
        #     plt.tight_layout()
        #     plt.savefig( '../figs/exp3/exp3_dynamic_15000.png' ,dpi=250 )
        #     plt.show()
        #
        # if t == 19999: #1, 7
        #     plot_freq = calculate_freq(counter_log[14999:19999] )
        #     freq_arms = [0 ,0.5 ,0 ,0 ,0 ,0 ,0 ,0.5 ,0 ,0]
        #     plt.plot( range( 10 ) ,freq_arms ,color='black' ,linestyle='--' ,label='Ground Truth' )
        #     plt.plot( range( 10 ) ,plot_freq ,color='red' ,label='exp3' )
        #     plt.legend()
        #     plt.title( "xt_exp3 20000 iteration" )
        #     plt.tight_layout()
        #     plt.savefig( '../figs/exp3/exp3_dynamic_20000.png' ,dpi=250 )
        #     plt.show()

    return P, avg_cum_loss

#
# ethaVector = [0.0001, 0.001, 0.01, 0.02, 0.03]
# #print(ethaVector, "-----ethaVector")
#
# all_etha_loss = []
# for etha in ethaVector:
#     R, avg_loss = Exp3( loss_data_static, T ,K , etha )
#     all_etha_loss.append(avg_loss)
#
# plt.loglog( range(T) ,all_etha_loss[0] ,color='b', linestyle='-', label='etha=0.0005')
# plt.loglog( range(T) ,all_etha_loss[1] ,color='g', linestyle='--',label='etha=0.001')
# plt.loglog( range(T) ,all_etha_loss[2] ,color='r', linestyle=':',label='etha=0.01')
# plt.loglog( range(T) ,all_etha_loss[3] ,color='c', linestyle='-.',label='etha=0.02')
# plt.loglog( range(T) ,all_etha_loss[4] ,color='m', linestyle='-', label='etha=0.03')
#
# plt.title( "Static dist: exp3")
# plt.xlabel( "time" )
# plt.ylabel( "average cumulative loss" )
# plt.legend()
# plt.tight_layout()
# plt.savefig('../figs/exp3_static.png', dpi=300)
# plt.show()

ethaVector = [0.0005, 0.001, 0.01, 0.02, 0.03]
all_etha_loss = []
for etha in ethaVector:
    R, avg_loss = Exp3( loss_data_dynamic, T ,K , etha )
    all_etha_loss.append(avg_loss)

plt.loglog( range(T) ,all_etha_loss[0] ,color='b', linestyle='-', label='etha=0.0005')
plt.loglog( range(T) ,all_etha_loss[1] ,color='g', linestyle='--',label='etha=0.001')
plt.loglog( range(T) ,all_etha_loss[2] ,color='r', linestyle=':',label='etha=0.01')
plt.loglog( range(T) ,all_etha_loss[3] ,color='c', linestyle='-.',label='etha=0.02')
plt.loglog( range(T) ,all_etha_loss[4] ,color='m', linestyle='-', label='etha=0.03')

plt.title( "Dynamic dist: exp3")
plt.xlabel( "time" )
plt.ylabel( "average cumulative loss" )
plt.legend()
plt.tight_layout()
plt.savefig('../figs/exp3_dynamic.png', dpi=300)
plt.show()


#
# Best lr rate

R, avg_loss = Exp3( loss_data_dynamic, T ,K , etha=0.0005 )
plt.loglog( range(T) ,avg_loss ,color='b', label='etha=0.0005')

plt.title( "average cumulative loss of exp3")
plt.xlabel( "time" )
plt.ylabel( "average cumulative loss" )
plt.legend()
plt.show()
