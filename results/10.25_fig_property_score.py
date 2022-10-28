import matplotlib.pyplot as plt
import numpy as np

#print(data.keys())

def plot(data, label):
    NNPP_list = []
    OEPP_list = []
    BLPS_list = []
    val4_list = []
    val5_list = []
    BLPS_keys = []
    start = 0.2
    ini = start
    label_interval = 0.9
    width = 0.5
    interval = 0.1
    end_interval = 0.2
    X_label_position_list = []

    for key in data.keys():
        #if len(data[key]) ==3:
            #print('OK')
        NNPP_list.append(data[key][0])
        OEPP_list.append(data[key][1])
        BLPS_list.append(data[key][2])
        val4_list.append(data[key][3])
        #val5_list.append(data[key][4])
    X1_list = []
    X2_list = []
    X3_list = []
    X4_list = []
    X5_list = []
    for key in data.keys():
        #if len(data[key]) ==3:
        after_start = start
        X1_list.append(start)
        X2_list.append(start+width+interval)
        X3_list.append(start+(width+interval)*2)
        X4_list.append(start+(width+interval)*3)
        #X5_list.append(start+(width+interval)*4)
        after_start += (width+interval)+label_interval 
        for i in range(2):
            after_start += width+interval
        X_label_position_list.append((after_start-start-label_interval)/2+start)
        #if not BLPS:
        #    after_start += width+interval
        start = after_start
        

    plt.bar(X1_list, NNPP_list, color='k', width=width, label='1')
    plt.bar(X2_list, OEPP_list, color='red', width=width,label='2')
    plt.bar(X3_list, BLPS_list, color='blue', width=width, label='3')
    plt.bar(X4_list, val4_list, color='green', width=width, label='4')
    #plt.bar(X5_list, val5_list, color='purple', width=width, label='4')

    #plt.bar( NNPP_list, X1_list,color='k', width=width, label='NNPP')
    #plt.bar( OEPP_list, X2_list,color='red', width=width,label='OEPP')
    #plt.bar( BLPS_list, X3_list,color='blue', width=width, label='BLPS')

    plt.ylabel(label)
    ticklabel=list(data.keys())
    plt.xticks(X_label_position_list, ticklabel,  rotation=0)
    plt.legend(loc='upper center', ncol=5,bbox_to_anchor=(0.5, 1.2))
    plt.xlim(ini-width/2-end_interval, max(X1_list+X2_list+X3_list+X4_list+X5_list)+width/2+end_interval)

if __name__ == '__main__':
    data = {}
    if False:
        data['50-60'] =   [2.5, 63.0, 25.9, 3.7, 4.9 ]
        data['50-60'] =   [ 2.0, 55.0, 21.0, 7.0, 4.0 ]
        data['60-70'] =  [ 0.0, 32.0, 52.0, 4.0, 2.0 ]
        data['70-80'] = [ 1.0, 22.0, 40.0, 18.0, 4.0, ]
        data['80-90'] = [ 0.0, 12.0, 42.0, 22.0, 15.0]
        data['90-100'] =  [ 0.0, 3.0, 31.0, 24.0, 41.0]
    data['50-60'] =   [ 63.0, 25.9, 3.7, 4.9 ]
    data['50-60'] =   [ 55.0, 21.0, 7.0, 4.0 ]
    data['60-70'] =  [ 32.0, 52.0, 4.0, 2.0 ]
    data['70-80'] = [ 22.0, 40.0, 18.0, 4.0, ]
    data['80-90'] = [ 12.0, 42.0, 22.0, 15.0]
    data['90-100'] =  [3.0, 31.0, 24.0, 41.0]

    #for key in data.keys():
    #    total=sum(data[key])
    #    data[key]=[data[key][i]/total for i in range(len(data[key]))]
    for i in range(4):
        total = 0
        for key in data.keys():
            total += data[key][i]
        for key in data.keys():
            data[key][i] = data[key][i]/total
             

    
    # plot eigenvalue MAE
    plt.rc('font', size=40)
    #plt.subplot(1,2,1)
    label = 'Ratio'
    new_data = {}
    for key in data.keys():
            new_data[key] = data[key]
    data = new_data 
    plot(data, label)

    plt.show()
