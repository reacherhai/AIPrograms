import pandas as pd
import numpy as np

def getData(name):
    ds = pd.read_csv(name,header=None)
    ds = ds.as_matrix()

    df = pd.DataFrame()

    train_labels = []

    for i in range(len(df)):
        list  = ds[i][0].split()
        if(int( list[1]) == 0):
            continue
        #print(list)
        dict = {}
        for j in range(4,55):
            if(list[j-1]!='NaN'):
                list[j-1] = float( list[j-1] )
            dict[j] = list[j-1]
        train_labels.append( int(list[1]))
        new =  pd.DataFrame(dict,index = [1])
        df = df.append(new,ignore_index=True)

    #print(train_labels)
    ns = df.as_matrix()

    def to_one_hot(labels,dimension = 25):
        results = np.zeros((len(labels),dimension))
        for i,label in enumerate(labels):
            results[i,label] = 1.
        return results

    test  = np.array(train_labels)

    #one_hot_train_labels = to_one_hot(train_labels)
    return ns,test


ns, labels = getData('./subject106.dat')
#print(labels)