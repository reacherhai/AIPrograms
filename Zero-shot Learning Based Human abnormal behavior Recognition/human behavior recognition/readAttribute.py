
import pandas as pd
import numpy as np

def read_excel():
    ds = pd.read_csv("./test2.csv",header = None)
    ds = np.array(ds)
    #ds = np.loadtxt(open("test.csv","rb"),skiprows=0)
    return ds
