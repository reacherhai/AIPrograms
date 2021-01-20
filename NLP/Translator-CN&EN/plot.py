import matplotlib.pyplot as plt
plt.switch_backend("agg")
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig,ax = plt.subplots()
    loc = ticker.MultipleLocator(base = 0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

