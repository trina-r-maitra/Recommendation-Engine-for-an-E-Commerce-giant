#===============================================================================
# This is a script to plot the Validation Data
#===============================================================================




import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n=str(15)
listFeatures = [1,5,10,15]

for i, c in enumerate(listFeatures):
    
    markerslist = ["^","v","s","+"]
    n = str(c)
    data1Feature = purchases = pd.read_csv('ValidationData_'+n+'Features', sep = ',')

    
    
    dataForPlot_1 = data1Feature[['RegParam','recall_M1']]
    dataForPlot_2 = data1Feature[['RegParam','recall_M2']]
    

    X1 = dataForPlot_1['RegParam']
    Y1 = dataForPlot_1['recall_M1']
    Y2 = dataForPlot_2['recall_M2']
    
    # Generating the Regularization Plot
    plt.plot(X1, Y1, marker = markerslist[i], label = n+" Features")#, X1, Y2)
    

    plt.ylabel('Average Recall (%)')
    plt.xlabel('Regularization Parameter')
    plt.grid(True)
plt.plot(X1, Y2, marker="o", label = 'Popularity Model')
plt.title('Variation of Average Recall vs Regularization Parameter')
plt.legend(bbox_to_anchor=(0.65, 0.92), loc=2, borderaxespad=0.)
plt.show()


