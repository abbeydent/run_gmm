import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv("dataset.csv", header = None)

print(dataset.head())

plt.scatter(dataset[0], dataset[1])
plt.savefig("scatterplot.png")