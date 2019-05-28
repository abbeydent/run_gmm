import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans


dataset = pd.read_csv("dataset.csv", header = None)

print(dataset.head())

plt.scatter(dataset[0], dataset[1])
plt.savefig("scatterplot.png")

kmeans_predictions = KMeans(n_clusters=3).fit_predict(dataset)
plt.scatter(dataset[0],dataset[1], c=kmeans_predictions)
plt.savefig("scatterplot_kmeans_3.png")