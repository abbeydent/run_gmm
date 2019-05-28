import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics


dataset = pd.read_csv("dataset.csv", header = None)

print(dataset.head())

plt.scatter(dataset[0], dataset[1])
plt.savefig("scatterplot.png")

# kmeans_predictions = KMeans(n_clusters=3).fit_predict(dataset)
# plt.scatter(dataset[0],dataset[1], c=kmeans_predictions)
# plt.savefig("scatterplot_kmeans_3.png")

kmeans_predictions_4 = KMeans(n_clusters=4).fit_predict(dataset)
plt.scatter(dataset[0], dataset[1], c=kmeans_predictions_4)
plt.savefig("scatterplot_kmeans_4.png")
print("K-mean 4 clusters")
print(metrics.silhouette_score(dataset, kmeans_predictions_4))

kmeans_predictions_3 = KMeans(n_clusters=3).fit_predict(dataset)
plt.scatter(dataset[0], dataset[1], c=kmeans_predictions_3)
plt.savefig("scatterplot_kmeans_3.png")
print("K-mean 3 clusters")
print(metrics.silhouette_score(dataset, kmeans_predictions_3))

kmeans_predictions_2 = KMeans(n_clusters=2).fit_predict(dataset)
plt.scatter(dataset[0], dataset[1], c=kmeans_predictions_2)
plt.savefig("scatterplot_kmeans_2.png")
print("K-mean 2 clusters")
print(metrics.silhouette_score(dataset, kmeans_predictions_2))



# gaussian_predictions = GaussianMixture(n_components=3).fit(dataset).predict(dataset)
# plt.scatter(dataset[0], dataset[1], c=gaussian_predictions)
# plt.savefig("scatterplot_gaussian_3.png")


gaussian_predictions_4 = GaussianMixture(n_components=4).fit(dataset).predict(dataset)
plt.scatter(dataset[0], dataset[1], c=gaussian_predictions_4)
plt.savefig("scatterplot_gaussian_4.png")
print("Gaussian 4 components")
print(metrics.silhouette_score(dataset, gaussian_predictions_4))

gaussian_predictions_3 = GaussianMixture(n_components=3).fit(dataset).predict(dataset)
plt.scatter(dataset[0], dataset[1], c=gaussian_predictions_3)
plt.savefig("scatterplot_gaussian_3.png")
print("Gaussian 3 components")
print(metrics.silhouette_score(dataset, gaussian_predictions_3))

gaussian_predictions_2 = GaussianMixture(n_components=2).fit(dataset).predict(dataset)
plt.scatter(dataset[0], dataset[1], c=gaussian_predictions_2)
plt.savefig("scatterplot_gaussian_2.png")
print("Gaussian 2 components")
print(metrics.silhouette_score(dataset, gaussian_predictions_2))



