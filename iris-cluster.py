# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
iris = pd.read_csv(r'C:\Users\jemmy\Documents\Skills Bootcamp\Edgehill University\Excel Files\iris.csv', header=None)
print(iris.head(10))
x = iris.iloc[:, [0, 1, 2, 3]]
x = np.array(x)

# Collecting the distortions into list
distortions = []
n_clusters = range(1,5)
for n in n_clusters:
 kmeanModel = KMeans(n_clusters=n)
 kmeanModel.fit(x)
 distortions.append(kmeanModel.inertia_)
# Plotting the distortions
plt.figure(figsize=(16, 8))
plt.plot(n_clusters, distortions, 'bx-')
plt.xlabel('n')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal clusters')
plt.show()

# Define the model
kmeans_model = KMeans(n_clusters=3, random_state=32932)
# Fit into our dataset fit
kmeans_predict = kmeans_model.fit_predict(x)
print(kmeans_predict)

# Visualising the clusters
plt.scatter(x[kmeans_predict == 0, 0], x[kmeans_predict == 0, 1], s=100, c='red', label='Setosa')
plt.scatter(x[kmeans_predict == 1, 0], x[kmeans_predict == 1, 1], s=100, c='blue', label='Versicolour')
plt.scatter(x[kmeans_predict == 2, 0], x[kmeans_predict == 2, 1], s=100, c='green', label='Virginica')
# Plotting the centroids of the clusters
plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], s=100,
c='yellow', label='Centroids')
plt.legend()
plt.show()