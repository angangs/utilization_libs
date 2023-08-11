import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt


class ClusteringMapsService:
    def __init__(self, df, weights=None):
        self.coordinates = df[['Longitude', 'Latitude']].copy()
        self.weights = weights
        self.centers = None
        self.cluster_labels = None

    def modified_k_mean(self, max_no_clusters, max_dist, weights=None):
        max_dist *= 0.01
        cluster_range = min(self.coordinates.shape[0], max_no_clusters)
        cluster_solution = {}
        for cluster in range(1, cluster_range + 1):
            spatial = KMeans(n_clusters=cluster)
            spatial.fit_predict(self.coordinates, sample_weight=weights)
            self.coordinates['cluster_labels'] = cluster_labels = spatial.labels_
            centers = spatial.cluster_centers_

            cl = None
            max_dist_list = []
            for cl in range(cluster + 1):
                lat_lon_cluster = self.coordinates[self.coordinates['cluster_labels'] == cl][[
                    'Latitude', 'Longitude']].values
                try:
                    d = squareform(pdist(lat_lon_cluster, metric='euclidean'))
                except MemoryError as e:
                    print(e)
                    spatial = KMeans(n_clusters=max_no_clusters)
                    spatial.fit_predict(self.coordinates)
                    cluster_labels = spatial.labels_
                    centers = spatial.cluster_centers_
                    return max_no_clusters, spatial, cluster_labels, centers

                max_dist_list.append(np.nanmax(d))
                if np.nanmax(d) > max_dist:
                    break

            cluster_solution[cluster] = {}
            cluster_solution[cluster]['solution'] = {'spatial': spatial, 'centers': centers}
            cluster_solution[cluster]['max_distance'] = max(max_dist_list)

            if cl == cluster:
                return cluster, spatial, cluster_labels, centers

        sorted_solutions = {k: v for k, v in sorted(cluster_solution.items(), key=lambda item: item[1]['max_distance'])}
        best_cluster = list(sorted_solutions.keys())[0]

        return best_cluster, cluster_solution[best_cluster]['solution']['spatial'], cluster_solution[best_cluster][
            'solution']['spatial'].labels_, cluster_solution[best_cluster]['solution']['centers']

    def k_means(self, max_no_clusters):
        spatial = KMeans(n_clusters=max_no_clusters)
        spatial.fit_predict(self.coordinates, sample_weight=self.weights)
        self.cluster_labels = spatial.labels_
        self.centers = spatial.cluster_centers_

    def performance_measurements(self):
        return metrics.silhouette_score(self.coordinates, self.cluster_labels, metric='euclidean'),\
               metrics.calinski_harabasz_score(self.coordinates, self.cluster_labels),\
               metrics.davies_bouldin_score(self.coordinates, self.cluster_labels)

    def plot_elbow_graph(self, max_no_clusters):
        nc = range(1, max_no_clusters)
        kmeans = [KMeans(n_clusters=i) for i in nc]
        score = [kmeans[i].fit(self.coordinates).score(self.coordinates) for i in
                 range(len(kmeans))]

        plt.plot(nc, score)
        plt.xlabel('Number of Clusters')
        plt.xticks(list(range(max_no_clusters + 1)))
        plt.ylabel('Score')
        plt.title('Elbow Curve')
        plt.grid(axis='x', c='g', linestyle='--')
        plt.show()
