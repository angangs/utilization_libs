import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt


def read_dataframe(filename, columns=None, index_col=None):
    # Read dataframe
    df = pd.read_csv(filename, index_col=index_col, encoding='utf-8-sig', sep=';')
    if columns is None:
        return df
    else:
        return df[columns]


def k_means(data, max_no_clusters, weights=None):
    spatial = KMeans(n_clusters=max_no_clusters)
    spatial.fit_predict(data, sample_weight=weights)
    cluster_labels = spatial.labels_
    centers = spatial.cluster_centers_
    return cluster_labels, centers


def performance_measurements(data, cluster_labels):
    return metrics.silhouette_score(data, cluster_labels, metric='euclidean'), \
           metrics.calinski_harabasz_score(data, cluster_labels), \
           metrics.davies_bouldin_score(data, cluster_labels)


def plot_elbow_graph(data, max_no_clusters):
    nc = range(1, max_no_clusters)
    kmeans = [KMeans(n_clusters=i) for i in nc]
    score = [kmeans[i].fit(data).score(data) for i in
             range(len(kmeans))]

    plt.plot(nc, score)
    plt.xlabel('Number of Clusters')
    plt.xticks(list(range(max_no_clusters + 1)))
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.grid(axis='x', c='g', linestyle='--')
    plt.show()


def clustering_dataframe(df, no_clusters=5, measure_performance=False, n_clusters_elbow=None):
    # Clustering dataframe
    silhouette_score, calinski_harabasz_score, davies_bouldin_score = None, None, None
    cluster_labels, centers = k_means(df, no_clusters)

    if measure_performance:
        silhouette_score, calinski_harabasz_score, davies_bouldin_score = performance_measurements(df, cluster_labels)

        if n_clusters_elbow:
            plot_elbow_graph(df, n_clusters_elbow)

    df['Label'] = cluster_labels

    return df, silhouette_score, calinski_harabasz_score, davies_bouldin_score


def get_stats_clustering_dataframe(df, stats_keys):
    sd_sum = {}
    sd_mean = {}
    sd_std = {}
    sd_count = {}
    for k in stats_keys:
        sd_sum[k] = 'sum'
        sd_mean[k] = 'mean'
        sd_std[k] = 'std'
        sd_count[k] = 'count'

    # Stats for clusters
    df_stasts_sum = df.groupby(by='Label').agg(sd_sum)
    df_stasts_mean = df.groupby(by='Label').agg(sd_mean)
    df_stasts_std = df.groupby(by='Label').agg(sd_std)
    df_stasts_mean_to_std = df_stasts_mean / df_stasts_std
    df_stasts_cnt = df.groupby(by='Label').agg(sd_count)
    return df_stasts_sum, df_stasts_cnt, df_stasts_mean, df_stasts_std, df_stasts_mean_to_std


def feature_importance(df):
    from lightgbm import LGBMClassifier
    lgbmc = LGBMClassifier()
    df.dropna(subset=['Label'], inplace=True)
    lgbmc.fit(df.iloc[:, :-1], df.iloc[:, -1])
    return lgbmc.feature_importances_
