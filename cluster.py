from sklearn.cluster import KMeans
from user import load_users, User


def cluster(users, n_clusters=7):
    labels = KMeans(n_clusters=n_clusters).fit_predict([tuple(user.genre_avgs_prefilled.values()) for user in users])

    clusters = [[] for i in range(n_clusters)]
    for n, label in enumerate(labels):
        clusters[label].append(users[n])

    for cluster in clusters:
        for user in cluster:
            user.cluster = cluster

    return clusters


if __name__ == '__main__':
    users = load_users()
    clusters = cluster(users)

    print(users[0].genre_avgs)
    print(clusters)
    print(users[0].recalculate_cluster_score().Cluster_score)