import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
from scipy import stats
from statsmodels.multivariate.manova import MANOVA

def hotelling_t2(X1, X2):
    n1, n2 = len(X1), len(X2)
    p = X1.shape[1]
    mean1, mean2 = np.mean(X1, axis=0), np.mean(X2, axis=0)
    S1, S2 = np.cov(X1.T), np.cov(X2.T)
    S_pooled = ((n1-1)*S1 + (n2-1)*S2) / (n1+n2-2)
    t2 = (mean1 - mean2).T.dot(np.linalg.inv(S_pooled)).dot(mean1 - mean2) * (n1*n2)/(n1+n2)
    f = t2 * (n1+n2-p-1) / ((n1+n2-2)*p)
    df1, df2 = p, n1+n2-p-1
    p_value = 1 - stats.f.cdf(f, df1, df2)
    return t2, f, p_value

df = pd.read_csv('/home/fp/eduFocus/jaekyeong/storage/aggregated_data.csv')

features = ['gaze', 'blink', 'emotion']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### 최적 클러스터 수 ###
max_clusters = 5
inertias = []
silhouette_scores = []

for k in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=777)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters: {optimal_clusters}")

### K-means 클러스터링 ###
kmeans = KMeans(n_clusters=optimal_clusters, random_state=777)
df['cluster'] = kmeans.fit_predict(X_scaled)

### 클러스터 중심점 분석 ###
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
print("\nCluster Centers:")
print(cluster_centers_df)

### 클러스터 간 거리 계산 ###
distances = pd.DataFrame(index=range(optimal_clusters), columns=range(optimal_clusters))
for i in range(optimal_clusters):
    for j in range(optimal_clusters):
        distances.iloc[i, j] = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
print("\nCluster Distances:")
print(distances)

### 클러스터 수에 따른 적절한 통계 분석 ###
if optimal_clusters == 2:
    print("\nPerforming Hotelling's T² test:")
    cluster_0 = X_scaled[df['cluster'] == 0]
    cluster_1 = X_scaled[df['cluster'] == 1]
    t2, f, p_value = hotelling_t2(cluster_0, cluster_1)
    print(f"Hotelling's T² statistic: {t2}")
    print(f"F-statistic: {f}")
    print(f"p-value: {p_value}")
elif optimal_clusters == 3:
    print("\nPerforming individual ANOVAs:")
    for feature in features:
        f_value, p_value = stats.f_oneway(*[group[feature].values for name, group in df.groupby('cluster')])
        print(f"{feature}: F-value = {f_value}, p-value = {p_value}")
else:
    print("\nPerforming MANOVA:")
    manova = MANOVA.from_formula('gaze + blink + emotion ~ cluster', data=df)
    print(manova.mv_test())

### Plotly를 사용한 3D 시각화 ###
fig = px.scatter_3d(df, x='gaze', y='blink', z='emotion',
                    color='cluster', 
                    hover_data=['ip_address', 'video_id', 'timestamp'],
                    labels={'cluster': 'Cluster'},
                    title='3D Visualization of Clusters')

fig.update_layout(scene = dict(
                    xaxis_title='Gaze',
                    yaxis_title='Blink',
                    zaxis_title='Emotion'),
                  width=900, height=700,
                  margin=dict(r=20, b=10, l=10, t=10))

### HTML 파일로 저장 ###
fig.write_html("cluster_visualization.html")

print("Analysis complete. 3D visualization saved as 'cluster_visualization.html'")