import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm

class ClusterRFM():

    def __init__(self, df, user_id:str, frequency_column:str, monetary_column:str, n_clusters=None,
                trained_model=None, trained_scaled_model=None):
        self.df = df
        self.user_id = user_id
        self.frequency_column = frequency_column
        self.monetary_column = monetary_column
        self.n_clusters = n_clusters
        self.trained_model = trained_model
        self.trained_scaled_model = trained_scaled_model
        self.scaler = None
        self.grouped_df = None

    def group(self):
        self.grouped_df = self.df.groupby(self.user_id).sum()
        return self.grouped_df

    def gen_rfm_features(self):
        df_features = self.group()
        df_features['avg_tk'] = df_features[self.monetary_column] / df_features[self.frequency_column]

        df_features['freq_rank'] = df_features[self.frequency_column].rank(ascending=True)
        df_features['mon_rank'] = df_features[self.monetary_column].rank(ascending=True)
        df_features['avg_rank'] = df_features['avg_tk'].rank(ascending=True)

        df_features['freq_rank_norm'] = (df_features['freq_rank'] / df_features['freq_rank'].max())*100
        df_features['mon_rank_norm'] = (df_features['mon_rank'] / df_features['mon_rank'].max())*100
        df_features['avg_rank_norm'] = (df_features['avg_rank'] / df_features['avg_rank'].max())*100

        df_features = df_features[['freq_rank_norm', 'mon_rank_norm', 'avg_rank_norm']]

        return df_features

    def scaling(self):
        rfm_df = self.gen_rfm_features()

        if self.trained_scaled_model == None:
            scaler = MinMaxScaler()
            scaler.fit(rfm_df)
            self.scaler = scaler
        else:
            scaler = self.trained_scaled_model

        df_scaled = scaler.transform(rfm_df)
        return df_scaled

    def calculate_clusters(self):

        # Calcular numero de clusters
        df_scaled = self.scaling()

        scores = []
        range_n_clusters = list(range(2,6))
        print(df_scaled.shape)
        print(df_scaled)
        print('Calculando Número de Clusters')
        for n in tqdm(range_n_clusters):
            clusterer = KMeans(n_clusters=n, init='random', random_state=20220720)
            cluster_labels = clusterer.fit_predict(df_scaled)
            silhouette_avg = silhouette_score(df_scaled, cluster_labels)
            scores.append(silhouette_avg)

        self.n_clusters = np.argmax(scores) + 2

    def clustering(self):

        df_scaled = self.scaling()

        if (self.n_clusters == None) and (self.trained_model == None):
            self.calculate_clusters()

        if self.trained_model == None:
            km = KMeans(n_clusters=self.n_clusters, init='random', random_state=20220720)
            km = km.fit(df_scaled)
            self.trained_model = km
            self.grouped_df['cluster'] = km.labels_
        else:
            km = self.trained_model
            self.grouped_df['cluster'] = km.predict(df_scaled)

        return self.grouped_df

    def cluster_customer(self):

        try:
            return self.grouped_df['cluster'].to_dict()
        except:
            df = self.clustering()
            return df['cluster'].to_dict()
