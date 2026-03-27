import torch
import gc
import os
import math
import argparse
import pickle

import numpy as np
import pandas as pd
import networkx as nx

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from datetime import timedelta

from sitif_data import load_pickle_custom

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class SimMatrixC:
    """
    Class Responsible for the similarity matrix and all of the calculation connected.

    This class is designed for fake news spreader detection tasks, as outlined in the paper:
    "Identifying Misinformation Spreaders: Challenges and Model Architectures".
    It constructs a UUSG using createUBERT_df_clustering. If the temporal architecture
    is needed to be omitted, please use createUBERT_df_averaging then.

    Args:
        tweets_df - DataFrame containing tweets infromation and content (pd.DataFrame)
        indices - list of user IDs to include (list)
        device - computation device (str)
        threshold - similarity threshold $\Pho$ for connections (float)
        emb_d - dimension of S-BERT embeddings (int)
        top_sim - top $k$ similar items to consider (int)
        max_length - max token length for S-BERT input (int)
        max_days - max temporal window for similarity calculations (int)
        max_interval - max interval between events to consider (int)
        clustering - if True, follows the clustering described in paper, averaging otherwise (bool)
    """
    def __init__(
            self, 
            tweets_df: pd.DataFrame, 
            indices: list, 
            device: str = "cuda", 
            threshold: float = 0.5, 
            emb_d: int = 384, 
            top_sim: int = 40,
            max_length: int = 128, 
            max_days: int = 7, 
            max_interval: int = 1,
            clustering: bool = True
        ):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

        tweets_df = tweets_df.sort_values(by="created")
        self.tweets_df = tweets_df
        self.indices = indices
        self.emb_d = emb_d
        self.max_length = max_length
        self.max_days = max_days
        self.max_interval = max_interval
        self.threshold = threshold
        self.top_sim = top_sim
        self.clustering = clustering

    def __len__(self):
        return len(self.indices)
    
    def compute_SBERT(self, text_vector):
        """
        Computes SBERT of a user.
        """
        encoded_inputs = self.tokenizer(text_vector, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
        input_ids = encoded_inputs["input_ids"].to(self.device)
        attention_mask = encoded_inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            # Obtain hidden states from the model
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # The embeddings are usually taken as the mean of the last hidden state
            embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding.cpu()

    def createUBERT_df_averaging(self):
        """
        Averages over SBERTs of a user to compute its vector representation as a mean of its tweets.
        
        Args: - implicit
            tweets_df (pandas.DataFrame)
            unique_user_ids (np.ndarray)
            emb_d (int)
        
        Returns:
            np.ndarray: A 2D numpy array of shape (num_users, emb_d) containing a vector representation of a user.
        """
        tweets_gb_user = self.tweets_df.groupby("user_id")[["text","created"]]

        user_embeddings = np.zeros((len(self.indices),  self.emb_d), dtype=np.float32)
    
        idx = 0
        for user_id, data in tqdm(tweets_gb_user):
            text = data["text"]
            if user_id in self.indices:
                embedding = self.compute_SBERT(list(text))
                user_embeddings[idx, :] = embedding.mean(dim=0).numpy()
                idx += 1

                if not (idx%2000):
                        free, total = torch.cuda.mem_get_info(torch.device('cuda:0'))
                        mem_used_MB = (total - free) / 1024 ** 2
                        torch.cuda.empty_cache()
                        gc.collect()

        return user_embeddings  

    def createUBERT_df_clustering(self):
        """
        Time-window clustering over SBERTs of a user to compute its vector representation as a mean of those clusters.
        
        Args: - implicit
            tweets_df (pandas.DataFrame)
            unique_user_ids (np.ndarray)
            emb_d (int)
        
        Returns:
            np.ndarray: A 2D numpy array of shape (num_users, emb_d) containing a vector representation of a user.
        """

        tweets_gb_user = self.tweets_df.groupby("user_id")[["text","created"]]
        final_tweet = self.tweets_df.iloc[-1]['created']
        init_tweet = self.tweets_df.iloc[0]['created']
        normalization_con = final_tweet-init_tweet

        user_embeddings = np.zeros((len(self.indices), self.emb_d), dtype=np.float32)
        
    
        idx = 0
        for user_id, data in tqdm(tweets_gb_user):

            if user_id in self.indices:
                text = data["text"]
                dates = list(data["created"])

                clusters = [0]
                current_cluster = 1
                current_date = dates[0]

                for date in dates[1:]:
                    # if -1, then w.r.t. to the last. If 0, then w.r.t. to the first, then the clusters will be of len: 1 day,
                    # but not the distance between them to be 1 day
                    if (date - current_date) <= timedelta(days=self.max_interval) and ((current_cluster - clusters[-1]) < self.max_days):
                        current_date = date
                    else:
                        clusters.append(current_cluster)
                        current_date = date
                    current_cluster += 1

                # if last element was not appended, we do in manually:

                if clusters[-1] != len(dates):
                    clusters.append(len(dates))


                # Cluster is calculated, hence I look over SBERT of each cluster seperately

                prev_cl = 0
                embedding = self.compute_SBERT(list(text))
                n_cnst = 0

                for i in clusters[1:]:
                    # time-dependent normalization
                    time_scaler_log_lin = (dates[i-1]-init_tweet)/normalization_con
                    time_scaler_log = 0.6 + 0.4 * (math.log(time_scaler_log_lin+1,2) / math.log(2,2))

                    # cluster-size-dependent normalization
                    cluster_size = (i-prev_cl)
                    normalization_scaler = math.log((1+cluster_size),2)*time_scaler_log
                    user_embeddings[idx, :] += (embedding[prev_cl:i].mean(dim=0).numpy())*normalization_scaler

                    n_cnst += normalization_scaler # acts as normalizing constant
                    
                    prev_cl = i
                
                user_embeddings[idx, :] /= n_cnst

                # added as pc stores cache and slows this function after 50% drastically
                
                if not (idx%2000):
                    free, total = torch.cuda.mem_get_info(torch.device('cuda:0'))
                    mem_used_MB = (total - free) / 1024 ** 2
                    torch.cuda.empty_cache()
                    gc.collect()

                idx += 1

        return user_embeddings  

    def normalizeSimMatrix(self, simMatrix):
        """
        Normalizes similarity matrix accordin to the steps describe in the paper.
        
        Args:
            simMatrix - A 2D numpy array of shape (num_users, num_users) containing pairwise cosine similarities (pandas.DataFrame) 
        
        Returns:
            pandas.DataFrame: Normalized similarity matrix. (in-place changes)
        """
        return (simMatrix >= self.threshold).astype(int)

    def normalizeSimMatrixKtop(self, simMatrix):
        """
        Normalizes similarity matrix. Takes top 100 of all similarities and applies the threshold.
        It is done, as the users are heavily clusteres and some users have none similar user, while
        Others have more than a thousand
        
        Args:
            simMatrix - A 2D numpy array of shape (num_users, num_users) containing pairwise cosine similarities (pandas.DataFrame) 
        
        Returns:
            pandas.DataFrame: Normalized similarity matrix. (in-place changes)
        """

        # simMatrix = (simMatrix >= threshold)
        simMatrix = simMatrix.apply(lambda row: ((row >= self.threshold) & row.isin(row.nlargest(self.top_sim))), axis=1).astype(int)

        return simMatrix


    def createSimMatrixWithBERT(self):
        """
        Compute a similarity matrix based on S-BERT of users (based on U-BERT).
        
        Returns:
            np.ndarray: A 2D numpy array of shape (num_users, num_users) containing pairwise cosine similarities.
        """
        if self.clustering:
            print("generating UBERT user embeddings via clustering, provided in the paper...")
            user_embeddings = self.createUBERT_df_clustering()
        else:
            print("averaging, simplified non-temporal version...")
            user_embeddings = self.createUBERT_df_averaging()
        print("Shape of user embeddings (not matrix yet):",user_embeddings.shape)

        similarity_matrix = cosine_similarity(user_embeddings)
        simMatrix = pd.DataFrame(similarity_matrix, index=self.indices, columns=self.indices)

        simMatrix =  self.normalizeSimMatrixKtop(simMatrix)
        return simMatrix

    def createSimCosScores(self, user_embeddings = None):
        """
        Compute a similarity cosine scores. Is used for adjusting the sim matrix.
        
        Args:
            user_embeddings - A 2D numpy array of shape (num_users, emb_d) containing a vector representation of a user (np.ndarray)
        
        Returns:
            np.ndarray: A 2D numpy array of shape (num_users, num_users) containing pairwise cosine similarities.
        """
        
        if not user_embeddings:
            if self.clustering:
                print("generating UBERT user embeddings via clustering, provided in the paper...")
                user_embeddings = self.createUBERT_df_clustering()
            else:
                print("generating UBERT user embeddings via averaging, simplified non-temporal version...")
                user_embeddings = self.createUBERT_df_averaging()

        print("Shape of user embeddings (not matrix yet):",user_embeddings.shape)

        similarity_matrix = cosine_similarity(user_embeddings)
        simMatrix = pd.DataFrame(similarity_matrix, index=self.indices, columns=self.indices)

        return simMatrix



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute user similarity matrices")

    # Data paths
    parser.add_argument("--gpickle_users", type=str, default="data\\gg_users_full.gpickle",  help="Path to user interaction graph (UUIG)")
    parser.add_argument("--tweets_df", type=str, default="data\\df_tweets_full.pickle", help="Path to DataFrame containing tweets information")
    parser.add_argument("--path", type=str, default="data", help="Path where to save newly created similarity model")

    # Optional parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device to run SBERT computations")
    parser.add_argument("--threshold", type=float, default=0.67, help="Similarity threshold $\Pho$ for edges")
    parser.add_argument("--emb_d", type=int, default=384, help="dimension of S-BERT embeddings")
    parser.add_argument("--top_sim", type=int, default=150, help="Top $k$ similar items to consider")
    parser.add_argument("--max_length", type=int, default=128, help="Max token length for S-BERT input")
    parser.add_argument("--max_days", type=int, default=7, help="Max temporal window for similarity calculations")
    parser.add_argument("--max_interval", type=int, default=1, help="Max interval between events to consider")
    parser.add_argument("--averaging", action="store_false", dest="clustering", help="Use averaging approach (default: clustering)")

    args = parser.parse_args()

    # ===== Load inputs =====
    tweets_df = load_pickle_custom(args.tweets_df)
    gpickle_users = load_pickle_custom(args.gpickle_users)

    # ===== Compute similarity =====
    simMCommon = SimMatrixC(
        tweets_df=tweets_df,
        indices=list(gpickle_users.nodes),
        device=args.device,
        threshold=args.threshold,
        emb_d=args.emb_d,
        top_sim=args.top_sim,
        max_length=args.max_length,
        max_days=args.max_days,
        max_interval=args.max_interval,
        clustering=args.clustering
    )

    simCosScores = simMCommon.createSimCosScores()
    simMatrix = simMCommon.normalizeSimMatrixKtop(simCosScores)

    # ===== Similarity model save =====
    simMatrix_nx = nx.DiGraph(simMatrix)
    sim_model_name = f"simMatrix_top{simMCommon.top_sim}_th{int(simMCommon.threshold*100)}.gpickle"
    final_path = os.path.join(args.path, sim_model_name)
    with open(final_path, 'wb') as f:
        pickle.dump(simMatrix_nx, f, pickle.HIGHEST_PROTOCOL)
