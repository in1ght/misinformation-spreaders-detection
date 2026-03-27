import torch
import networkx
import pickle
import os
import argparse

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from collections import deque
from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Dataset + Data Loader
class FNSDDataset(torch.utils.data.Dataset): # 
    """
    FNSDDataset: Fake News Spreaders Detection Dataset. Later, for clarity Fake News was changed to Misinformation.
    Do not confuse it with "Functional Neurological Symptom Disorder Dataset", which is also named FNSDDataset.

    This dataset class is designed for fake news spreader detection tasks, as outlined in the paper:
    "Identifying Misinformation Spreaders: Challenges and Model Architectures".
    
    This Dataset supports configurable input parts, making it suitable for full or ablation-based experiments.
    
    Args:
        usern_nx - user-user-interaction graph with metadata such as missiformation spread (networkx.DiGraph)
        tweet_nx - tweet-tweet-interaction graph (networkx.DiGraph)
        user_sim - user-user-similairty binary graph created by class SimMatrixC (networkx.DiGraph)
        user_features - user features, that are used as separate class to the model (pd.DataFrame)
        ti_X - tweet X that is used for the GCN (pd.DataFrame)
        hops_ui - hops (depth) that is used in GCN for user-user interaction GCN (int)
        hops_us - hops (depth) that is used in GCN for user-user similarity GCN (int)
        hops_ti - hops (depth) that is used in GCN for tweet-tweet-interaction GCN (int)
        threshold - set the thresholds to split the data. Common is 0,5 and binary,
        if 2 are given, then we have a case of [non-spreaders; potential spreaders; spreaders](list (multiclass) or float (binary))
        users_given - a set that sets which users are used for the dataset (set)
        ui_metadata - a metadata information for threshold, which is used if usern_nx is not given (networkx.DiGraph)
        ui_X - user-user-interaction X that is used for the GCN (pd.DataFrame)
        us_X - user-user-dimilarity X that is used for the GCN (usually is the same as ui_X) (pd.DataFrame)
        parts - parts that model will use, if ablation study is performed or if the complexity is cut (dict)

    Returns:
        dict
        For each user, returns a dictionary containing:
        - Label (1 or -1)
        - Graph adjacency matrices (UI, US, TI) if enabled
        - Corresponding feature matrices
        - Number of original tweets associated with the user

    """
    def __init__(
            self,
            usern_nx: networkx.DiGraph,
            tweet_nx: networkx.DiGraph,
            user_sim: networkx.DiGraph,
            user_features: pd.DataFrame,
            ti_X: pd.DataFrame,
            hops_ui: int = 3, 
            hops_us: int = 1, 
            hops_ti: int = 1,
            threshold: list = [0.5], 
            users_given: set = None ,
            ui_metadata: networkx.DiGraph = None, 
            ui_X: pd.DataFrame = None, 
            us_X: pd.DataFrame = None,
            parts: dict = {"user_inter": True,"user_sim": True,"tweets_inter": True,"user_feat": True}
        ):
        if ui_metadata:
            self.ui_metadata = ui_metadata
        else:
            self.ui_metadata = usern_nx
        
        if parts['user_inter']: 
            self.uu_graph = usern_nx
            self.hops_ui = hops_ui

        self.u_features = user_features.sort_index()

        self.ui_X = ui_X if ui_X is not None else self.u_features # Optional, if it is assumed that ui_X from get_item
        self.us_X = us_X if us_X is not None else self.u_features # Is different for the GCN. 

        self.part_us = parts['user_sim']
        self.part_ti = parts['tweets_inter']
        self.part_ui = parts['user_inter']
        self.part_uf = parts['user_feat']

        if parts['tweets_inter']: 
            self.tt_graph = tweet_nx
            self.t_features = ti_X
            self.hops_ti = hops_ti
        
        if parts['user_sim']:
            self.uu_graph_sim = user_sim
            self.hops_us = hops_us
            

        # countermeasure to nodes with out_degree == 0, as we want to avoid them
        # it is done mostly due to split of graph, if it is splitted
        isolated_nodes = {n for n in self.ui_metadata.nodes() if self.ui_metadata.out_degree(n) == 0}
        # nodes that will not cause any error further and are required
        valid_nodes = {
            node for node in self.u_features.index
            if node in self.ui_metadata.nodes and node in users_given and node not in isolated_nodes}
        valid_nodes = sorted(valid_nodes)
        if isinstance(threshold,float):
            # Binary classification {-1, 1}
            self.labels = pd.Series({
                            node: 1 if (self.ui_metadata.nodes[node]['score_graph']>threshold) else -1
                            for node in valid_nodes})
        else:
            # Multiclass classification
            # or ninary classification {0, 1}
            s_thresholds = sorted(threshold)

            def get_class(score):
                for enum, tr in enumerate(s_thresholds): # 0.3 0.7
                    if score <= tr:
                        return enum
                return len(s_thresholds)

            self.labels = pd.Series({
                node: get_class(self.ui_metadata.nodes[node]['score_graph'])
                for node in valid_nodes})


        if parts['user_inter']:
            self.user_A_hat, self.encode_user = self.adjacency_A_hat(self.uu_graph)
            self.user_ADA_05 = self.degree_D_hat(self.user_A_hat)
            self.user_ADA_05 = self.multiplication_GCN(self.user_A_hat,self.user_ADA_05)

        if parts['user_sim']:
            self.user_sim_A_hat, self.encode_user_sim = self.adjacency_A_hat(self.uu_graph_sim)
            self.user_sim_ADA_05 = self.degree_D_hat(self.user_sim_A_hat)
            self.user_sim_ADA_05 = self.multiplication_GCN(self.user_sim_A_hat,self.user_sim_ADA_05)

        if parts['tweets_inter']: 
            self.tweet_A_hat, self.encode_tweet = self.adjacency_A_hat(self.tt_graph)
            self.tweet_ADA_05 = self.degree_D_hat(self.tweet_A_hat)
            self.tweet_ADA_05 = self.multiplication_GCN(self.tweet_A_hat,self.tweet_ADA_05)

            # init all tweets
            self.tweets_wrt_u = None
            self.get_user_tweets()

        # id user -> user -> (tweets)

        # GCN acts as sliding windows on whole images to learn features from neighboring cells.
        # major difference between GCN and CNN is that it is developed to work on non-euclidean 
        # data structures where the order of nodes and edges can vary

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        # 3 hops - > 1 matrix
        actual_idx = self.labels.index[idx]
        label = self.labels.iloc[idx]

        fnsd = {'label': label}

        # User
        if self.part_ui:
            neighbors, neighbors_og = self.get_n_hops_neighbors(self.encode_user,self.uu_graph,actual_idx, self.hops_ui)  #  GCN((A*D^0.5*A)*(X)* (W + b))
            user_X = self.ui_X.loc[neighbors_og].values # features for the part of features
            user_D = self.get_rowcol_from_list(neighbors,self.user_ADA_05)
            fnsd['ui'] = user_D
            fnsd['ui_f'] = user_X

        if self.part_uf:
            user_features = self.u_features.loc[actual_idx].values
            fnsd['user_features'] = user_features
        
        
        if self.part_us:
            # # User similarity graph # Start
            # # Old option via explicit cosine values
            # top_sim_ilocs = np.argpartition(shorty.loc[actual_idx], -self.top_sim)[-self.top_sim:]
            # # explicitely done, since iloc works differently then np[]
            # user_D_sim = np.array([
            #     [self.uu_graph_sim.iloc[row, col] for col in top_sim_ilocs] for row in top_sim_ilocs
            # ])
            # user_X_sim = self.u_features.iloc[top_sim_ilocs].values
            # # Newer version
            neighbors, neighbors_og = self.get_n_hops_neighbors(self.encode_user_sim,self.uu_graph_sim,actual_idx,self.hops_us)
            user_X_sim = self.us_X.loc[neighbors_og].values
            user_D_sim = self.get_rowcol_from_list(neighbors,self.user_sim_ADA_05)
            # # User similarity graph # End
            fnsd['us'] = user_D_sim
            fnsd['us_f'] = user_X_sim

        if self.part_ti:
            # Tweet 
            # So current split gives me elements with out_degree == 0, I am not sure if I should try to work with those elements
            # or simply delete them, for now I will save them.
            tweets_list_og_user = self.tweets_wrt_u[actual_idx]
            tweets_list, tweets_list_og = self.get_n_hops_neighbors(self.encode_tweet,self.tt_graph,tweets_list_og_user,self.hops_ti)
            tweet_D = self.get_rowcol_from_list(tweets_list,self.tweet_ADA_05)
            tweet_X =  self.t_features.loc[tweets_list_og].values
            fnsd['ti'] = tweet_D
            fnsd['ti_f'] = tweet_X
            fnsd['tweets_l'] = len(tweets_list_og_user)
        
        return fnsd
        # return user_D, user_D_sim, tweet_D, user_X, user_X_sim, tweet_X, len(tweets_list_og_user), label
        # # UI graph, US graph, TI graph, UI features, US features, TI features, len(OG_user_tweets), Labels

    # HELPERS

    def adjacency_A_hat(self,graph): 
        """
        Constructs the normalized adjacency matrix \hat{A} = A + I.

        From the paper:
        The normalized adjacency matrix is defined as $\hat{A} = A + I$,
        $I$ represents the identity matrix and $A$ represents the adjacency user 
        matrix derived from the user interaction graph (or UUIG, UUSG or TTIG)

        Args:
            graph - graph structure (either UUIG, UUSG or TTIG) (networkx.DiGraph)

        Returns:
            Tuple[csr_matrix, dict]:
                - normalized adjacency matrix with self-loops
                - Encoder from node IDs to matrix indices
        """
        # Could be rewritten as numpy, where it is of shape (len(nodes), len(nodes)),
        # But it would store all 0 as well, assuming our data is sparse, it is not memmory efficient
        nodes = list(graph.nodes)
        nodes.sort()
        nodes_id = {v: i for i, v in enumerate(nodes)}  # reverse mapping of ids
        edges = set()
        for n in nodes:
            edges.add((nodes_id[n], nodes_id[n])) # cause A^ = A + I
            for _, t in graph.edges(n):
                edges.add((nodes_id[n], nodes_id[t]))
        row, col = tuple(zip(*edges))
        row = np.asarray(row, dtype=np.int32)
        col = np.asarray(col, dtype=np.int32)
        return csr_matrix((np.ones_like(row), (row, col)), shape=(len(nodes), len(nodes))), nodes_id

    def degree_D_hat(self,graph):
        """
        Computes the normalized degree matrix (\hat{D}) for a graph.

        From the paper:
        $\hat{D}_{ii} = \sum_j \hat{A}_ij$, where $f$ is the activation function (e.g.,  ReLU), 
        $W$ is the trainable weight matrix, $b$ is the trainable bias vector, 
        and $X$ represents the user feature matrix

        Args:
            graph - adjacency matrix (sparse or dense)

        Returns:
            csr_matrix:
                - a diagonal degree matrix in sparse format
        """
        degrees =   (np.array(graph.sum(axis=1))**(-1/2)).flatten()
        # diagonal matrix with the degrees
        d_hat = csr_matrix((degrees, (np.arange(len(degrees)), np.arange(len(degrees)))), 
                    shape=(graph.shape[0], graph.shape[0]))
        return d_hat
    
    def multiplication_GCN(self,a,d):
        return d @ a @ d    # @ x :No x because we need to include 3-hop neighbors, hence
                            # we do it in get_item
    
    def get_n_hops_neighbors(self,encoder,graph,start_idx, hops):
        """
        Retrieves neighbors up to N hops away from a starting node or set of nodes.

        Args:
            encoder - mapping to encode node IDs (dict-like)
            graph - graph structure (either UUIG, UUSG or TTIG) (networkx.DiGraph)
            start_idx - starting node ID or set of node IDs (int or set)
            hops - maximum hops to traverse (int)

        Returns:
            Tuple[list, list]:
                - list of encoded node IDs
                - list of original node IDs
        """
         # 300 -> 1 hop -> 9000 # additional data - so keep hops relatively low
        if isinstance(start_idx, set):
            queue = deque([(idx, 0) for idx in start_idx])
        else:
            queue = deque([(start_idx, 0)])
        neighbors = set()
        nodes = list()
        while queue:
            node = queue.popleft()
            if node[0] not in neighbors:
                neighbors.add(node[0])
                nodes.append(node[0])
                if node[1] < hops:
                    for _, neighbor in graph.edges(node[0]):
                        queue.append((neighbor,(node[1]+1)))
        return list(map(encoder.get, nodes)), nodes
                                                    # It might need non-encoded ids
                                                    # so it might be better to return both encoded and not
                                                    # to avoid decoding it again
    
    def get_user_tweets(self):
        """
        Collects and aggregates tweets associated with each user. In-place for the class.
        Gathers all tweets related to each user, including implicitly associated with mentions, 
        replies and parents.

        Args: - implicit
            - self.ui_metadata and self.labels
        """
        self.tweets_wrt_u = {}
        # Creates a list of tweets for every user user
        for user_id in tqdm(self.labels.index, disable=True):
            # If user has any tweets, might not be the case actually
            # So user has tweets and edges. So I assume all tweets with mentions or replies
            # are not considered as tweets of a node, but as edges of a node?
            tweets = list(self.ui_metadata.nodes[user_id].get('tweets', []))
            edges = [] # might be duplicates, because some edges might correspond to the same id?
            for edge in self.ui_metadata.edges(user_id):
                edge_tweet = self.ui_metadata.edges[edge]
                # So we can go up the tweet tree, down and it can mentioned
                edges.extend(zip(edge_tweet["parent"], edge_tweet["parent_date"]) if "parent" 
                             in self.ui_metadata.edges[edge] else [])
                # Down i.e. Reply
                edges.extend(zip(edge_tweet["replies"], edge_tweet["replies_date"]) if "replies" 
                             in self.ui_metadata.edges[edge] else [])
                # Left i.e. Mention
                # Mention if it is the initial user
                edges.extend(zip(edge_tweet["mentions"], edge_tweet["mentions_date"]) if "mentions" 
                             in self.ui_metadata.edges[edge] else [])
                # Could there be duplicates? I am not sure. Fow now omiited.
            tweets.extend(edges)
            tweets.sort(key=lambda x: x[1], reverse=True)
            # only ids, datetime could be used in sorting, so that 
            # graph will find a correlation w.r.t.
            # Store only the first element of each tuple
            self.tweets_wrt_u[user_id] = set([x[0] for x in tweets])
        # 270 000 -> BERT -> 270 000 * 300 = 90 000 000 cells 8 bit 

    def get_rowcol_from_list(self,list_ids,ada_data):
        """
        Generates all pairwise index combinations from list_ids into a square matrix.

        Args:
            list_ids - list of indices (list)
            ada_data - A*D*A (e.g., interaction or similarity matrix)

        Returns:
            np.ndarray:
                - square matrix of shape (len(list_ids), len(list_ids))
        """
        np_ids = np.asarray(list_ids)
        ids_1, ids_2 = np_ids.repeat(np_ids.shape[0]), np.hstack([np_ids] * np_ids.shape[0]) 
        # so now we will have   11 12 13 - 1st row ( for [1,2,3] case )
        # then we will have     21 22 23 - 
        # then                  31 32 33 - 
        matrix_D = np.array(ada_data[ids_1,ids_2]).reshape((np_ids.shape[0], np_ids.shape[0]))

        return matrix_D
    



def custom_collate(batch):
    """
    Custom collate function for batching FNSDDataset.

    Args:
        batch - list where each item is a dictionary (list)

    Returns:
        Tuple[dict, torch.Tensor]:
            - A dictionary with batched tensors
            - Labels - tensor (batch_size, 1)
    """
    # we get in this order:
    # UI graph, US graph, TI graph, UI features, US features, TI features, len_OG_user_tweets, Labels
    part_us = 'us' in batch[0]
    part_ti = 'ti' in batch[0]
    part_ui = 'ui' in batch[0]
    part_uf = 'user_features' in batch[0]

    # Determine max sizes + Initialize tensors for batching
    if part_ui:
        user_size = max(item['ui_f'].shape[0] for item in batch)
        ui_X = np.zeros((len(batch), user_size, batch[0]['ui_f'].shape[1]), dtype=np.float32)
        ui_G = np.zeros((len(batch), user_size, user_size), dtype=np.float32)
    
    if part_us:
        user_size_sim = max(item['us_f'].shape[0] for item in batch) # sim 
        us_X = np.zeros((len(batch), user_size_sim, batch[0]['us_f'].shape[1]), dtype=np.float32)   # sim 
        us_G = np.zeros((len(batch), user_size_sim, user_size_sim), dtype=np.float32)             # sim 

    if part_ti:
        tweet_size = max(item['ti_f'].shape[0] for item in batch)
        tweet_amount_per_user = [item['tweets_l'] for item in batch]
        ti_X = np.zeros((len(batch), tweet_size, batch[0]['ti_f'].shape[1]), dtype=np.float32)
        ti_g = np.zeros((len(batch), tweet_size, tweet_size), dtype=np.float32)

    if part_uf:
        user_features = np.zeros((len(batch), 1, batch[0]['user_features'].shape[0]), dtype=np.float32)

    labels = np.zeros((len(batch), 1), dtype=np.float32)
    # Fill tensors with batch data
    # broadcasting not possible due to diefferent sizes
    # ZERO padding is done here to adjust all matrices in a batch
    # what_tweets_to_consider = list() # - I think I need to add this

    for i, d in enumerate(batch):
        # loop through batch to add corresponding data to specific slots in np
        labels[i, 0] = d['label']

        if part_ui:
            ui_X_relic = d['ui_f']
            ui_G_relic = d['ui']
            ui_X[i, :ui_X_relic.shape[0], :] = ui_X_relic
            ui_G[i, :ui_G_relic.shape[0], :ui_G_relic.shape[0]] = ui_G_relic

        if part_us: 
            us_X_relic = d['us_f']            # sim 
            us_G_relic = d['us']                 # sim 
            us_X[i, :us_X_relic.shape[0], :] = us_X_relic              # sim 
            us_G[i, :us_G_relic.shape[0], :us_G_relic.shape[0]] = us_G_relic    # sim 

        if part_ti:
            ti_X_relic= d['ti_f']
            ti_g_relic = d['ti']
            ti_X[i, :ti_X_relic.shape[0], :] = ti_X_relic
            ti_g[i, :ti_g_relic.shape[0], :ti_g_relic.shape[0]] = ti_g_relic

        if part_uf:
            user_features[i, 0, :] = d['user_features']
        
    # Add the data to the dictionary as input to the model
    fnsd = dict()
    
    if part_ui:
        fnsd['ui'] = torch.from_numpy(ui_G)
        fnsd['ui_f'] = torch.from_numpy(ui_X)
    
    if part_uf:
        fnsd['user_features'] = torch.from_numpy(user_features)

    if part_us:
        fnsd['us'] = torch.from_numpy(us_G)
        fnsd['us_f'] = torch.from_numpy(us_X)

    if part_ti:
        fnsd['ti'] = torch.from_numpy(ti_g)
        fnsd['ti_f'] = torch.from_numpy(ti_X)
        fnsd['tweets_l'] = torch.tensor(tweet_amount_per_user)

    return fnsd, torch.from_numpy(labels)
    
def load_pickle_custom(file_name):
    """
    Loads a pickle file from the given path.

    Args:
        file_name - path to the pickle file (str)

    Returns:
        Any:
            The Python object stored in the pickle file.
    """
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def compose_sium_graph(graph1, graph2, path, sium_name):
    """
    Effectively constructs a data for the SIUM model:
    combines the user interaction and user similarity matrices using a logical OR operation
    
    Note: graph1 and graph2 description can be used interchangeable, as the function
    is agnostic to the order.

    Note2: saves the model in the location given, no returns.

    Args:
        graph1 - user-user-interaction graph with metadata such as missiformation spread (networkx.DiGraph)
        graph2 - user-user-similarity binary graph created by class SimMatrixC (networkx.DiGraph)
        path - path where to save newly created graph (str)
        sium_name - dataset's unique name to save it by (str)
    """
    sium_data = networkx.compose(graph1, graph2)
    with open(f'{os.path.join(path, sium_name)}.gpickle', 'wb') as f:
        pickle.dump(sium_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constructs a Similarity-Interaction-United-Graph (data for SIUM)")

    # Data paths and other parameters
    parser.add_argument("--data_name", type=str, required=True, help="Dataset's unique name to save it by ")
    parser.add_argument("--path", type=str, default="data", help="Path where to save newly created graph ")
    parser.add_argument("--gpickle_users", type=str, default="data\\gg_users_full.gpickle", help="Path to user interaction graph (UUIG)")
    parser.add_argument("--sim_matrix", type=str, default="data\\simMatrix_nx.gpickle", help="Path to similarity graph (UUSG)")

    args = parser.parse_args()

    # ===== Load inputs =====
    gpickle_users = load_pickle_custom(args.gpickle_users)
    simMatrix_nx = load_pickle_custom(args.sim_matrix)

    # ===== Run the function =====
    compose_sium_graph(gpickle_users,simMatrix_nx,args.path,args.data_name)
