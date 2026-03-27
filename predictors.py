import torch

import torch.nn as nn

from torch.nn.parameter import Parameter
from collections.abc import Callable

class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) implementation, as outlined in the paper:
    "Identifying Misinformation Spreaders: Challenges and Model Architectures".
    
    From the paper:
    GCNs were implemented as follows:
    $$\text{GCN}(X, \hat{D}, \hat{A}) = f\left( \hat{D}^{-\frac{1}{2}} \hat{A} \hat{D}^{-\frac{1}{2}} X W + b \right)$$

    Args:
        input_features - input feature dim (int)
        output_features - output feature dim (int)

    Returns:
        torch.Tensor:
            - transformed node features after GCN(\text{GCN}(X, \hat{D}, \hat{A}))
    """
    def __init__(
            self, 
            input_features: int, 
            output_features: int
        ):
        
        # Note: activation should be added outside of a layer 
        super().__init__()
        self.kernel = Parameter(torch.empty((input_features, output_features), dtype=torch.float32))

        # Not sure which is better:
        # torch.nn.init.xavier_uniform_(self.kernel) # torch.nn.init.kaiming_uniform_(weights_kaiming, a=0)
        # will use:
        torch.nn.init.kaiming_uniform_(self.kernel, a=0)

        # bias
        self.bias = Parameter(torch.zeros(output_features, dtype=torch.float32))

    def forward(self, x, a):
        # linear transformation
        partial = x @ self.kernel
        # propagates the features through the graph structure
        partial = a @ partial
        # bias term
        partial += self.bias

        return partial
    
    
class GNN(nn.Module):
    """
    Graph Neural Network (GNN) with neighbor aggregation implementation.
    Was used in ablation study, but was decided against, in favour of GCN.

    Args:
        input_features - input feature dim (int)
        output_features - output feature dim (int)

    Returns:
        torch.Tensor:
            - transformed node features after GNN
    """
    def __init__(
            self, 
            input_features: int, 
            output_features: int, 
        ):
        # Note: activation should be added outside of a layer 
        super().__init__()
        self.kernel = Parameter(torch.empty((input_features, output_features), dtype=torch.float32))

        torch.nn.init.kaiming_uniform_(self.kernel, a=0)

        # bias
        self.bias = Parameter(torch.zeros(output_features, dtype=torch.float32))

    def forward(self, x, a):
        # aggregate neighbors
        partial = a @ x
        # linear transformation
        partial = partial @ self.kernel
        # Add bias
        partial += self.bias

        return partial
    

class Predictor(nn.Module):
    """
    A Multi-Source Hybrid Model, as outlined in the paper:
    "Identifying Misinformation Spreaders: Challenges and Model Architectures".

    PredictorABLATION was used to generate the final results, as it supports additionally all
    of the parts of the ablation study, and therefore was used instead.

    Args:
        ui_gcn - defines GNN layer size for UUIG (list)
        us_gcn - defines GNN layer size for UUSG (list)
        ti_gcn - defines GNN layer size for TTIG (list)
        fnn_layers - list of FNN (list)
        user_feat - dimension of User Features (int)
        ui_X - dimension for UUIG (int)
        us_X - dimension for UUSG (int)
        tweet_f - tweet embeddings input dimension (int)
        mha_dim - multi-head attention dimension (int)
        mha_heads - number of attention heads (int)
        GNN - graph neural network layer class (GCN or GNN) (callable)
        parts - Compatibility variable, to support the same input as PredictorABLATION, not used

    Returns:
        torch.Tensor:
            - model predictions (logits) of shape (batch_size, 1)
    """
    def __init__(
            self, 
            ui_gcn: list, 
            us_gcn: list,  
            ti_gcn: list, 
            fnn_layers: list, 
            user_feat: int,
            ui_X: int, 
            us_X: int,
            tweet_f: int,
            mha_dim: int,
            mha_heads: int, 
            GNN: Callable, 
            parts: dict # Compatibility, NOT used
        ):
        super(Predictor, self).__init__()

        # GNN option: GCN is recommended
        self.GNN = GNN

        # User Interation
        self.layers_ui = self.create_layers(ui_X,ui_gcn)

        # User Similarity
        self.layers_us = self.create_layers(us_X,us_gcn)

        # Tweet Interation
        self.layers_ti = self.create_layers(tweet_f,ti_gcn)

        # Multi-head attention
        self.query = torch.nn.Linear(ti_gcn[-1], mha_dim)
        self.key = torch.nn.Linear(ti_gcn[-1], mha_dim)
        self.value = torch.nn.Linear(ti_gcn[-1], mha_dim)
        self.multiheadattention = torch.nn.MultiheadAttention(mha_dim, mha_heads,batch_first=True)

        # FNN layers
        layer_dims = [(ui_gcn[-1]+us_gcn[-1]+mha_dim+user_feat)] + fnn_layers + [1] # as predicor
        self.layers_fnn = torch.nn.ModuleList([torch.nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)])


    def forward(self, ui, ui_f, user_features, us, us_f, ti, ti_f, tweets_l):
        # User Interation
        x_ui = ui_f
        for gcn_layer in self.layers_ui:
            x_ui = torch.nn.functional.relu(gcn_layer(x_ui, ui))

        # User Similarity
        x_us = us_f
        for gcn_layer in self.layers_us:
            x_us = torch.nn.functional.relu(gcn_layer(x_us, us))

        # Tweet Interaction
        x_ti = ti_f
        for gcn_layer in self.layers_ti:
            x_ti = torch.nn.functional.relu(gcn_layer(x_ti, ti))

        # Tweet MHA + max_pool

        # mask based on tweet lengths (tweets_l)
        batch_size, max_tweets, feature_dim = x_ti.shape

        mask = torch.ones((batch_size,max_tweets),dtype=torch.bool,device=x_ti.device)
        for e,i in enumerate(tweets_l):
            mask[e,:i] = False

        # multi-head attention with mask
        q_tweets = self.query(x_ti)
        k_tweets = self.key(x_ti)
        v_tweets = self.value(x_ti)
        mha_tweets, _ = self.multiheadattention(q_tweets, k_tweets, v_tweets, key_padding_mask = mask)

        # so that padding doest affect max pooling
        mha_tweets_masked = mha_tweets.clone()
        mha_tweets_masked[mask] = float('-inf') 
        mha_tweets, _ = torch.max(mha_tweets_masked, dim=1)
        mha_tweets[torch.isinf(mha_tweets)] = 0

        # Merge all together and FNN
        x_merged = torch.concat([user_features[:, 0, :], x_ui[:, 0, :], x_us[:, 0, :], mha_tweets], dim=1) 

        for fnn_layer in self.layers_fnn[:-1]:
            x_merged = torch.nn.functional.relu(fnn_layer(x_merged))
        
        prediction = self.layers_fnn[-1](x_merged)

        return prediction
    
    def create_layers(self,input_features: int ,output_features: list):
        layer_dims = [input_features] + output_features
        return torch.nn.ModuleList([self.GNN(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)])
    



class PredictorABLATION(nn.Module):
    """
    A Multi-Source Hybrid Model, as outlined in the paper:
    "Identifying Misinformation Spreaders: Challenges and Model Architectures".

    PredictorABLATION was used to generate the final results. 
    PredictorABLATION supports additionally all of the parts of the ablation study,
    and is highly flexible, rearanging the pipeline according to the input.

    Args:
        ui_gcn - defines GNN layer size for UUIG (list)
        us_gcn - defines GNN layer size for UUSG (list)
        ti_gcn - defines GNN layer size for TTIG (list)
        fnn_layers - list of FNN (list)
        user_feat - dimension of User Features (int)
        ui_X - dimension for UUIG (int)
        us_X - dimension for UUSG (int)
        tweet_f - tweet embeddings input dimension (int)
        mha_dim - multi-head attention dimension (int)
        mha_heads - number of attention heads (int)
        GNN - graph neural network layer class (GCN or GNN) (callable)
        parts - defines what parts of the model are ablated (dict)
        convl - if True, applies FNN layers; otherwise returns merged features; -> 
        was designed this way to test RF, Decision Tree instead of FNN, but it was decided on FNN (bool, default=True)
        out - output dimension of the final layer (int)

    Returns:
        torch.Tensor:
            if convl == True  -> model predictions
            if convl == False -> concatenated feature representation
    """
    def __init__(
            self, 
            ui_gcn: list,  
            us_gcn: list, 
            ti_gcn: list, 
            fnn_layers: list, 
            user_feat: int,
            ui_X: int, 
            us_X: int,
            tweet_f: int,
            mha_dim: int,
            mha_heads: int,
            GNN: Callable,
            parts: dict, 
            convl: bool = True, 
            out: int = 1
        ):        
        super(PredictorABLATION, self).__init__()

        # GNN option: GCN is recommended
        self.GNN = GNN

        self.parts = parts # user interaction? , user imilarity? , user tweets? , user features concatination?
        # parts = {"user_inter": True,"user_sim": True,"tweets_inter": True,"user_feat": True}

        dim_input = 0

        # User Interation
        if self.parts["user_inter"] == True:
            self.layers_ui = self.create_layers(ui_X,ui_gcn)
            dim_input += ui_gcn[-1]
        

        # User Similarity
        if self.parts["user_sim"] == True:
            self.layers_us = self.create_layers(us_X,us_gcn)
            dim_input += us_gcn[-1]

        # Tweet Interation
        if self.parts["tweets_inter"] == True:
            self.layers_ti = self.create_layers(tweet_f,ti_gcn)

            # Multi-head attention
            self.query = torch.nn.Linear(ti_gcn[-1], mha_dim)
            self.key = torch.nn.Linear(ti_gcn[-1], mha_dim)
            self.value = torch.nn.Linear(ti_gcn[-1], mha_dim)
            self.multiheadattention = torch.nn.MultiheadAttention(mha_dim, mha_heads,batch_first=True)

            dim_input += mha_dim
        
        # User features concatination? - bool
        if self.parts["user_feat"] == True:
            dim_input += user_feat

        # FNN layers
        layer_dims = [dim_input] + fnn_layers + [out] # as predicor
        self.layer_dims = layer_dims
        self.convl = convl
        if self.convl:
            self.layers_fnn = torch.nn.ModuleList([torch.nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)])


    def forward(self, **kwargs): # self, ui, ui_f, user_features, us, us_f, ti, ti_f, tweets_l
        
        # Data
        fnsd_res = []

        # User features concatination - bool data
        if self.parts["user_feat"] == True:
            user_features = kwargs.get('user_features')
            fnsd_res.append(user_features[:, 0, :])

        # User Interation
        if self.parts["user_inter"] == True:
            ui = kwargs.get('ui') 
            ui_f = kwargs.get('ui_f')
            x_ui = ui_f
            for gcn_layer in self.layers_ui:
                x_ui = torch.nn.functional.relu(gcn_layer(x_ui, ui))
            fnsd_res.append(x_ui[:, 0, :])

        # User Similarity
        if self.parts["user_sim"] == True:
            us = kwargs.get('us')      # Optional
            us_f = kwargs.get('us_f')  # Optional
            x_us = us_f
            for gcn_layer in self.layers_us:
                x_us = torch.nn.functional.relu(gcn_layer(x_us, us))
            fnsd_res.append(x_us[:, 0, :])

        # Tweet Interaction
        if self.parts["tweets_inter"] == True:
            ti = kwargs.get('ti')   
            ti_f = kwargs.get('ti_f') 
            tweets_l = kwargs.get('tweets_l')  
            x_ti = ti_f
            for gcn_layer in self.layers_ti:
                x_ti = torch.nn.functional.relu(gcn_layer(x_ti, ti))

            # Tweet MHA + max_pool

            # mask based on tweet lengths (tweets_l)
            batch_size, max_tweets, feature_dim = x_ti.shape

            mask = torch.ones((batch_size,max_tweets),dtype=torch.bool,device=x_ti.device)
            for e,i in enumerate(tweets_l):
                mask[e,:i] = False

            # multi-head attention with mask
            q_tweets = self.query(x_ti)
            k_tweets = self.key(x_ti)
            v_tweets = self.value(x_ti)
            mha_tweets, _ = self.multiheadattention(q_tweets, k_tweets, v_tweets, key_padding_mask = mask)

            # so that padding doest affect max pooling
            mha_tweets_masked = mha_tweets.clone()
            mha_tweets_masked[mask] = float('-inf') 
            mha_tweets, _ = torch.max(mha_tweets_masked, dim=1)
            mha_tweets[torch.isinf(mha_tweets)] = 0

            fnsd_res.append(mha_tweets)

        # Merge all together and FNN
        x_merged = torch.concat(fnsd_res, dim=1) 
        
        if self.convl:
            for fnn_layer in self.layers_fnn[:-1]:
                x_merged = torch.nn.functional.relu(fnn_layer(x_merged))
            
            prediction = self.layers_fnn[-1](x_merged)

            return prediction
        else:
            # if I want to use any other baseline:
            # for example RF, GradientBooster.....
            # with ablation study it was decided upon initially proposed FNN
            return x_merged
    
    def create_layers(self,input_features: int ,output_features: list):
        layer_dims = [input_features] + output_features
        return torch.nn.ModuleList([self.GNN(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)])
