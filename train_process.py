import torch
import os
import re
import json
import random
import networkx
import argparse

import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.utils import compute_class_weight
from predictors import GCN, PredictorABLATION
from sitif_data import FNSDDataset, custom_collate, load_pickle_custom
from dict_operation import create_replacement_dict, transform_json_to_dict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def train(
        model: nn.Module, 
        device: torch.device, 
        train: DataLoader, 
        optimizer: torch.optim, 
        loss_function: callable, 
        epochs: int, 
        weights: np.ndarray, 
        checkpoint: callable=None, 
        name_model: str = "_", 
        validation: DataLoader = None, 
        early_stopping_tl: int = 3, 
        keep_noted: bool = True
    ):
    """
    Trains a given model using specified model, training data, and loss function.

    Args:
        model - the neural network model of class PredictorABLATION or Predictor (nn.Module)
        device - the device for the computation (torch.device)
        train - a training data loader with features and labels (DataLoader)
        optimizer - an optimizer instance from PyTorch (torch.optim)
        loss_function - loss function (callable)
        epochs - total number of epochs (int)
        weights - a ndarray (or tensor) containing weights for handling class imbalance in loss (np.ndarray)
        checkpoint (optional) - a function for saving model checkpoints (callable)
        name_model (optional): a name for the checkpoints saved (str)
        validation (optional): A validation data loader, if provided, enables early stopping (DataLoader)
        early_stopping_tl (optional): number of epochs to trigger an early stopping (int)
        keep_noted (optional): if true -> shows the progress with all prints, was used for debugging (bool)

    Returns:
        Tuple[nn.Module, float, int]:
            The best validation loss achieved during training (float)
            The epoch at which it occurred (int)
            The trained model, was trained in-place (nn.Module)
    """

    es_tolerance = 0
    best_loss = float('inf')
    val_loss = float('inf')
    final_epoch = 0

    for e in range(1, epochs + 1):
        if keep_noted:
            print(f'Epoch {e} of {epochs}')
        tbar = tqdm(train, disable=(not keep_noted))
        losses = 0
        cant = 0
        model.train()

        for x, y in tbar:
            optimizer.zero_grad()
            x_proc = {k: v.to(device) for k, v in x.items()}

            if isinstance(loss_function, SimpleCrossEntropyLoss):
                y_proc = y.to(device).long().view(-1)
            else:
                y_proc = y.to(device).float()

            pred = model(**x_proc)

            match loss_function:
                case HingeLoss():
                    loss = loss_function(pred, y_proc)
                    class_idx = (y_proc[:, 0] > 0).int().cpu().numpy()

                case torch.nn.BCELoss():
                    pred = torch.sigmoid(pred)
                    loss = loss_function(pred, y_proc)
                    class_idx = y_proc[:, 0].int().cpu().numpy()

                case SimpleCrossEntropyLoss():
                    loss = loss_function(pred, y_proc)
                    class_idx = y_proc.detach().cpu().numpy()

                case _:
                    raise ValueError(f"Unsupported loss function type: {type(loss_function)}")
                
            weight = weights[class_idx]
            loss = torch.mean(weight * loss)

            loss.backward()
            optimizer.step()

            losses += loss.item()
            cant += 1
            tbar.set_postfix(loss=(losses / cant))
            
        loss = losses / cant
        if keep_noted:
            print(f'Final average loss:\t{loss}')
        if checkpoint:
            checkpoint(model=model, epoch=e, name_model = name_model)
        # Validation eval check + early stopping
        if validation:
            model.eval()
            losses = 0
            cant = 0
            if keep_noted:
                all_preds = []
                all_labels = []
            with torch.no_grad():
                for x, y in validation:
                    x_proc = {k: v.to(device) for k, v in x.items()}
                    
                    if isinstance(loss_function, SimpleCrossEntropyLoss):
                        y_proc = y.to(device).long().view(-1)
                        class_idx = y_proc.detach().cpu().numpy()
                        weight = weights[class_idx]
                        pred = model(**x_proc)

                        loss = loss_function(pred, y_proc)
                        loss = torch.mean(weight * loss)
                        losses += loss.item()
                        cant += 1

                        if keep_noted:
                            pred_labels = torch.argmax(pred, dim=1).cpu().numpy()
                            true_labels = y_proc.cpu().numpy()
                            mask = np.isin(true_labels,[0,2])
                            filtered_true = true_labels[mask]
                            filtered_pred = pred_labels[mask]
                            all_preds.extend(filtered_pred)
                            all_labels.extend(filtered_true)

                    else:
                        y_proc = y.to(device).float()
                        class_idx = (y_proc[:, 0] > 0).int().cpu().numpy()
                        weight = weights[class_idx]
                        pred = model(**x_proc)
                        if not isinstance(loss_function, HingeLoss):
                            pred = torch.sigmoid(pred)
                            loss = loss_function(pred, y_proc)
                        else:
                            loss = loss_function(pred, y_proc)
                        loss = torch.mean(weight * loss)
                        losses += loss.item()
                        cant += 1

                        if keep_noted:
                            y_binary = (pred[:, 0] > 0).int().cpu().numpy()
                            all_preds.extend(y_binary)
                            all_labels.extend(class_idx)

            val_loss = losses / cant

            if keep_noted:
                if isinstance(loss_function, SimpleCrossEntropyLoss):
                    val_f1 = f1_score(all_labels, all_preds, labels=[0, 2], average='macro')
                    pr_s = precision_score(all_labels, all_preds, labels=[0, 2], average='macro')
                    rc_s = recall_score(all_labels, all_preds, labels=[0, 2], average='macro')
                else:
                    val_f1 = f1_score(all_labels, all_preds)
                    pr_s = precision_score(all_labels, all_preds)
                    rc_s = recall_score(all_labels, all_preds)
                mtthws_s = matthews_corrcoef(all_labels, all_preds)
                print(f'Final validatin loss:\t{val_loss:.3f}, f1_score:\t{val_f1:.3f}, ps: {pr_s:.2f}, rs: {rc_s:.2f}, mtthws_s: {mtthws_s:.2f}')

            if early_stopping_tl:
                if val_loss < best_loss:
                    best_loss = val_loss
                    es_tolerance = 0
                else:
                    es_tolerance += 1
                    if es_tolerance >= early_stopping_tl:
                        final_epoch = e
                        break

    return model, best_loss, final_epoch


class HingeLoss: 
    """
    Implements the Hinge Loss function.

    Args:
        margin - a margin value used in the loss calculation (int)
        device - the device for the computation (torch.device)
    """
    def __init__(self, 
                 margin: int = 1, 
                 device: torch.device = None
                ):
        if not device:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.margin = torch.scalar_tensor(margin, dtype=torch.float32, device=device)
        self.zero = torch.scalar_tensor(0, dtype=torch.float32, device=device)

    def __call__(self, pred, true):
        res = self.margin - pred * true
        res = torch.where(res >= self.zero, res, self.zero)
        return res



class SimpleCrossEntropyLoss:
    """
    Implements a simplified version of Cross Entropy Loss for a multi-class classification tasks. 

    Args:
        ignore_index - A bool to ignore specific indices in the target labels (bool)
        reduction - an input value reduction to nn.CrossEntropyLoss (str)
    """
    def __init__(self, ignore_index=None, reduction='none'):
        if ignore_index:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index,reduction = reduction)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self, logits, targets):
        return self.loss_fn(logits, targets)

def make_json_serializable(obj):
    """
    converts a Python object into a JSON-serializable format.

    Args:
        obj - input object to convert (any)

    Returns:
        JSON-serializable version of the input object (any)
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, type):  # catches classes like HingeLoss
        return obj.__name__
    else:
        return obj

def checkpoint(model, epoch:int, name_model:str = ""):
    """
    A function, given a model saves it's as an checkpoint. In-place, no return.

    Args:
        model - the neural network model to save (nn.Module)
        epoch - epoch number used for the name (int)
        name_model - a name used to crto save a model (str)

    Notes:
        -epoch is working as boolean for the model saving procedure, with the following behaviour:
            - If epoch  >= 0 -> model saves the model state dictionary including epoch in the name.
            - If epoch == -1 -> model saves the model state dictionary excluding epoch in the name.
            - If epoch  < -1 -> saves metrics to *_metrics_CORRUPTED.json, as an error occured.
        - checkpoint files are stored in a directory named '<name_model>'.
        - no return.
    """
    model_dir = name_model # before it was slighlty modified
    os.makedirs(model_dir, exist_ok=True)
    if (epoch < 0):
        suffix = "_metrics.json" if epoch == -1 else "_metrics_CORRUPTED.json"
        model_path = f'{model_dir}{os.sep}{model_dir}{suffix}'
        with open(model_path, 'w') as f:
            json.dump(make_json_serializable(model), f, indent=4, sort_keys=True)
    else:
        model_path = f'{model_dir}{os.sep}model-epoch_{epoch}.pt'
        torch.save(model.state_dict(), model_path)

def preprocess(df):
    """
    Preprocesses a dataframe by removing non-informative columns and scaling selected features. 
    Not In-place.

    Behaviour:
        1. Removes columns with only a single unique value
        2. Applies standardization (StandardScaler) to the columns that are not in [-1, 1]
        3. Clips the standardized values to [-2, 2]

    Args:
        df - dataframe to preprocess (pd.DataFrame)

    Returns:
        preprocess dataframe (pd.DataFrame)
    """
    # Identifying columns with a single unique value
    one_value_columns = [c for c in df.columns if df[c].nunique() == 1]
    df = df.drop(one_value_columns, axis=1)
    print(f'Removing columns with one value {one_value_columns}')
    
    # Identifying columns with values strictly between -1 and 1
    valid_columns = [c for c in df.columns if -1 <= df[c].min() and df[c].max() <= 1]
    print(f'There are {len(valid_columns)} columns between -1 and 1')
    
    df_freeze = df[valid_columns].astype(np.float32)
    df_process = df.drop(valid_columns, axis=1)
    
    # Scaling the remaining columns
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_process)
    df_process = pd.DataFrame(np.clip(scaled_values, -2, 2), columns=df_process.columns, index=df.index)
    
    # Concatenating processed and unprocessed data
    df = pd.concat([df_freeze, df_process], axis=1)
    return df



def hp_to_str(checkpoint, i): 
    """
    Converts a hyperparameter value into a filesystem-safe string identifier.

    Behavior:
        1. int, float, str, bool -> converted directly to str
        2. Class -> represented by the class name
        3.1. list (<4  elements) -> expanded and is converted directly to str 
        3.2. list (>=4 elements) -> summarized with length and a random key
        3.3. list (1st element class/object) -> represented by the class name
        4. object -> represented by the class name
        5. DataFrame -> summarized with length and a random key

    Args:
        checkpoint - name prefix (str)
        i - hyperparameter to convert (any)

    Returns:
        str:
            A string in the format "hp_<checkpoint>_<value>"
    """
    # used for saving the options and name them appropriately
    def safe_str(x):
        s = str(x)
        return re.sub(r"[<>:\"/\\|?*']", "", s)  # remove invalid chars
    if isinstance(i, (int, float, str, bool)):
        return f"hp_{checkpoint}_{safe_str(i)}"
    elif isinstance(i, type): # if it's a class 
        return f"hp_{checkpoint}_{i.__name__}" # e.g. BCELoss
    elif isinstance(i, list):  # special case for lists
        hp_to_str_i = i[1] if isinstance(i[0], bool) else i
        if isinstance(i[0], type):
            second_el = "ERROR" if len(i) < 2 else i[1]
            return f"hp_{checkpoint}_{i[0].__name__}_{safe_str(second_el)}" # if optimizer
        if not isinstance(hp_to_str_i, list):
            if isinstance(hp_to_str_i, type): # if it's a class 
                return f"hp_{checkpoint}_{hp_to_str_i.__name__}" # e.g. BCELoss
            elif hasattr(hp_to_str_i, "__class__"):
                # If DataFrame (or similar), force compact representation
                if "DataFrame" in hp_to_str_i.__class__.__name__:
                    unique_key = ''.join(random.choices("0123456789", k=5))
                    return f"hp_{checkpoint}_len_{len(hp_to_str_i)}_key_{unique_key}"
                return f"hp_{checkpoint}_{hp_to_str_i.__class__.__name__}"
        if len(hp_to_str_i) == 0:
            return f"hp_{checkpoint}_0_empty_False"
        if len(hp_to_str_i) <= 4:
            # Expand full list
            parts = [safe_str(x) for x in hp_to_str_i]
            return f"hp_{checkpoint}_" + "_".join(parts)
        else:
            # Compact representation
            unique_key = ''.join(random.choices("0123456789", k=4))
            return f"hp_{checkpoint}_len_{len(hp_to_str_i)}_key_{unique_key}"
    elif hasattr(i, "__class__"):  # if it's an object
        return f"hp_{checkpoint}_{i.__class__.__name__}"
    else:  # fallback
        return f"hp_{checkpoint}_{safe_str(i)}"



def train_given_dict(
        model_hyperparameters: dict,
        name_model: str,

        gpickle_users: networkx.DiGraph,
        gpickle_tweets: networkx.DiGraph,
        simMatrix_nx: networkx.DiGraph,
        df_tweets_e_BERT: pd.DataFrame,
        users_feature_df_preproc: pd.DataFrame,
        weights: np.ndarray,
        checkpoint_p: callable,
        device: torch.device,

        train_ids: set,
        val_ids: set,
        threshold: float = 0.5,
        keep_noted: bool = True, 
        save_model_locally: bool = True
    ):
    """
    Trains a model given a dictionary of hyperparameters. An example of the structure
    of model_hyperparameters is:

    {"user_inter": [True,[32, 32, 32]],"user_sim": [True,[32]],"tweets_inter": [True,[64,16]],
    "user_feat": [True,all_data_no_stats], "ui_and_us_X": list(tree_metric),"fnn": [32, 64, 32], 
    "optimizer": [torch.optim.AdamW,0.0005], "loss": HingeLoss, "max_epochs": 20, "batch": 12, 
    'early_stopping_tl': 5, "mha": [30,3]}

    Args:
        model_hyperparameters - dictionary of model hyperparameters and overall setup (dict)
        name_model - checkpoint name (str)
        gpickle_users - user-user-interaction graph with metadata such as missiformation spread (networkx.DiGraph)
        gpickle_tweets - tweet-tweet-interaction graph (networkx.DiGraph)
        simMatrix_nx - user-user-similarity binary graph created by class SimMatrixC (networkx.DiGraph)
        df_tweets_e_BERT - SentenceTransformer tweet embeddings (pd.DataFrame)
        users_feature_df_preproc - preprocessed user features (pd.DataFrame)
        weights - a ndarray (or tensor) containing weights for handling class imbalance in loss (np.ndarray)
        checkpoint_p - a function for saving model checkpoints (callable)
        device - computation device (torch.device)
        train_ids, val_ids - identifiers for training and validation samples
        threshold - sets the thresholds to split the data. Common is 0,5 and binary, can be list for multiple-choice setting (float or list)
        keep_noted - if True, prints all of the updates (bool)
        save_model_locally - if True, saves trained models (bool)

    Returns:
        Tuple[nn.Module, float, int]:
            The trained model, was trained in-place (nn.Module)
            The best validation loss achieved during training (float)
            The epoch at which it occurred (int)
    """

    if isinstance(model_hyperparameters['user_feat'][1], list):
        model_hyperparameters['user_feat'][1] = users_feature_df_preproc[model_hyperparameters['user_feat'][1]]
    if isinstance(model_hyperparameters['ui_and_us_X'], list):
        model_hyperparameters['ui_and_us_X'] = users_feature_df_preproc[model_hyperparameters['ui_and_us_X']]

    uf_len = len(model_hyperparameters['user_feat'][1].columns)
    uius_len = len(model_hyperparameters['ui_and_us_X'].columns)

    parts = {"user_inter": model_hyperparameters['user_inter'][0],"user_sim": model_hyperparameters['user_sim'][0],
                "tweets_inter": model_hyperparameters['tweets_inter'][0],"user_feat": model_hyperparameters['user_feat'][0]}

    ds_train = FNSDDataset(gpickle_users,gpickle_tweets,simMatrix_nx,model_hyperparameters['user_feat'][1],df_tweets_e_BERT,len(model_hyperparameters['user_inter'][1]),
                            len(model_hyperparameters['user_sim'][1]),len(model_hyperparameters['tweets_inter'][1]),threshold,
                            train_ids,parts = parts, us_X = model_hyperparameters['ui_and_us_X'], ui_X = model_hyperparameters['ui_and_us_X']) # SET TO TRAIN + VAL IDS
    ds_val = FNSDDataset(gpickle_users,gpickle_tweets,simMatrix_nx,model_hyperparameters['user_feat'][1],df_tweets_e_BERT,len(model_hyperparameters['user_inter'][1]),
                            len(model_hyperparameters['user_sim'][1]),len(model_hyperparameters['tweets_inter'][1]),threshold,
                            val_ids,parts = parts, us_X = model_hyperparameters['ui_and_us_X'], ui_X = model_hyperparameters['ui_and_us_X']) # SET TO TEST IDS
    
    dl_train = torch.utils.data.DataLoader(ds_train,batch_size=model_hyperparameters['batch'],collate_fn=custom_collate,shuffle=True)
    dl_val = torch.utils.data.DataLoader(ds_val,batch_size=20,collate_fn=custom_collate,shuffle=False)

    out = len(threshold) + 1 if (model_hyperparameters['loss'] == SimpleCrossEntropyLoss) else 1

    model = PredictorABLATION(model_hyperparameters['user_inter'][1],model_hyperparameters['user_sim'][1],model_hyperparameters['tweets_inter'][1],
                                model_hyperparameters['fnn'],uf_len,uius_len,uius_len,len(df_tweets_e_BERT.columns),model_hyperparameters['mha'][0],
                                model_hyperparameters['mha'][1],GCN,parts, out=out)
    
    model.to(device)

    optimizer = model_hyperparameters['optimizer'][0](model.parameters(), lr=model_hyperparameters['optimizer'][1])
    
    loss = None
    if (model_hyperparameters['loss'] == torch.nn.BCELoss):
        loss = model_hyperparameters['loss']().to(device=device)
    else:
        loss = model_hyperparameters['loss']()

    model, best_loss, final_epoch = train(model, device, dl_train, optimizer, loss, model_hyperparameters['max_epochs'],
                                    weights, checkpoint=(checkpoint_p if save_model_locally else None), 
                                    name_model = name_model, validation=dl_val, 
                                    early_stopping_tl=model_hyperparameters['early_stopping_tl'],keep_noted=keep_noted)
    
    return model, best_loss, final_epoch

def get_weights(
        user_features: pd.DataFrame,
        threshold: list, 
        users_given: set,
        ui_metadata: networkx.DiGraph, 
        users_feature_df_preproc: pd.DataFrame,
        device: torch.device
    ):
    """
    Calculates weights for the weighted loss calculation, to account for class imbalance.

    Partially, a part of FNSDDataset, used to calculate labels specifically for weight calculation, 
    to omit creating a separate dataset.
    
    Args:
        user_features - user features, that are as separate class to the model (pd.DataFrame)
        threshold - set the thresholds to split the data. Common is 0,5 and binary,
        if 2 are given, then we have a case of [non-spreaders; potential spreaders; spreaders](list (multiclass) or float (binary))
        users_given - a set that sets which users are used for the dataset (set)
        ui_metadata - a metadata information for threshold, which is used if usern_nx is not given (networkx.DiGraph)
        users_feature_df_preproc - preprocessed user features (pd.DataFrame)
        device - computation device (torch.device)

    Returns:
        np.ndarray
            - weights, a ndarray (or tensor) containing weights for handling class imbalance in loss
    """
    if isinstance(user_features, list):
        user_features = users_feature_df_preproc[user_features]
    u_features = user_features.sort_index()
    isolated_nodes = {n for n in ui_metadata.nodes() if ui_metadata.out_degree(n) == 0}
    # nodes that will not cause any error further and are required
    valid_nodes = {
        node for node in u_features.index
        if node in ui_metadata.nodes and node in users_given and node not in isolated_nodes}
    valid_nodes = sorted(valid_nodes)

    # Multiclass classification
    # or binary classification {0, 1}
    s_thresholds = sorted(threshold)
    def get_class(score):
        for enum, tr in enumerate(s_thresholds): # 0.2 0.5 1
            if score <= tr:
                return enum
        return len(s_thresholds)
    
    labels = pd.Series({
        node: get_class(ui_metadata.nodes[node]['score_graph'])
        for node in valid_nodes})
    
    classes = labels.values
    unique_classes = np.array(list(range(0, (len(threshold)+1))))

    if not isinstance(threshold, list):
        threshold = list(threshold)
        
    weights = compute_class_weight('balanced', classes=unique_classes, y=classes)
    weights = torch.from_numpy(np.asarray(weights)).float().to(device)

    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model  and save it (optional), given the path to hyperparameters in json format")

    # Required paths / identifiers
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model, saves the model in the folder by this name")
    parser.add_argument("--model_hp_json", type=str, help="Path to model hyperparameters JSON file, if not included," \
        "looks for model_hyperparameters.json in the folder by the name of model_name")
    
    # Data paths
    parser.add_argument("--gpickle_users", type=str, default="data\\gg_users_full.gpickle", help="Path to user interaction graph (UUIG)")
    parser.add_argument("--gpickle_tweets", type=str, default="data\\gg_tweets_full.gpickle", help="Path to tweet interaction graph (TTIG)")
    parser.add_argument("--sim_matrix", type=str, default="data\\simMatrix_nx.gpickle", help="Path to similarity graph (UUSG)")
    parser.add_argument("--tweets_embeddings", type=str, default="data\\df_tweets_embeddings_bert.pickle", help="Path to tweet BERT embeddings (DataFrame)")
    parser.add_argument("--user_features", type=str, default="data\\df_user_feature_sets___full_enrich.pickle", help="Path to preprocessed user features")
    parser.add_argument("--split_pkl", type=str, default="data\\users_profile_split_temporal.pickle", help="Path to the split (tuple)")

    # Optional hyperparameters / training settings
    parser.add_argument("--val_size_of_train", type=float, default=0.14, help="Fraction of train data to use for validation")
    parser.add_argument("--threshold", type=float, nargs='+', default=[0.5],
                    help="Threshold(s) for class prediction. Single float = binary, list = multiclass")
    parser.add_argument("--keep_noted", action="store_true", help="Print updates during training")
    parser.add_argument("--save_model_locally", action="store_true", help="Save trained model locally")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda or cpu)")

    args = parser.parse_args()

    # ===== Load inputs =====
    model_name = os.path.join("models", args.model_name)
    if args.model_hp_json:
        with open(args.model_hp_json, "r") as f:
            model_hyperparameters = json.load(f)
    else:
        if ((os.path.isdir(model_name)) and ("model_hyperparameters.json" in os.listdir(model_name))):
            json_path = os.path.join(model_name, "model_hyperparameters.json")
            with open(json_path, "r") as f:
                model_hyperparameters = json.load(f)
        else:
            raise ValueError("Variable model_hp_json was not given and not found using default path (model_name folder)")
    
    gpickle_users = load_pickle_custom(args.gpickle_users)
    gpickle_tweets = load_pickle_custom(args.gpickle_tweets)
    simMatrix_nx = load_pickle_custom(args.sim_matrix)
    split_pkl = load_pickle_custom(args.split_pkl)

    df_tweets_e_BERT = load_pickle_custom(args.tweets_embeddings)
    users_feature_df_preproc = load_pickle_custom(args.user_features)
    users_feature_df_preproc = preprocess(users_feature_df_preproc)

    if not (0 < args.val_size_of_train < 1):
        raise ValueError("val_size_of_train must be between 0 and 1 (0:1), 0 and 1 excluded")
    train_val_ids = sorted(split_pkl[0][0]) 
    split_tr_vl = int(len(train_val_ids)*(1-args.val_size_of_train))

    train_ids = set(train_val_ids[:split_tr_vl]) # 66 % from original dataset -> if 0.14 is used
    val_ids = set(train_val_ids[split_tr_vl:])   # 10 % from original dataset -> if 0.14 is used

    checkpoint_p = partial(checkpoint)
    device = torch.device(args.device)
    keep_noted = args.keep_noted
    save_model_locally = args.save_model_locally

    # ===== Prepare hyperparameters =====
    replacement_dict = create_replacement_dict(users_feature_df_preproc, HingeLoss=HingeLoss, SimpleCrossEntropyLoss=SimpleCrossEntropyLoss)
    model_hp = transform_json_to_dict(model_hyperparameters, replacement_dict)

    # ===== Prepare threshold and adapt it to the loss =====
    threshold = args.threshold
    for i in threshold:
        if not (0 < i < 1):
            raise ValueError("threshold must be between 0 and 1 (0:1), 0 and 1 excluded")
    if ((len(threshold) == 1) and (model_hp["loss"] is HingeLoss)):
        threshold = threshold[0]

    # ===== weights =====
    weights = get_weights(
        user_features=model_hp['user_feat'][1],
        threshold=(threshold if isinstance(threshold, list) else [threshold]), # we specifically need it as list
        users_given=train_val_ids,
        ui_metadata=gpickle_users,
        users_feature_df_preproc = users_feature_df_preproc,
        device = device
    )

    # ===== Run training =====
    _, best_loss, final_epoch = train_given_dict(
        model_hp,
        model_name,
        gpickle_users,
        gpickle_tweets,
        simMatrix_nx,
        df_tweets_e_BERT,
        users_feature_df_preproc,
        weights,
        checkpoint_p,
        device,
        train_ids,
        val_ids,
        threshold,
        keep_noted, 
        save_model_locally
    )

    print(f"The best validation loss achieved during training:\t{best_loss}")
    print(f"The epoch at which it occurred:\t{final_epoch}")
