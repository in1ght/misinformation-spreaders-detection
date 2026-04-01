import torch
import os
import json
import argparse
import pickle

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx

from torch.utils.data import DataLoader
from sitif_data import FNSDDataset, custom_collate, load_pickle_custom
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from tqdm import tqdm

from predictors import GCN, PredictorABLATION
from train_process import SimpleCrossEntropyLoss, HingeLoss, preprocess
from dict_operation import create_replacement_dict, transform_json_to_dict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def predict(
        dl: DataLoader,
        model: nn.Module,  
        epoch: int = 20, 
        name: str = "fixed", 
        threshold: float = 0.5, # RELIC
        tqdm_disable: bool = False
    ):
    """
    Given a model generates predictions on a given dataloader.
    A relic,that was replaced later by prediction.
    Is still used by the function draw_data_pred, so was not removed.

    Args:
        dl - testing dataloader (DataLoader)
        model - a trained model (PredictorABLATION) (nn.Module)
        epoch - epoch number of the checkpoint to load (int)
        name - directory name for loading checkpoints (str)
        threshold - RELIC, not used
        tqdm_disable - if True, disables progress bar (bool)

    Returns:
        pd.DataFrame:
            DataFrame with columns 'User', 'y_true', 'y_pred' containing true and predicted labels.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(f'final_{name}/model-epoch_{epoch}.pt'))
    model = model.to(device)
    y_true = list()
    y_pred = list()
    tbar = tqdm(dl, disable=tqdm_disable)
    for x, y in tbar:

        x_proc = {k: v.to(device) for k, v in x.items()}

        # Make the predictions
        pred = model(**x_proc)

        y_binary = (pred[:, 0] > 0).int().cpu().numpy()

        y_proc = y.to(device).float()
        class_idx = (y_proc[:, 0] > 0).int().cpu().numpy()

        y_true.extend(class_idx)
        y_pred.extend(y_binary)
        
    df = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred}, 
                      columns=['User', 'y_true', 'y_pred'])
    df.set_index('User')
    return df

def draw_data_pred(
        model_hp: dict, 

        gpickle_users: nx.DiGraph,
        gpickle_tweets: nx.DiGraph,
        simMatrix_nx: nx.DiGraph,
        df_tweets_e_BERT: pd.DataFrame,
        users_feature_df_preproc: pd.DataFrame,
        test_ids: set,

        last_epoch: int = 20, 
        plotting: bool = True, 
        name: str = "fixed", 
        threshold = 0.5
    ):
    """
    Evaluates a trained model across multiple epochs and plots metrics if plotting == True.

    Args:
        model_hp - dictionary of model hyperparameters and overall setup (dict)

        gpickle_users - user-user-interaction graph with metadata such as missiformation spread (networkx.DiGraph)
        gpickle_tweets - tweet-tweet-interaction graph (networkx.DiGraph)
        simMatrix_nx - user-user-similarity binary graph created by class SimMatrixC (networkx.DiGraph)
        df_tweets_e_BERT - SentenceTransformer tweet embeddings (pd.DataFrame)
        users_feature_df_preproc - preprocessed user features (pd.DataFrame)
        weights - a ndarray (or tensor) containing weights for handling class imbalance in loss (np.ndarray)

        last_epoch - last (max) epoch to evaluate (int)
        plotting - if True, generates performance plots (bool)
        name - directory name for loading checkpoints (str)
        threshold - thresholds to split the data. Common is 0,5 and binary, can be list for multiple-choice setting (float or list)

    Returns:
        List[dict]:
            Each dictionary contains:
            - 'name': metric name (Precision, Recall, Matthews_corrcoef, F1 Score)
            - 'result': list of metric values per epoch
    """
    
    precision_list, recall_list, matthews_corrcoef_list, f1_score_list = list(), list(), list(), list()
    indexes = [x for x in range(1,last_epoch+1)]

    # model = Predictor(*hyperparameters)
    if isinstance(model_hp['user_feat'][1], list):
        model_hp['user_feat'][1] = users_feature_df_preproc[model_hp['user_feat'][1]]
    if isinstance(model_hp['ui_and_us_X'], list):
        model_hp['ui_and_us_X'] = users_feature_df_preproc[model_hp['ui_and_us_X']]

    uf_len = len(model_hp['user_feat'][1].columns)
    uius_len = len(model_hp['ui_and_us_X'].columns)

    parts = {"user_inter": model_hp['user_inter'][0],"user_sim": model_hp['user_sim'][0],
                "tweets_inter": model_hp['tweets_inter'][0],"user_feat": model_hp['user_feat'][0]}
    
    out = len(threshold) + 1 if (isinstance(model_hp['loss'], SimpleCrossEntropyLoss)) else 1
    
    

    model = PredictorABLATION(model_hp['user_inter'][1],model_hp['user_sim'][1],model_hp['tweets_inter'][1],
                                model_hp['fnn'],uf_len,uius_len,uius_len,len(df_tweets_e_BERT.columns),model_hp['mha'][0],
                                model_hp['mha'][1],GCN,parts, out=out)

    ds_test = FNSDDataset(gpickle_users,gpickle_tweets,simMatrix_nx,model_hp['user_feat'][1],df_tweets_e_BERT,len(model_hp['user_inter'][1]),
                               len(model_hp['user_sim'][1]),len(model_hp['tweets_inter'][1]),0.5,
                               test_ids,parts = parts, us_X = model_hp['ui_and_us_X'], ui_X = model_hp['ui_and_us_X']) # SET TO TEST IDS
        
    dl_test = torch.utils.data.DataLoader(ds_test,batch_size=20,collate_fn=custom_collate,shuffle=False)

    for epoch in tqdm(indexes):

        results_simple = predict(dl_test, model, epoch, name, threshold, tqdm_disable = True)

        # Compute metrics
        precision = precision_score(results_simple['y_true'].values, results_simple['y_pred'].values)
        recall = recall_score(results_simple['y_true'].values, results_simple['y_pred'].values)
        matthews_corrcoef_score = matthews_corrcoef(results_simple['y_true'].values, results_simple['y_pred'])
        f1_score_ = f1_score(results_simple['y_true'].values, results_simple['y_pred'].values)

        precision_list.append(precision)
        recall_list.append(recall)
        matthews_corrcoef_list.append(matthews_corrcoef_score)
        f1_score_list.append(f1_score_)

    metrics_data = [
        {"name":"Precision score","result": precision_list},
        {"name":"Recall score","result": recall_list},
        {"name":"Matthews_corrcoef","result": matthews_corrcoef_list},
        {"name":"F1 Score","result": f1_score_list}
    ]

    # plotting
    if plotting == True:
        # 2 by 2 subplot
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        axs = axs.flatten()

        for ax, res_d in zip(axs,metrics_data):
            ax.set_title(res_d["name"])
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Performance')
            ax.plot(indexes, res_d['result'])

        # adjust layout
        plt.tight_layout()
        plt.show()
    
    return metrics_data


def prediction(
        dl: DataLoader,
        model: nn.Module, 
        path: str, 
        tqdm_disable: bool = True, 
        device: str = "cuda", 
        multiclass: bool = False
    ):
    """
    A more general version  a function `predict`, which was used for prediction.
    Generates predictions from a trained model with optional multiclass support.

    Args:
        dl - testing dataloader (DataLoader)
        model - a trained model (PredictorABLATION) (nn.Module)
        path - path to the saved model checkpoint(s) (str)
        tqdm_disable - if True, disables progress bar (bool)
        device - computation device (torch.device)
        multiclass - if True, predicts multiclass labels (bool)

    Returns:
        pd.DataFrame:
            DataFrame with columns 'y_true' and 'y_pred' containing the true and predicted labels.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()

    y_true, y_pred = list(), list()
    tbar = tqdm(dl, disable=tqdm_disable)
    # BCELoss - special case - lets see how it works with base system
    # Also add automatic saving of json to hypertuning
    with torch.no_grad():
        for x, y in tbar:
            x_proc = {k: v.to(device) for k, v in x.items()}

            if multiclass:
                y_proc = y.to(device).long().view(-1)  # multiclass targets should be long/int

                # Make predictions
                pred = model(**x_proc)
                y_pred_model = torch.argmax(pred, dim=1).cpu().numpy()
                class_idx = y_proc.cpu().numpy()

            else:
                y_proc = y.to(device).float()
                # Make the predictions
                pred = model(**x_proc)

                y_pred_model = (pred[:, 0] > 0).int().cpu().numpy()
                # class_idx is true, cause initially given variables
                # are that of hinge loss i.e.: -1 and 1. so we make it binary
                class_idx = (y_proc[:, 0] > 0).int().cpu().numpy()
                


            y_true.extend(class_idx)
            y_pred.extend(y_pred_model)
        
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    return df

def prediction_table_per_epoch(
        model_hp: dict, 
        model_name: str, 
        path: str,

        gpickle_users: nx.DiGraph,
        gpickle_tweets: nx.DiGraph,
        simMatrix_nx: nx.DiGraph,
        df_tweets_e_BERT: pd.DataFrame,
        users_feature_df_preproc: pd.DataFrame,
        test_ids: set,

        last_epoch: int = 20,
        epochs_singular: bool = False,
        save: bool = True,
        tqdm_disable: bool = True,
        simMatrix_nx_repl = None, # RELIC
        raw_results_return: bool = False
    ):
    """
    Evaluates a trained model across multiple epochs, computes classification metrics, if raw_results_return, saves results.

    Args:
        model_hp - dictionary of model hyperparameters and feature selections (dict)
        model_name - name of the model used for saving metrics (str)
        path - path to the saved model checkpoint(s) (str)
        
        gpickle_users - user-user-interaction graph with metadata such as missiformation spread (networkx.DiGraph)
        gpickle_tweets - tweet-tweet-interaction graph (networkx.DiGraph)
        simMatrix_nx - user-user-similarity binary graph created by class SimMatrixC (networkx.DiGraph)
        df_tweets_e_BERT - SentenceTransformer tweet embeddings (pd.DataFrame)
        users_feature_df_preproc - preprocessed user features (pd.DataFrame)
        weights - a ndarray (or tensor) containing weights for handling class imbalance in loss (np.ndarray)

        last_epoch - last (max) epoch to evaluate (int)
        epochs_singular - if True, evaluates only the last_epoch (bool)
        save - if True, saves computed metrics as CSV and Pickle (bool)
        tqdm_disable - if True, disables progress bar (bool)
        simMatrix_nx_repl - RELIC, not used, placeholder for legacy compatibility
        raw_results_return - if True, returns raw predictions and true labels instead of DataFrame (bool)

    Returns:
        if raw_results_return is False -> DataFrame with columns ['epoch', 'precision', 'recall', 'matthews_corrcoef', 'f1'] 
        if raw_results_return is True  -> Tuple of raw y_true and y_pred arrays per epoch 
    """
    precision_list, recall_list, matthews_corrcoef_list, f1_score_list = list(), list(), list(), list()
    if epochs_singular:
        epochs = [last_epoch]
    else:
        epochs = list(range(1, last_epoch + 1))

    # model unpacking -> the attributes can be given as list or as pandas, which is a relic
    # but to make everything compatable
    if isinstance(model_hp['user_feat'][1], list):
        model_hp['user_feat'][1] = users_feature_df_preproc[model_hp['user_feat'][1]]
    if isinstance(model_hp['ui_and_us_X'], list):
        model_hp['ui_and_us_X'] = users_feature_df_preproc[model_hp['ui_and_us_X']]

    uf_len = len(model_hp['user_feat'][1].columns)
    uius_len = len(model_hp['ui_and_us_X'].columns)

    parts = {
        "user_inter": model_hp['user_inter'][0],
        "user_sim": model_hp['user_sim'][0],
        "tweets_inter": model_hp['tweets_inter'][0],
        "user_feat": model_hp['user_feat'][0]
    }
    
    # to optimize it for multi-class classification, although prediction def does
    # support it yet
    # out = len(model_hp['threshold']) + 1 if \
    #     (model_hp['loss'].__name__ == "SimpleCrossEntropyLoss" and model_hp['loss'].__module__ == "__main__") else 1
    out = len(model_hp['threshold']) + 1 if (isinstance(model_hp['loss'], SimpleCrossEntropyLoss)) else 1
    average_type = "macro" if out >= 2 else "binary"
    
    model = PredictorABLATION(
        model_hp['user_inter'][1],model_hp['user_sim'][1],model_hp['tweets_inter'][1],
        model_hp['fnn'],uf_len,uius_len,uius_len,len(df_tweets_e_BERT.columns),model_hp['mha'][0],
        model_hp['mha'][1],GCN,parts, out=out
    )

    ds_test = FNSDDataset(
        gpickle_users,gpickle_tweets,simMatrix_nx,model_hp['user_feat'][1],
        df_tweets_e_BERT,len(model_hp['user_inter'][1]),
        len(model_hp['user_sim'][1]),len(model_hp['tweets_inter'][1]),model_hp['threshold'],
        test_ids,parts = parts, us_X = model_hp['ui_and_us_X'], 
        ui_X = model_hp['ui_and_us_X']
    )
        
    dl_test = torch.utils.data.DataLoader(ds_test,batch_size=64,collate_fn=custom_collate,shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    y_true_results, y_pred_results = np.zeros(shape=(len(epochs),len(ds_test))), np.zeros(shape=(len(epochs),len(ds_test)))

    for epoch in tqdm(epochs, desc="Evaluating epochs", disable=tqdm_disable):
        path_epoch = os.path.join(path, f'model-epoch_{epoch}.pt')

        results = prediction(dl_test, model, path_epoch, tqdm_disable = True, device = device, multiclass=(out > 1))
        
        precision_list.append(precision_score(results["y_true"], results["y_pred"],average=average_type))
        recall_list.append(recall_score(results["y_true"], results["y_pred"],average=average_type))
        matthews_corrcoef_list.append(matthews_corrcoef(results["y_true"], results["y_pred"]))
        f1_score_list.append(f1_score(results["y_true"], results["y_pred"],average=average_type))
        y_true_results[(0 if epochs_singular else epoch-1)] = results["y_true"]
        y_pred_results[(0 if epochs_singular else epoch-1)] = results["y_pred"]

    df_metrics = pd.DataFrame({
        "epoch": epochs,
        "precision": precision_list,
        "recall": recall_list,
        "matthews_corrcoef": matthews_corrcoef_list,
        "f1": f1_score_list
    })

    if save:
        csv_path = os.path.join(path, f"{model_name}_metrics.csv")
        pkl_path = os.path.join(path, f"{model_name}_metrics.pkl")

        df_metrics.to_csv(csv_path, index=False)
        df_metrics.to_pickle(pkl_path)

    if raw_results_return:
        return y_true_results, y_pred_results
    
    return df_metrics


def save_all_metrics(
        path: str, 
        start_name: str, 
        last_epoch: int,
        epochs_singular: bool, 
        replacement_dict: dict, 
        show_progress: bool = False
    ):
    """
    Iterates through all folders inside `path` whose names start with `start_name`
    signifying hyperparameters used and having at least as many epochs as last_epoch
    is given, then analyzes it and saves results of all epochs (or 1) in the same folder.

    Note: No return, metrics are saved directly into each model's folder.

    Args:
        path - parental path to the saved folders (str)
        start_name - directory prefix for folder containing models which are evaluated (str)
        last_epoch - last (max) epoch to evaluate (int)
        epochs_singular - if True, evaluates only the last_epoch (bool)
        replacement_dict - dictionary mapping placeholders to objects (dict)
        show_progress - if True, enables progress bar display (bool)
    """
    tbar = tqdm(os.listdir(path), desc="Evaluating models", disable=not show_progress)
    for file_item in tbar:
        folder_path = os.path.join(path, file_item)

        if (
            (os.path.isdir(folder_path)) and 
            (file_item.startswith(start_name)) and
            ("model_hyperparameters.json" in os.listdir(folder_path)) and
            (not (f"{file_item}_metrics.csv" in os.listdir(folder_path)))
        ):
            json_path = os.path.join(folder_path, "model_hyperparameters.json")
            with open(json_path, "r", encoding="utf-8") as f:
                model_hp = json.load(f)

            model_hp = transform_json_to_dict(model_hp,replacement_dict)

            tbar.set_postfix(current_model=file_item)
                
            # =============== Validation on if any epochs are missing =============== #
            missing_epochs = [
                epoch for epoch in ([last_epoch] if epochs_singular else range(1, last_epoch + 1))
                if not os.path.exists(os.path.join(folder_path, f"model-epoch_{epoch}.pt"))
            ]
            if missing_epochs:
                print(f"{folder_path}:\tmissing following epochs\t{missing_epochs}\n")
                continue
            # =============== Validation on if any epochs are missing =============== #       
            
            prediction_table_per_epoch(model_hp,file_item,folder_path,last_epoch,epochs_singular,True,True)


def get_1epoch_metrics(
        path: str, 
        start_name: str, 
        epoch: int, 
        show_progress: bool = False, 
        save: bool = False
    ):
    """
    Collects metrics for a specific epoch across multiple models, used for further analysis.

    Args:
        path - parental path to the saved folders (str)
        start_name - directory prefix for folder containing models which are to be evaluated (str)
        epoch - specific epoch to extract metrics for (int)
        show_progress - if True, enables progress bar display (bool)
        save - if True, saves computed metrics as CSV and Pickle (bool)

    Returns:
        pd.DataFrame:
            DataFrame indexed by model folder names containing metrics for the specified epoch.
    """
    tbar = tqdm(os.listdir(path), desc="Evaluating models", disable=not show_progress)
    rows, index = [], []
    for file_item in tbar:
        folder_path = os.path.join(path, file_item)
        if (
            (os.path.isdir(folder_path)) and 
            (file_item.startswith(start_name)) and
            ((f"{file_item}_metrics.pkl" in os.listdir(folder_path)))
        ):
            obj = pd.read_pickle(os.path.join(folder_path, f"{file_item}_metrics.pkl")).loc[epoch]
            rows.append(obj)
            index.append(file_item)
    df = pd.DataFrame(rows, index=index)

    if save:
        csv_path = os.path.join(path, f"{start_name}_all_metrics.csv")
        pkl_path = os.path.join(path, f"{start_name}_all_metrics.pkl")
        df.to_csv(csv_path, index=True)
        df.to_pickle(pkl_path)

    return df


def get_best_epoch(
        path, 
        start_name, 
        epochs: list, 
        main_attribute: str = "matthews_corrcoef", 
        show_progress: bool = False
    ):
    """
    Used during validation, determines the best epoch across multiple models based on a chosen metric.
    Was used to choose 13 epoch as an upper limit.

    Args:
        path - parental path to the saved folders (str)
        start_name - directory prefix for folder containing models which are to be evaluated (str)
        epochs - list of epochs to evaluate (list)
        main_attribute - metric to determine best epoch (str)
        show_progress - if True, enables progress bar display (bool)

    Returns:
        pd.DataFrame:
            DataFrame indexed by epoch, containing mean metrics across models, sorted by `main_attribute` descending.
    """
    tbar = tqdm(epochs, desc="Evaluating models", disable=not show_progress)
    rows, index = [], []
    for epoch in tbar:
        values = get_1epoch_metrics(path,start_name,epoch,False).mean()
        rows.append(values)
        index.append(epoch)
    print(epoch)
    df = pd.DataFrame(rows, index=index)
    return df.sort_values(by=main_attribute, ascending=False)



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction and evaluation for a trained model")

    # Required paths / identifiers
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--path", type=str, help="Path to model checkpoints directory, if not included, " \
        "uses model_name as the folder and percieves it as the path to the model")
    parser.add_argument("--model_hp_json", type=str, help="Path to model hyperparameters JSON file, if not included," \
        "looks for model_hyperparameters.json in the path folder")

    # Data paths
    parser.add_argument("--gpickle_users", type=str, default="data\\gg_users_full.gpickle", help="Path to user interaction graph (UUIG)")
    parser.add_argument("--gpickle_tweets", type=str, default="data\\gg_tweets_full.gpickle", help="Path to tweet interaction graph (TTIG)")
    parser.add_argument("--sim_matrix", type=str, default="data\\simMatrix_nx.gpickle", help="Path to similarity graph (UUSG)")
    parser.add_argument("--tweets_embeddings", type=str, default="data\\df_tweets_embeddings_bert.pickle", help="Path to tweet BERT embeddings (DataFrame)")
    parser.add_argument("--user_features", type=str, default="data\\df_user_feature_sets___full_enrich.pickle", help="Path to preprocessed user features")
    parser.add_argument("--split_pkl", type=str, default="data\\users_profile_split_temporal.pickle", help="Path to the split (tuple)")
    
    # Optional arguments
    parser.add_argument("--last_epoch", type=int, default=13, help="Last epoch to evaluate")
    parser.add_argument("--epochs_singular", action="store_true", help="Evaluate only last epoch")
    parser.add_argument("--save", action="store_true", help="Save metrics")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable progress bars and comments")

    args = parser.parse_args()

    # ===== Load inputs =====
    if args.path:
        path = args.path
    else:
        if (
            (os.path.isdir(os.path.join("models", args.model_name))) and 
            ("model-epoch_1.pt" in os.listdir(os.path.join("models", args.model_name)))
            ):
            path = os.path.join("models", args.model_name)
        else:
            raise ValueError("Variable path was not given and not found using default path")

    if args.model_hp_json:
        with open(args.model_hp_json, "r") as f:
            model_hyperparameters = json.load(f)
    else:
        if (
            (os.path.isdir(path)) and 
            ("model_hyperparameters.json" in os.listdir(path))
        ):
            json_path = os.path.join(path, "model_hyperparameters.json")
            with open(json_path, "r", encoding="utf-8") as f:
                model_hyperparameters = json.load(f)
        else:
            raise ValueError("Variable model_hp_json was not given and not found using default path")

    gpickle_users = load_pickle_custom(args.gpickle_users)
    gpickle_tweets = load_pickle_custom(args.gpickle_tweets)
    simMatrix_nx = load_pickle_custom(args.sim_matrix)
    split_pkl = load_pickle_custom(args.split_pkl)

    df_tweets_e_BERT = load_pickle_custom(args.tweets_embeddings)
    users_feature_df_preproc = load_pickle_custom(args.user_features)
    users_feature_df_preproc = preprocess(users_feature_df_preproc)

    test_ids = set(split_pkl[0][1]) 

    # ===== Prepare hyperparameters =====
    replacement_dict = create_replacement_dict(users_feature_df_preproc, HingeLoss=HingeLoss,SimpleCrossEntropyLoss=SimpleCrossEntropyLoss)
    model_hp = transform_json_to_dict(model_hyperparameters, replacement_dict)

    # ===== Run prediction =====
    prediction_table_per_epoch(
        model_hp=model_hp,
        model_name=args.model_name,
        path=path,
        users_feature_df_preproc=users_feature_df_preproc,
        gpickle_users=gpickle_users,
        gpickle_tweets=gpickle_tweets,
        simMatrix_nx=simMatrix_nx,
        df_tweets_e_BERT=df_tweets_e_BERT,
        test_ids=test_ids,
        last_epoch=args.last_epoch,
        epochs_singular=args.epochs_singular,
        save=args.save,
        tqdm_disable=args.disable_tqdm,
        simMatrix_nx_repl=None,
        raw_results_return=False
    )