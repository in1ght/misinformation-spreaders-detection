import os
import json
import argparse
import warnings

import pandas as pd
import networkx as nx

from tqdm import tqdm
from dict_operation import create_replacement_dict, transform_json_to_dict
from write_predictions import prediction_table_per_epoch
from sitif_data import load_pickle_custom
from train_process import SimpleCrossEntropyLoss, HingeLoss, preprocess

warnings.simplefilter(action='ignore', category=FutureWarning)

def save_all_metrics(
        path: str, 
        start_name: str, 
        last_epoch: int,
        epochs_singular: bool, 
        replacement_dict: dict, 

        gpickle_users: nx.DiGraph,
        gpickle_tweets: nx.DiGraph,
        simMatrix_nx: nx.DiGraph,
        df_tweets_e_BERT: pd.DataFrame,
        users_feature_df_preproc: pd.DataFrame,
        test_ids: set,

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

        gpickle_users - user-user-interaction graph with metadata such as missiformation spread (networkx.DiGraph)
        gpickle_tweets - tweet-tweet-interaction graph (networkx.DiGraph)
        simMatrix_nx - user-user-similarity binary graph created by class SimMatrixC (networkx.DiGraph)
        df_tweets_e_BERT - SentenceTransformer tweet embeddings (pd.DataFrame)
        users_feature_df_preproc - preprocessed user features (pd.DataFrame)
        test_ids - set of user ids that are used for the evaluation (set)

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
            
            # prediction_table_per_epoch(model_hp,file_item,folder_path,last_epoch,epochs_singular,True,True)
            prediction_table_per_epoch(
                model_hp=model_hp,
                model_name=file_item,
                path=folder_path,
       
                gpickle_users=gpickle_users,
                gpickle_tweets=gpickle_tweets,
                simMatrix_nx=simMatrix_nx,
                df_tweets_e_BERT=df_tweets_e_BERT,
                users_feature_df_preproc=users_feature_df_preproc,
                test_ids=test_ids,

                last_epoch=last_epoch,
                epochs_singular=epochs_singular,
                save=True,
                tqdm_disable=True,
                raw_results_return=False
            )


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
    parser = argparse.ArgumentParser(description="Run evaluation for multiple trained models")

    # Path to look for the models
    parser.add_argument("--path", type=str, default=".", help="Parent directory containing model folders")

    # Data paths
    parser.add_argument("--gpickle_users", type=str, default="data\\gg_users_full.gpickle", help="Path to user interaction graph (UUIG)")
    parser.add_argument("--gpickle_tweets", type=str, default="data\\gg_tweets_full.gpickle", help="Path to tweet interaction graph (TTIG)")
    parser.add_argument("--sim_matrix", type=str, default="data\\simMatrix_nx.gpickle", help="Path to similarity graph (UUSG)")
    parser.add_argument("--tweets_embeddings", type=str, default="data\\df_tweets_embeddings_bert.pickle", help="Path to tweet BERT embeddings (DataFrame)")
    parser.add_argument("--user_features", type=str, default="data\\df_user_feature_sets___full_enrich.pickle", help="Path to preprocessed user features")
    parser.add_argument("--split_pkl", type=str, default="data\\users_profile_split_temporal.pickle", help="Path to the split (tuple)")

    # Optional arguments
    parser.add_argument("--start_name", type=str, default="", help="Prefix of model folders to evaluate")
    parser.add_argument("--last_epoch", type=int, default=13, help="Last epoch to evaluate")
    parser.add_argument("--epochs_singular", action="store_true", help="Evaluate only last epoch")
    parser.add_argument("--show_progress", action="store_true", help="Enable progress bar")

    args = parser.parse_args()
    
    # ===== Load required data =====
    gpickle_users = load_pickle_custom(args.gpickle_users)
    gpickle_tweets = load_pickle_custom(args.gpickle_tweets)
    simMatrix_nx = load_pickle_custom(args.sim_matrix)
    split_pkl = load_pickle_custom(args.split_pkl)

    df_tweets_e_BERT = load_pickle_custom(args.tweets_embeddings)
    users_feature_df_preproc = load_pickle_custom(args.user_features)
    users_feature_df_preproc = preprocess(users_feature_df_preproc)

    replacement_dict = create_replacement_dict(users_feature_df_preproc, HingeLoss=HingeLoss, SimpleCrossEntropyLoss=SimpleCrossEntropyLoss)
    test_ids = set(split_pkl[0][1]) 

    # ===== Run function =====
    save_all_metrics(
        path=args.path,
        start_name=args.start_name,
        last_epoch=args.last_epoch,
        epochs_singular=args.epochs_singular,
        replacement_dict=replacement_dict,

        gpickle_users =gpickle_users,
        gpickle_tweets = gpickle_tweets,
        simMatrix_nx = simMatrix_nx,
        df_tweets_e_BERT = df_tweets_e_BERT,
        users_feature_df_preproc = users_feature_df_preproc,
        test_ids=test_ids,

        show_progress=args.show_progress
    )
