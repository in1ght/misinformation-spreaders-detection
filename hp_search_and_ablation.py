import torch
import copy
import networkx
import argparse
import os
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
from functools import partial

from predictors import GCN, PredictorABLATION
from sitif_data import FNSDDataset, custom_collate, load_pickle_custom
from dict_operation import create_replacement_dict, transform_json_to_dict, transform_json_to_dict_configuration, load_hp_json_custom, transform_dict_to_json, is_json_serializable
from train_process import HingeLoss, SimpleCrossEntropyLoss, checkpoint, hp_to_str, train, preprocess, get_weights, make_json_serializable

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def hyperparameter_tunning(
        model_hyperparameters: dict,
        part_change: str,
        hyperparameters: list,

        gpickle_users: networkx.DiGraph,
        gpickle_tweets: networkx.DiGraph,
        simMatrix_nx: networkx.DiGraph,
        df_tweets_e_BERT: pd.DataFrame,
        users_feature_df_preproc: pd.DataFrame,
        weights: np.ndarray,
        device: torch.device,
        train_ids: set,
        test_ids: set,

        keep_noted: bool = False, 
        previous_run: list = None,
        checkpoint_str = False,
        threshold: float = 0.5
    ):
    """
    Performs a grid search over a specific hyperparameter of the model, keeping all other hyperparameters fixed.

    An example of the structure of model_hyperparameters is:

    {"user_inter": [True,[32, 32, 32]],"user_sim": [True,[32]],"tweets_inter": [True,[64,16]],
    "user_feat": [True,all_data_no_stats], "ui_and_us_X": list(tree_metric),"fnn": [32, 64, 32], 
    "optimizer": [torch.optim.AdamW,0.0005], "loss": HingeLoss, "max_epochs": 20, "batch": 12, 
    'early_stopping_tl': 5, "mha": [30,3]}

    An example of the part_change structure is:
    part_change = "user_sim" with hyperparameters = [[32,32],[64,32],[64,64],[32,16],[32],[64]]
    This will run a grid hyperparameter search on user_sim part of the original model, replacing it
    with the values givem in the list.

    Args:
        model_hyperparameters - dictionary of model hyperparameters and overall setup (dict)
        part_change - part of the original model_hyperparameters to perform hyperparameters search on (str)
        hyperparameters - hyperparameters to test (list)

        gpickle_users - user-user-interaction graph with metadata such as missiformation spread (networkx.DiGraph)
        gpickle_tweets - tweet-tweet-interaction graph (networkx.DiGraph)
        simMatrix_nx - user-user-similarity binary graph created by class SimMatrixC (networkx.DiGraph)
        df_tweets_e_BERT - SentenceTransformer tweet embeddings (pd.DataFrame)
        users_feature_df_preproc - preprocessed user features (pd.DataFrame)
        weights - a ndarray (or tensor) containing weights for handling class imbalance in loss (np.ndarray)
        
        device - computation device (torch.device)
        train_ids, test_ids - identifiers for training and testing (validation) samples
        
        keep_noted - if True, prints all of the updates (bool)
        previous_run - previous results for comparrison in results (list)
        threshold - thresholds to split the data. Common is 0,5 and binary, can be list for multiple-choice setting (float or list)

    Returns:
        Tuple:
            res - list of results for each hyperparameter: [best_loss, final_epoch, tested_hyperparameter]
            best_result - entry from res with the lowest best_loss
    """
    
    # Ablation hyperparameters must also include False/True
    checkpoint_part = partial(checkpoint)
    if part_change in ['user_inter','user_sim','tweets_inter','user_feat']:
        if not isinstance(hyperparameters[0][0], bool):
            hyperparameters = [[True,item] for item in hyperparameters]
    res = []
    if previous_run:
        res.append(previous_run)
    tbar = tqdm(hyperparameters)
    for i in tbar:
        initil_tune = copy.deepcopy(i)
        current_model_hp = copy.deepcopy(model_hyperparameters)
        current_model_hp[part_change] = i

        # Leave the possibility to simply send the list of features to include
        # instead of the wholde datasets as input.
        if isinstance(current_model_hp['user_feat'][1], list):
            current_model_hp['user_feat'][1] = users_feature_df_preproc[current_model_hp['user_feat'][1]]
        if isinstance(current_model_hp['ui_and_us_X'], list):
            current_model_hp['ui_and_us_X'] = users_feature_df_preproc[current_model_hp['ui_and_us_X']]

        uf_len = len(current_model_hp['user_feat'][1].columns)
        uius_len = len(current_model_hp['ui_and_us_X'].columns)

        parts = {"user_inter": current_model_hp['user_inter'][0],"user_sim": current_model_hp['user_sim'][0],
                 "tweets_inter": current_model_hp['tweets_inter'][0],"user_feat": current_model_hp['user_feat'][0]}

        ds_train = FNSDDataset(gpickle_users,gpickle_tweets,simMatrix_nx,current_model_hp['user_feat'][1],df_tweets_e_BERT,len(current_model_hp['user_inter'][1]),
                               len(current_model_hp['user_sim'][1]),len(current_model_hp['tweets_inter'][1]),threshold,
                               train_ids,parts = parts, us_X = current_model_hp['ui_and_us_X'], ui_X = current_model_hp['ui_and_us_X']) # SET TO TRAIN + VAL IDS
        ds_val = FNSDDataset(gpickle_users,gpickle_tweets,simMatrix_nx,current_model_hp['user_feat'][1],df_tweets_e_BERT,len(current_model_hp['user_inter'][1]),
                               len(current_model_hp['user_sim'][1]),len(current_model_hp['tweets_inter'][1]),threshold,
                               test_ids,parts = parts, us_X = current_model_hp['ui_and_us_X'], ui_X = current_model_hp['ui_and_us_X']) # SET TO TEST IDS
        
        dl_train = torch.utils.data.DataLoader(ds_train,batch_size=current_model_hp['batch'],collate_fn=custom_collate,shuffle=True)
        dl_val = torch.utils.data.DataLoader(ds_val,batch_size=20,collate_fn=custom_collate,shuffle=False)

        model = PredictorABLATION(current_model_hp['user_inter'][1],current_model_hp['user_sim'][1],current_model_hp['tweets_inter'][1],
                                  current_model_hp['fnn'],uf_len,uius_len,uius_len,len(df_tweets_e_BERT.columns),current_model_hp['mha'][0],
                                  current_model_hp['mha'][1],GCN,parts)
        
        model.to(device)

        optimizer = current_model_hp['optimizer'][0](model.parameters(), lr=current_model_hp['optimizer'][1])
        
        if (current_model_hp['loss'] == torch.nn.BCELoss):
            loss = current_model_hp['loss']().to(device=device)
        else:
            loss = current_model_hp['loss']()

        model_name = hp_to_str(checkpoint_str,i) if checkpoint_str else "default"
        _, best_loss, final_epoch = train(model, device, dl_train, optimizer, loss, current_model_hp['max_epochs'], 
                                        weights, checkpoint = checkpoint_part if bool(checkpoint_str) else False, 
                                        name_model = model_name, validation=dl_val, 
                                        early_stopping_tl=current_model_hp['early_stopping_tl'],keep_noted=keep_noted)

        res.append([best_loss,final_epoch,initil_tune])

        if part_change == 'user_feat':
            tbar.set_postfix(hyperparameters=len(initil_tune[1]),last_res=best_loss,best_res=min(res, key = lambda initil_tune : initil_tune[0])[0])
        elif part_change == 'ui_and_us_X':
            tbar.set_postfix(hyperparameters=len(initil_tune),last_res=best_loss,best_res=min(res, key = lambda initil_tune : initil_tune[0])[0])
        else:
            tbar.set_postfix(hyperparameters=initil_tune,last_res=best_loss,best_res=min(res, key = lambda initil_tune : initil_tune[0]))
        
        if checkpoint_str:
            # ==== if it was modified ====
            current_model_hp_save = copy.deepcopy(model_hyperparameters)
            current_model_hp_save[part_change] = initil_tune
            # ==== if it was modified ====
            json_model = transform_dict_to_json(current_model_hp_save, replacement_dict)
            code_save = -1
            if not is_json_serializable(json_model):
                code_save = -2
                json_model = {"Error":"JSON was not serializable, could not save the JSON hyperparameters"}
                print("\n\nWARNING: JSON was not serializable, could not save the JSON hyperparameters!\n\n")
            checkpoint(json_model, code_save, model_name) # saving

    return res, min(res, key = lambda i : i[0])



def all_hyperparameters_tunning(
        ideal_model: dict,
        hp_tunes: dict, 

        gpickle_users: networkx.DiGraph,
        gpickle_tweets: networkx.DiGraph,
        simMatrix_nx: networkx.DiGraph,
        df_tweets_e_BERT: pd.DataFrame,
        users_feature_df_preproc: pd.DataFrame,
        weights: np.ndarray,
        device: torch.device,
        train_ids: set,
        test_ids: set,

        keep_noted: bool = False,
        previous_ideal_model: list = None, 
        save_checkpoints: bool = False, 
        improve: bool = True,
        threshold: float = 0.5
    ):
    """
    Performs hyperparameter tuning for all specified parameters in the model. 
    Uses hyperparameter_tunning function. For a clearer picture, please relate to it's description.

    An example of hp_tunes:
    hp_tunes = {
        'mha':[[30,2],[24,2],[24,3]],
        'fnn':[[32, 64, 32],[64,32]]
    }
    
    Optimized for avoiding redundant evaluations, saves intermediate checkpoints if `save_checkpoints=True`.
    Updates `ideal_model` with the best-performing hyperparameters if needed.

    Args:
        ideal_model - dictionary of current best hyperparameters (dict)
        hp_tunes - dictionary of hyperparameter lists to test and corresponding part names (dict)

        gpickle_users - user-user-interaction graph with metadata such as missiformation spread (networkx.DiGraph)
        gpickle_tweets - tweet-tweet-interaction graph (networkx.DiGraph)
        simMatrix_nx - user-user-similarity binary graph created by class SimMatrixC (networkx.DiGraph)
        df_tweets_e_BERT - SentenceTransformer tweet embeddings (pd.DataFrame)
        users_feature_df_preproc - preprocessed user features (pd.DataFrame)
        weights - a ndarray (or tensor) containing weights for handling class imbalance in loss (np.ndarray)
        device - computation device (torch.device)
        train_ids, test_ids - identifiers for training and testing (validation) samples

        keep_noted - if True, prints all of the updates (bool)
        previous_ideal_model - optional previous results to skip already computed runs (list)
        save_checkpoints - if True, saves checkpoints for each hyperparameter (bool)
        improve - if True, updates ideal_model with best results after tuning (bool)
        threshold - thresholds to split the data. Common is 0,5 and binary, can be list for multiple-choice setting (float or list)

    Returns:
        Tuple:
            ideal_model - updated dictionary with best hyperparameters found (dict)
            results_extra - list of results for each hyperparameter:
                            [parameter_name, list of results from hyperparameter_tunning]
    """
    # do not put in values that are already in the ideal model, as the evaluation will
    # be then performed twice for this. Just the new. The model does compare it to the
    # old hyperparameters itself.

    first_key = next(iter(hp_tunes))
    if not (ideal_model[first_key] in hp_tunes[first_key]) and not previous_ideal_model:
        if isinstance(ideal_model[first_key], list) and isinstance(ideal_model[first_key][0], bool):
            hp_tunes[first_key].append(ideal_model[first_key][1])
        else:
            hp_tunes[first_key].append(ideal_model[first_key])

    

    # with the code above basically.
    tbar =  tqdm(hp_tunes.items())
    results_extra = []
    # Approximate waiting time 20 minutes per run (1/3 hour) - hardcoded for 
    # Acer Predator PH-315-52 (Intel i7-9750H CPU, NVIDIA GeForce GTX 1660 Ti GPU)
    overal_len = sum(len(v) for v in hp_tunes.values())
    # The loop for hypertunning itself

    for k, v in tbar:

        if ((ideal_model[k] in v) and (first_key != k)): # to skip if the value is accidently duplicated
            v.remove(ideal_model[k]) # so that the model does not have to run the same value twice by accident - saves a lot of time
        if previous_ideal_model: # i.e. if it's a second run or the old results are given
            previous_ideal_model[2] = previous_ideal_model[2][k] # saves one run

        remaining_hours = int(overal_len/3)
        remaining_minutes = int(((overal_len/3)-remaining_hours)*60)
        tqdm.write(f"Starting tuning for '{k}' - approx time left: {remaining_hours}h {remaining_minutes}m")

        #best_res, best_res_k = hyperparameter_tunning(ideal_model,k,v,keep_noted,previous_run = previous_ideal_model, checkpoint_str = k if save_checkpoints else False)
        best_res, best_res_k = hyperparameter_tunning(ideal_model,k,v,gpickle_users,gpickle_tweets,simMatrix_nx,
                                                      df_tweets_e_BERT,users_feature_df_preproc,weights,device,
                                                      train_ids,test_ids,keep_noted,previous_run = previous_ideal_model, 
                                                      checkpoint_str = k if save_checkpoints else False, threshold = threshold)
        # always a case unless stated otherwise (for final results after successful hyperparameterization)
        if improve:
            ideal_model[k] = best_res_k[2]

        overal_len -= len(v)
        remaining_hours = int(overal_len/3)
        remaining_minutes = int(((overal_len/3)-remaining_hours)*60)
        tbar.set_postfix(hyperparameters=k,best_loss=best_res_k[0],approx_waiting_time=f"{remaining_hours}h: {remaining_minutes}m")
        print(f"####\tThe result for {k} is:\t{best_res_k[2]} with loss: {best_res_k[0]}\t####")

        results_extra.append([k,best_res])
        previous_ideal_model = [best_res_k[0],best_res_k[1],ideal_model]
        
    return ideal_model,results_extra

def load_hp_json(given_path: str, folder: str, default_name: str) -> dict:
    """
    Load a JSON hyperparameters file. If `given_path` is provided, load from it.
    Otherwise, look for `default_name` inside `folder`. Raises ValueError if not found.
    """
    if given_path:
        path_to_load = given_path
    else:
        path_to_load = os.path.join(folder, default_name)
        if not os.path.isfile(path_to_load):
            raise ValueError(f"File {default_name} not found in {folder} and no path provided.")

    with open(path_to_load, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for a model")

    # Required paths / identifiers
    parser.add_argument("--model_hp_json", type=str, default="grid_search\\model_hyperparameters.json",
        help="Path to model hyperparameters JSON file, if not included, looks for model_hyperparameters.json in the folder by the name " \
        "of model_name. To support partial hyperparameter search, a default model to start hyperparameter search from is always required.")
    parser.add_argument("--model_hp_configurations", type=str, default="grid_search\\model_hyperparameters_configurations.json",
        help="Path to model hyperparameter configurations JSON file, if not included, looks for model_hyperparameters_configurations.json "
        "in the folder by the name of model_name. Please, look for the task description in the python file for the structure of the required file")
    
    # Data paths
    parser.add_argument("--gpickle_users", type=str, default="data\\gg_users_full.gpickle", help="Path to user interaction graph (UUIG)")
    parser.add_argument("--gpickle_tweets", type=str, default="data\\gg_tweets_full.gpickle", help="Path to tweet interaction graph (TTIG)")
    parser.add_argument("--sim_matrix", type=str, default="data\\simMatrix_nx.gpickle", help="Path to similarity graph (UUSG)")
    parser.add_argument("--tweets_embeddings", type=str, default="data\\df_tweets_embeddings_bert.pickle", help="Path to tweet BERT embeddings (DataFrame)")
    parser.add_argument("--user_features", type=str, default="data\\df_user_feature_sets___full_enrich.pickle", help="Path to preprocessed user features")
    parser.add_argument("--split_pkl", type=str, default="data\\users_profile_split_temporal.pickle", help="Path to the split (tuple)")

    # Optional hyperparameters / training settings
    parser.add_argument("--val_size_of_train", type=float, default=0, help="Optional. NOTE: If provided, code will use only the training part of the split and utilize " \
        "a fraction of the training data for validation. Otherwise (if NOT given), will use the test part of the split for the evaluation.")
    parser.add_argument("--threshold", type=float, nargs='+', default=[0.5], help="Threshold(s) for class prediction. Single float = binary, list = multiclass")
    parser.add_argument("--keep_noted", action="store_true", help="Print updates during training")
    parser.add_argument("--save_checkpoints", action="store_true", help="Save checkpoints for intermediate hyperparameter runs")
    parser.add_argument("--store_results", action="store_true", help="If stated, will save the best model and additional data")
    parser.add_argument("--improve", action="store_true", help="Update ideal_model with best found results (This can be used in combination with " \
    "save_checkpoints to create models with different configuration, and then run them through the write_predictions -> test all models in a folder.)")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # ===== Load inputs =====
    model_hyperparameters = load_hp_json_custom(args.model_hp_json, "model_hyperparameters.json")
    model_hp_configurations = load_hp_json_custom(args.model_hp_configurations, "model_hyperparameters_configurations.json")

    gpickle_users = load_pickle_custom(args.gpickle_users)
    gpickle_tweets = load_pickle_custom(args.gpickle_tweets)
    simMatrix_nx = load_pickle_custom(args.sim_matrix)
    split_pkl = load_pickle_custom(args.split_pkl)

    df_tweets_e_BERT = load_pickle_custom(args.tweets_embeddings)
    users_feature_df_preproc = load_pickle_custom(args.user_features)
    users_feature_df_preproc = preprocess(users_feature_df_preproc)

    save_checkpoints = args.save_checkpoints
    improve = args.improve
    keep_noted = args.keep_noted
    store_results = args.store_results
    checkpoint_p = partial(checkpoint)
    device = torch.device(args.device)

    # ===== Load ids and choose the split =====
    train_ids = sorted(split_pkl[0][0]) 
    if args.val_size_of_train == 0:
        test_ids = split_pkl[0][1]
    else:
        if not (0 < args.val_size_of_train < 1):
            raise ValueError("val_size_of_train must be between 0 and 1 (0:1), 0 and 1 excluded")
        split_tr_vl = int(len(train_ids)*(1-args.val_size_of_train))
        train_ids = set(train_ids[:split_tr_vl])
        test_ids = set(train_ids[split_tr_vl:])

    # ===== Prepare hyperparameters =====
    replacement_dict = create_replacement_dict(users_feature_df_preproc, HingeLoss=HingeLoss, SimpleCrossEntropyLoss=SimpleCrossEntropyLoss)
    model_hp = transform_json_to_dict(model_hyperparameters, replacement_dict)
    hp_tunes = transform_json_to_dict_configuration(model_hp_configurations, replacement_dict)

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
        users_given=train_ids,
        ui_metadata=gpickle_users,
        users_feature_df_preproc = users_feature_df_preproc,
        device=device
    )

    # ===== Run hyperparameter search =====
    ideal_model, results_extra = all_hyperparameters_tunning(
        model_hp,
        hp_tunes,
        gpickle_users,
        gpickle_tweets,
        simMatrix_nx,
        df_tweets_e_BERT,
        users_feature_df_preproc,
        weights,
        device,
        train_ids,
        test_ids,
        keep_noted, 
        None,
        save_checkpoints,
        improve,
        threshold
    )

    print(f"\n\nThe best model found hyperparameter search:\n{ideal_model}\n\n")
    print(f"List of results for each model, giving a more detailed :\n{results_extra}\n\n")

    if store_results:
        with open("best_model.json", "w", encoding="utf-8") as f:
            json.dump(make_json_serializable(ideal_model), f, indent=4)
        
        with open("additional_results.json", "w", encoding="utf-8") as f:
            json.dump(make_json_serializable(results_extra), f, indent=4)

        print(f"Both outputs saved to the same folder, where the python file is.")
