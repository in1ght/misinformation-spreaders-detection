import torch
import warnings
import os
import json

import torch.nn as nn
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

def transform_json_to_dict(
        json_dict: dict, 
        replacement_dict: dict
    ):
    """
    Converts hyperparameters from a JSON dict only with strings to a functions-compatable Python dictionary.

    Args:
        json_dict - JSON of strings (dict)
        replacement_dict - dictionary mapping placeholders to objects (dict)

    Returns:
        dict:
            functions-compatable Python dictionary
    """
    # Hardcoded and inplace
    # 1 - user_feat - example: [true, "all_data_no_stats"]
    # validation check - I assume it is list
    if not isinstance(json_dict['user_feat'][0],bool):
        json_dict['user_feat'] = [bool(json_dict['user_feat']),(replacement_dict[json_dict['user_feat']] if json_dict['user_feat'] else [])]
    else:
        json_dict['user_feat'][1] = replacement_dict[json_dict['user_feat'][1]] if json_dict['user_feat'][1] else []
    # 2 - ui_and_us_X - example: "tree_metric"
    json_dict['ui_and_us_X'] = replacement_dict[json_dict['ui_and_us_X']]
    # 3 - optimizer - example: ["torch.optim.AdamW",0.0005]
    json_dict['optimizer'][0] = replacement_dict[json_dict['optimizer'][0]]
    # 4 - loss
    json_dict['loss'] = replacement_dict[json_dict['loss']]
    # 5 - if no threshold - add one
    json_dict.setdefault('threshold', 0.5)

    return json_dict

def transform_json_to_dict_configuration(
        json_dict: dict, 
        replacement_dict: dict
    ):
    """
    Converts hyperparameters from a JSON dict only with strings to a functions-compatable Python dictionary.
    
    This is an option for configuration specifically. Was not added during the research process, hence to keep
    compatability and the original cversion in tact, this temporaly creates a solution.

    Args:
        json_dict - JSON of strings (dict)
        replacement_dict - dictionary mapping placeholders to objects (dict)

    Returns:
        dict:
            functions-compatable Python dictionary
    """
    # Hardcoded and inplace
    # 1 - user_feat - example: [true, "all_data_no_stats"]
    if 'user_feat' in json_dict:
        # validation check - I assume it is list
        for i, hp_configuration in enumerate(json_dict['user_feat']):
            if not isinstance(hp_configuration[0],bool):
                json_dict['user_feat'][i] = replacement_dict.get(hp_configuration, [])
            else:
                json_dict['user_feat'][i] = replacement_dict.get(hp_configuration[1], [])
                
    # 2 - ui_and_us_X - example: "tree_metric"
    if 'ui_and_us_X' in json_dict:
        for i, hp_configuration in enumerate(json_dict['ui_and_us_X']):
            json_dict['ui_and_us_X'][i] = replacement_dict[hp_configuration]
            
    # 3 - optimizer - example: ["torch.optim.AdamW",0.0005]
    if 'optimizer' in json_dict:
        for i, hp_configuration in enumerate(json_dict['optimizer']):
            json_dict['optimizer'][i][0] = replacement_dict[hp_configuration[0]]
            
    # 4 - loss
    if 'loss' in json_dict:
        for i, hp_configuration in enumerate(json_dict['loss']):
            json_dict['loss'][i] = replacement_dict[hp_configuration]

    return json_dict

def create_replacement_dict(
        users_feature_df_preproc: pd.DataFrame,
        HingeLoss: callable,
        SimpleCrossEntropyLoss: callable
    ):
    """
    Constructs a replacement dictionary used to convert hyperparameters from a JSON dict only with strings 
    to a functions-compatable Python dictionary.

    Args:
        users_feature_df_preproc - preprocessed user features (pd.DataFrame)

    Returns:
        dictionary:
            dictionary mapping placeholders to objects.
    """
    # Existing feature sets
    user_stats_tree_metrics = (
        [item for item in users_feature_df_preproc.columns if item.startswith("tree")] +
        [item for item in users_feature_df_preproc.columns if item.startswith("user_stats_")] +
        ['statuses_count', 'favorites_count', 'follower_count', 'friends_count', 'listed_count']
    )

    content_evol_tree_metrics = (
        [item for item in users_feature_df_preproc.columns if item.startswith("nn_")] +
        [item for item in users_feature_df_preproc.columns if item.startswith("tree")]
    )

    tree_metric = [item for item in users_feature_df_preproc.columns if item.startswith("tree")]

    us_tm_ce = user_stats_tree_metrics + [item for item in users_feature_df_preproc.columns if item.startswith("nn_")]

    tweet_stats = [item for item in users_feature_df_preproc.columns if item.startswith("stats_")]

    all_data_no_stats = [col for col in users_feature_df_preproc.columns if col not in tweet_stats]

    content_evol_tree_metrics_user_stats = user_stats_tree_metrics + [item for item in users_feature_df_preproc.columns if item.startswith("nn_")]

    # Constant dictionary mapping names to feature sets
    # Feature sets
    feature_dict = {
        'user_stats_tree_metrics': user_stats_tree_metrics,
        'content_evol_tree_metrics': content_evol_tree_metrics,
        'tree_metric': tree_metric,
        'us_tm_ce': us_tm_ce,
        'tweet_stats': tweet_stats,
        'all_data_no_stats': all_data_no_stats,
        'content_evol_tree_metrics_user_stats': content_evol_tree_metrics_user_stats
    }

    # Optimizers - 2 names possibe
    optimizer_dict = {
        'torch.optim.AdamW': torch.optim.AdamW,
        'AdamW': torch.optim.AdamW,
        'torch.optim.Adam': torch.optim.Adam,
        'Adam': torch.optim.Adam
    }

    # Losses - 2 names possibe
    loss_dict = {
        'HingeLoss': HingeLoss,
        'torch.nn.BCELoss': torch.nn.BCELoss,
        'BCELoss': torch.nn.BCELoss,
        'SimpleCrossEntropyLoss': SimpleCrossEntropyLoss
    }

    # Combined replacement dictionary
    replacement_dict = {**feature_dict, **optimizer_dict, **loss_dict}

    return replacement_dict


def load_hp_json_custom(
        given_path: str,
        default_name: str
    ):
    """
    Loads a JSON hyperparameters file from a given path.

    Args:
        given_path - path to the json (str)
        default_name - default file name to look for in the folder (str)

    Returns:
        Loaded JSON object as a dictionary (dict)
    """
    if given_path:
        path_to_load = given_path
    else:
        path_to_load = os.path.join(".", default_name)
        if not os.path.isfile(path_to_load):
            raise ValueError(f"File {default_name} not found in base folder and no path provided.")

    with open(path_to_load, "r", encoding="utf-8") as f:
        return json.load(f)
    

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

def transform_dict_to_json(json_dict: dict, replacement_dict: dict) -> dict:
    """
    Converts hyperparameters from a functions-compatable Python dictionary to a JSON dict
    using names from a replacement mapping.

    This is essentially the reverse of transform_json_to_dict.

    Args:
        json_dict - functions-compatable Python dictionary (dict)
        replacement_dict - dictionary mapping placeholders to objects (dict)

    Returns:
        dict:
            JSON dict only with strings
    """

    # Build reverse mapping, convert lists to tuples
    reverse_dict = {
        tuple(v) if isinstance(v, list) else v: k
        for k, v in replacement_dict.items()
        if isinstance(v, (str, int, float, bool, type, list))
    }

    def reverse_lookup(val):
        """
        Look up the original key for a value, supporting lists as tuples.
        """
        key = reverse_dict.get(tuple(val) if isinstance(val, list) else val, val)
        return key

    # 1 - user_feat - [bool, object/list]
    user_feat_val = json_dict['user_feat']
    if not isinstance(user_feat_val[0], bool):
        # Handle case where first element is not bool
        json_dict['user_feat'] = [
            bool(user_feat_val[0]),
            reverse_lookup(user_feat_val[1]) if user_feat_val[1] else []
        ]
    else:
        json_dict['user_feat'][1] = reverse_lookup(user_feat_val[1]) if user_feat_val[1] is not None else []

    # 2 - ui_and_us_X
    json_dict['ui_and_us_X'] = reverse_lookup(json_dict['ui_and_us_X'])

    # 3 - optimizer - [object, lr]
    json_dict['optimizer'][0] = reverse_lookup(json_dict['optimizer'][0])

    # 4 - loss
    json_dict['loss'] = reverse_lookup(json_dict['loss'])

    # 5 - threshold default
    json_dict.setdefault('threshold', 0.5)

    return json_dict

def is_json_serializable(obj):
    """
    Checks if an object (dict) is JSON serializable (i.e. if I can save it as JSON)

    Returns:
        True if serializable, False if not.
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False
