import numpy as np
import pandas as pd
import argparse
import pickle
import torch
import random_forest_impl
import dataset_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--dataset_dir', default='./Intent_Dataset', help='Location of the formed dataset')
    parser.add_argument('--model_dir', default='./models', help='Location to save the models')
    parser.add_argument('--model', default='random_forest', choices=['random_forest', 'svc','logistic_regression','nn'])

    parser.add_argument('--intent', default='What_are_the_available_modes_of_transport_in_this_state')
    parser.add_argument('--user_input', default='userInput_0')
    parser.add_argument('--random_seed', type=int, default=42)

    # Logistic Regression Optional Arguments:

    # Support Vector Classifier Optional Arguments:

    # Random Forest Optional Arguments:

    # Neural Network Optional Arguments:



    args = parser.parse_args()
    print(args)

    print("Training Model ...")
    if args.model == 'random_forest':
        random_forest_impl

