
import argparse
import model
from data_processor import data_processor
from model import model
import sys
import logging
import warnings
import os
import nltk

def get_args():
    """Argument parser.
    Returns:
        Dictionary of arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--title',
        type=str,
        required=True,
        help="Parameter tells the name of the movie."
    )
    parser.add_argument(
        '--description',
        type=str,
        required=True,
        help="Brief description about the movie."
    )
    args, extra_args = parser.parse_known_args()
    return args

def check_model_file_exists(filepath):
    """This function just check if the path exists or not. So if we have the model saved in our local then we will not run the training
    each time we pass new command line arguments. if model is not saved it will do the whole training.
    param:
        filepath: string: input filepath where the models are getting saved.
    return:
        returns a boolean yes or no for if there is a model saved on local or not.
    """
    return os.path.exists(filepath)

def main_function_start(movies_data_path):
    """Function for calling preprocess and model class.
    param: 
        movies_data_path: string: input is a filepath for the model data.

    return:
        final_answer: dict: returns a dict containing command line passed title and description and prediction as well.
    """
    nltk.download('stopwords')
    logistic_regression_pkl_file  = "./saved_models/logistic_regression_model.pkl"
    trained_model_present = check_model_file_exists(logistic_regression_pkl_file)
    if not trained_model_present:
        logging.info("Calling Data Processor for data preprocessing.")
        preprocessed_data_file_path = data_processor(movies_data_path).preprocess_data()
        logging.info("Data preprocessing is done.")
        prediction = model(args.title, args.description, preprocessed_data_file_path, trained_model_present).classification_model()

    else:
        prediction = model(args.title, args.description, "", trained_model_present).classification_model()

    final_answer = {}
    final_answer["title"] = args.title
    final_answer["description"] = args.description
    final_answer["genre"] = prediction
    warnings.simplefilter(action='ignore', category=FutureWarning)
    return final_answer

if __name__ == "__main__":
    fmtStr = "%(asctime)s: %(levelname)s: %(funcName)s Line:%(lineno)d %(message)s"
    dateStr = "%m/%d/%Y %I:%M:%S %p"
    logging.basicConfig(level=logging.DEBUG, format=fmtStr, datefmt=dateStr)

    logging.info("parsing the command line argumnets.")
    args = get_args()
    logging.info("Parsed attributes are Description = %s and Title = %s", args.description, args.title)
    movies_data_path = "./data/movies_metadata.csv"
    print("final answer prediciton is ----> ", main_function_start(movies_data_path))
    
    