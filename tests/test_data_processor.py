import pytest
import sys, os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join('..', 'code')))
from code.data_processor import text_cleaner, data_processor

def test_text_cleaning():
    cleaned_text = text_cleaner().text_cleaning("hi I am lakshmi \ who are you / Man")
    assert(cleaned_text == "hi i am lakshmi who are you man")

def test_remove_stopwords_from_text():
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    cleaned_text = text_cleaner().remove_stopwords_from_text("I am Lakshmi Mukkawar. I like nature.", stop_words)
    assert(cleaned_text == "I Lakshmi Mukkawar. I like nature.")


def test_get_list_of_genres_only():
    input_data = "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}, {'id': 10751, 'name': 'Family'}]"
    validation_data = ["Animation", "Comedy", "Family"]
    cleaned_list_of_genres = data_processor("./tests/test_file.csv").get_list_of_genres_only(input_data)
    assert(cleaned_list_of_genres == validation_data)

def test_data_cleaning():
    processor_instance = data_processor("./tests/test_file.csv")
    data = pd.read_csv("./tests/test_file.csv")
    cleaned_data = processor_instance.data_cleaning(data)
    cleaned_new_cols = list(cleaned_data.columns)
    validation_new_cols = ["id", "title", "description", "genres", "clean_description", "extra_clean_description"]
    assert(cleaned_new_cols == validation_new_cols)

def test_drop_duplicate_records():
    processor_instance = data_processor("./tests/test_file.csv")
    data = pd.read_csv("./tests/test_file.csv")
    unique_data = processor_instance.drop_duplicate_records(data, "id")
    assert(data.shape[0] != unique_data.shape[0])

