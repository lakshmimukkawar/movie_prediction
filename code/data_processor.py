import pandas as pd 
import ast
import re
import logging

class data_processor():
    """ Class data_preprocessor helps in preprocessing the data for out movie predictor classifier.
    """
    def __init__(self, movies_data_path):
        """Initialzation function for class. 
        param:
            movies_data_path: string: input data file for building a classifier.ss
        """
        self.movies_data_path = movies_data_path

    def preprocess_data(self):
        """ This is the main instance method which gets called from the class object. It reads data, clean it, removes duplicates, etc.
        return:
            preprocessed_data_filename: string: return a filename path where the preprocessed file is kept.
        """
        movies_metadata = pd.read_csv(self.movies_data_path)
        required_columns = ["id", "title", "overview", "genres"]
        movies_metadata =  movies_metadata[required_columns]
        movies_metadata.rename(columns={"overview":"description"}, inplace=True)
        movies_metadata =  self.drop_duplicate_records(movies_metadata, "id")
        movies_metadata = movies_metadata[movies_metadata.description.notnull()]
        movies_metadata = movies_metadata[movies_metadata.genres != "[]"]
        movies_metadata["new_genres"] = movies_metadata["genres"].apply(lambda x: self.get_list_of_genres_only(x))
        movies_metadata = self.data_cleaning(movies_metadata)
        preprocessed_data_filename = "./data/preprocessed_data.csv"
        logging.info("Saving a preprocessed data into csv file at location %s", preprocessed_data_filename)
        movies_metadata.to_csv(preprocessed_data_filename, index = False)
        logging.info("Saving preprocessed data into csv file complete.")
        return preprocessed_data_filename

    def data_cleaning(self, data):
        """This function takes the data and apply some text cleaning methods like removing stopwords, removing punctuations, etc.
        Param:
            data: DataFrame: input is dataframe containing title and descriptions column.

        return:
            data: DataFrame: returns a cleaned dataframe containing columns like clean_description and extra_clean_description.
        """
        from nltk.corpus import stopwords
        logging.info("Data cleaning by removing stop words and removing special characters, punctuations, etc.")
        text_cleanerer = text_cleaner()
        data["clean_description"] = data["description"].apply(lambda x: text_cleanerer.text_cleaning(x))
        stop_words = set(stopwords.words('english'))
        data["extra_clean_description"] = data["clean_description"].apply(lambda x: text_cleanerer.remove_stopwords_from_text(x, stop_words))
        logging.info("Data Cleaning step done.")
        return data

    
    def get_list_of_genres_only(self, data):
        """This function helps in rearragning the genres list in the data. main genres column contains a dict with id and genre.
        but we want just the list of genres not the ids.
        param:
            data: string: input is a string of genres

        return:
            genres_list: List: returns a list contaning genres.
        """
        genres_list = []
        data = data[1: -1]
        data = ast.literal_eval(data)
        if type(data) is tuple:
            for genre in data:
                genres_list.append(genre["name"])
        else:
            genres_list.append(data["name"])
        return genres_list

    def drop_duplicate_records(self, data, param):
        """This is function for removing duplicates.
        param: 
            data: DataFrame: input dataframe from which we want to remove duplicates.
            param: string: parameter for which we are checking duplicates in the dataframe

        return:
            returns DataFrame without duplicates.
        """
        return data.drop_duplicates(subset=param)

class text_cleaner():
    """Thies text_cleaner class is for cleaning the data.
    """
    def text_cleaning(self, data):
        """This function actually removes punctuations, etc from data.
        param:
            data: string: input string which we want to clean.

        return:
            data: string: returns the string with cleaner version.
        """
        data = re.sub("\/'", "", data)
        data = re.sub("[^a-zA-Z]", " ", data)
        data = " ".join(data.split())
        data = data.lower()
        return data

    def remove_stopwords_from_text(self, text, stop_words):
        """This function helps in removing the stop words from the text.
        param:
            text: string: input text 
            stop_words: list: list of english stop words

        return:
            returns a text without stopwords.
        """
        text_wo_stopwords = [word for word in text.split() if not word in stop_words]
        return ' '.join(text_wo_stopwords)
