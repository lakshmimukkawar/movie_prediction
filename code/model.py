import pandas as pd
import logging
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
from sklearn.metrics import f1_score
import nltk
from ast import literal_eval
from data_processor import text_cleaner
import pickle

class model():
    """THis class is for applying models to our data."""
    def __init__(self, description, title, preprocessed_data_file_path, trained_model_present):
        """Init function for initializging description, title and prepreocessed data file path
        """
        self.description = description
        self.title = title
        self.preprocessed_data_file_path = preprocessed_data_file_path
        self.trained_model_present = trained_model_present

    def splitting_data_for_models(self, data, label):
        """This function is splitting the data in ttrain and test set.
        param:
            data: DataFrame: input dataframe for splitting
            label: numpy.array: labels for our classification model.

        return:
            xtrain: numpy.array: returns a xtrain array containing 80% of original data for training purpose.
            xval: numpy.arrray: returns a xval array containg 20% of original data for testing purpose.
            ytrain: numpy.array: return a ytrain array of labels for training.
            yval: numpy.array: returns a yval array of labels for testing.
        """
        xtrain, xval, ytrain, yval = train_test_split(data['extra_clean_description'], label, test_size=0.2, random_state=9)
        
        return xtrain, xval, ytrain, yval                   
    
    def create_tfidf_vectorizer(self, xtrain, xval):
        """This function is creating a tfidf vectorizer.
        param:
            xtrain: numpy.array: input array of columns for testing.
            xval: numpy.array: input array of columns for testing.

        return:
            xtrain_tfidf: returns a tfidf vectorizer for training data
            xval_tfidf: return a tdifd vectorizer for test data
            tfidf_vectorizer: returns a tfidf vectorizer
        """
        tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
        xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
        xval_tfidf = tfidf_vectorizer.transform(xval)
        # saving the tfidf vectorizer
        pickle.dump(tfidf_vectorizer, open("./saved_models/vectorizer.pickle", "wb"))
        return xtrain_tfidf, xval_tfidf, tfidf_vectorizer
    
    def save_model(self, model, filename):
        """This function is for saving the model on disk.
        param:
            model: class 'sklearn.multiclass.OneVsRestClassifier': input as a classifier
            filename: string: input filepath for saving the model
        """
        model_pkl = open(filename, 'wb')
        pickle.dump(model, model_pkl)
        model_pkl.close()
    
    def logistic_regression_model(self, xtrain_tfidf, ytrain, xval_tfidf, yval, trained_model_present):
        """This is a logistic regression model function.
        param:
            xtrain_tfidf: <class 'scipy.sparse.csr.csr_matrix'>: tfidf vectorier of training set
            ytrain: numpy.array: labels of training set
            xval_tfidf:  <class 'scipy.sparse.csr.csr_matrix'>: tfidf vectorier of testing set
            yval: numpy.array: labels of training set
            trained_model_present: boolean: this tells us if we have already saved our model on the local or not.
        """
        if trained_model_present:
            logistic_regression_pkl_file = "./saved_models/logistic_regression_model.pkl"
            logistic_model = self.open_saved_models_file(logistic_regression_pkl_file)
            return logistic_model, 0
        else:
            lr = LogisticRegression(solver='lbfgs')
            logistic_model = OneVsRestClassifier(lr)
            logistic_model.fit(xtrain_tfidf, ytrain)
            # save the model
            self.save_model(logistic_model, "./saved_models/logistic_regression_model.pkl")
            y_pred = logistic_model.predict(xval_tfidf)
            f11_score = self.get_f1_score(logistic_model, xval_tfidf, yval)
            return logistic_model, f11_score
        
    def random_forest_model(self, xtrain_tfidf, ytrain, xval_tfidf, yval, trained_model_present):
        """This is a random forest model function.
        param:
            xtrain_tfidf: <class 'scipy.sparse.csr.csr_matrix'>: tfidf vectorier of training set
            ytrain: numpy.array: labels of training set
            xval_tfidf:  <class 'scipy.sparse.csr.csr_matrix'>: tfidf vectorier of testing set
            yval: numpy.array: labels of training set
            trained_model_present: boolean: this tells us if we have already saved our model on the local or not.
        """
        if trained_model_present:
            random_forest_pkl_file  = "./saved_models/random_forest_model.pkl"
            forest_classifier = self.open_saved_models_file(random_forest_pkl_file)
            f1_scoree = 0
            return forest_classifier, 0
        else:
            n_estimators = 100
            forest_classifier = RandomForestClassifier(n_estimators = n_estimators, random_state=99)
            forest_classifier.fit(xtrain_tfidf,ytrain)
            self.save_model(forest_classifier, "./saved_models/random_forest_model.pkl")
            f1_scoree = self.get_f1_score(forest_classifier, xval_tfidf, yval)
            return forest_classifier, f1_scoree

    def open_saved_models_file(self, filename):
        """This function is used to open the saved models file.
        param:
            filename: string: input filepath for the saved model

        return:
            returns a classifier saved in that file.
        """
        with open(filename, 'rb') as file:
            classifiers = pickle.load(file)
        return classifiers
    
    def get_f1_score(self, classifierr, xval_tfidf, yval):
        """This function calculates the f1 score of  the model
        param:
            classifier: sklearn.multiclass.OneVsRestClassifier: classifier passed as an input.
            xval_tfidf: <class 'scipy.sparse.csr.csr_matrix'>: input tfiddf vectorized testing set.
            yval: numpy.array: input array of label for testing.

        return:
            returns f1_Score of the model.
        """
        y_pred = classifierr.predict(xval_tfidf)
        f11_score = f1_score(yval, y_pred, average="micro")
        return f11_score
    
    def get_multilable_binarizer(self, data):
        """This function gets the mulitilabel binarizer for our data.
        param:
            data: numpy.array: input array of label as in genres

        return:
            lable_binarizer: <class 'sklearn.multiclass.OneVsRestClassifier'>: returns a multilabel classifier.
            labels: list: returns a list of one hot encoded labels.
        """

        lable_binarizer = MultiLabelBinarizer()
        lable_binarizer.fit(data)
        labels = lable_binarizer.transform(data)
        # saving the multilabel binarizer
        pickle.dump(lable_binarizer, open("./saved_models/multilabel_binarizer.pickle", "wb"))
        return  lable_binarizer, labels
    
    def predictt(self, test_data, clf, tfidf_vectorizer, multilabel_binarizer):
        """This is the main function that gets called for prediction.
        param:
            test_data: string: input description taken from the command line arguments.
            clf: classifier for prediciton
            tfidf_vectorizer: tfidf vectorizer for converting the test_data to vectorize form before passing it to prediciton.
            multilabel_binarizer: multilabel binarizer for getting the actual type of the genre predicted.
        """
        
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        text_cleanerer = text_cleaner()
        test_data = text_cleanerer.text_cleaning(test_data)
        test_data = text_cleanerer.remove_stopwords_from_text(test_data, stop_words)
        test_data_vec = tfidf_vectorizer.transform([test_data])
        test_data_pred = clf.predict(test_data_vec)
        return multilabel_binarizer.inverse_transform(test_data_pred)
    
    def classification_model(self):
        """This is the main function from the model class which binds all the functions together. 
        return:
            ans: dict: returns a dict containing input passed via cmd and the prediciton value.
        """
        if self.trained_model_present:
            clf, f11_score = self.logistic_regression_model(np.arange(0), np.arange(0), np.arange(0), np.arange(0), self.trained_model_present)
            tfidf_vectorizer = self.open_saved_models_file("./saved_models/vectorizer.pickle")
            multilabel_binarizer = self.open_saved_models_file("./saved_models/multilabel_binarizer.pickle")
        else:
            movies_metadata = pd.read_csv(self.preprocessed_data_file_path, converters={"new_genres": literal_eval})
            movies_metadata["extra_clean_description"] = movies_metadata["extra_clean_description"].fillna(value="")
            multilabel_binarizer, labels  = self.get_multilable_binarizer(movies_metadata["new_genres"])

            xtrain, xval, ytrain, yval = self.splitting_data_for_models(movies_metadata, labels)
            xtrain_tfidf, xval_tfidf, tfidf_vectorizer = self.create_tfidf_vectorizer(xtrain, xval)
            clf, f11_score = self.logistic_regression_model(xtrain_tfidf, ytrain, xval_tfidf, yval, self.trained_model_present)

        pred = self.predictt(self.description, clf, tfidf_vectorizer, multilabel_binarizer)
      
        ans = ""
        if len(pred) == 1 and len(pred[0]) == 0:
            return "can't predict"
        elif len(pred) == 1 and len(pred[0]) == 1:
            ans = pred[0][0]
            return ans
        else:
            for item in pred:
                ans = ans + " " + item[0]
            return ans

    