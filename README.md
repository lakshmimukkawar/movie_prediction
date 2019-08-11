# movie_prediction
command line app for movie genre prediction.

### Installation guide:

1. Need python3.
2. Create a virtual environment using venv (please install venv from pip for creating environment)
3. python3 -m venv {name of env} for creating the environment
4. source {name of env}/bin/activate for activating that environment
5. git clone this repository and cd movie_prediction
6. go to code folder and install packages needed for it by pip install -r requirements.txt
7. Now for running our classifier we will classifier script.
8. ./classifier --title "othello" --description "The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic."
9. prediction will be shown via "final answer prediciton is ----> " statement. If the genre is "can't predict" that means that
our classifer has not learn that far to predict the genre of this particular movie description.
10. For running test cases, go back into the movie_prediction. run:- using python -m pytest
11. There is a dockerfile written for this app. This dockerfile is very basic one, can be improved alot.
12. For building an image, go into the folder where dockerfile is and run:- docker build -t movie_predictor:v1 .
13. For running the image, run:- docker run movie_predictor:v1

### Description:

1. I have used python3 for developing this command line application which takes title and description arguments.
2. Some validation is done on the arguments passed in the main.py file.
3. Really basic and simple libraries from python are used for training the model like pandas, sklearn, numpy, nltk, etc.
4. This is very exciting problem and has different approaches to implement.
5. First task is to preprocess the data and then apply models. As this is a text data we need to apply vectorizer on this.
I have chosen tfidf vectorizer which is really basic one but we need to experiment with different vecotrizer such as word2vec, Glove which might give us better results.
6. Random Forest classifier and Logistic Regression has been applied here because these will not take more time in training, its recommended to start applying some basic algorithms and then go ahead with complicated ones like neural networks. Again lots of experimentation can be done on the models and the hyperparameters.
7. The F1 score of the model is around 0.47 which is low but after doing some hyperparameterization it can be increased.
F1 score is used for model validation as to seek balance between precision and recall and it is used when there is a unbalanced data.
8. The original data is not balanced one. so we need to down sample or upsample it to get the good results.





