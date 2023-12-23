# ML_Project
in this project we were give some data of IMDB, Amazon, Yelp reviwes with target output such that:
1- positive review
0- negative review
and our misiion is to train a model with these data to predict future reviews as negative or positive

## steps we followed
### data preprocessing phase
- we examined the distribution of samples in each class.
- we dropped unnecessary columns that will have no effect on target.
- we used spacy to eliminate stop words and perform lemmatization for each sentiment.
- we used Tf-idf as embedding technique.
- we split the data into training and testing sets to examine model perfomance.

### training phase
we used 2 techniques to train the model 
- Initial Experiment:
    we implemented sentiment classification using a Linear Support Vector Classifier (LinearSVC) with “Grid Search” to identify the optimal parameters for achieving the highest accuracy on the testing dataset.

- Subsequent Experiment:
    we used Artificial Neural Network (ANN) for classification
