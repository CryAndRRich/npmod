import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class NaiveBayesData():
    def __init__(self, data_path):
        self.DATA = pd.read_csv(data_path, lineterminator='\n')

        self.X_NUMPY, self.Y_NUMPY = self.DATA['text'].to_numpy(), self.DATA['label\r'].to_numpy()
        self.X_NUMPY_TRAIN, self.X_NUMPY_TEST, self.Y_NUMPY_TRAIN, self.Y_NUMPY_TEST = train_test_split(self.X_NUMPY, self.Y_NUMPY, test_size=0.2, random_state=42)

        vectorizer = CountVectorizer()
        self.X_NUMPY_TRAIN = vectorizer.fit_transform(self.X_NUMPY_TRAIN).toarray()
        self.X_NUMPY_TEST = vectorizer.transform(self.X_NUMPY_TEST).toarray()

    def get_numpy_data(self):
        return self.X_NUMPY_TRAIN, self.X_NUMPY_TEST, self.Y_NUMPY_TRAIN, self.Y_NUMPY_TEST
    

