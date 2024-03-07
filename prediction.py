from helpers import extract_passwords_features, extract_person_features
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


class SvcModel:
    def __init__(self, first_folder_path, n=0):
        self.model = SVC(kernel="linear", probability=True, random_state=42)

        features = []
        labels = []
        for folder1 in os.listdir(first_folder_path):
            for file in os.listdir(first_folder_path + "/" + folder1):
                file = first_folder_path + "/" + folder1 + "/" + file
                if n == 0:
                    features.append(extract_passwords_features(file))
                else:
                    features.append(extract_person_features(file))
                labels.append(folder1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.2,
                                                                                random_state=42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy

    def my_classes(self):
        return self.model.classes_


class GbcModel:
    def __init__(self, first_folder_path, n=0):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)

        features = []
        labels = []
        for folder1 in os.listdir(first_folder_path):
            for file in os.listdir(first_folder_path + "/" + folder1):
                file = first_folder_path + "/" + folder1 + "/" + file
                if n == 0:
                    features.append(extract_passwords_features(file))
                else:
                    features.append(extract_person_features(file))
                labels.append(folder1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.2,
                                                                                random_state=42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy

    def my_classes(self):
        return self.model.classes_
