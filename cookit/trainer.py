import joblib
from termcolor import colored

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cookit.data import get_data


class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pass

    def run(self):
        """fits model"""
        pass

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test"""

    def save_model_locally(self):
        """Save the model into a .joblib format"""

        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))


if __name__ == "__main__":
    # Get and clean data
    df = get_data(nrows=1000)

    y = df["classes"]
    X = df.drop("classes", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Train and save model, locally and
    trainer = Trainer(X_train, y_train)
    trainer.set_experiment_name('xp2')
    trainer.run()
    score = trainer.evaluate(X_test, y_test)
    print(f"Score of model : {score}")

    trainer.save_model_locally()
