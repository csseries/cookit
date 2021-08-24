import joblib
from cookit.data import get_data

class Predictor():

    def __init__(self):
        """
        A basic call for predictions.
        """
        self.model = self._get_model()

    def _get_model(self, path_to_joblib='model.joblib'):
        #pipeline = joblib.load(path_to_joblib)
        #return pipeline
        return None

    def predict(self, X_test):
        print(f"Received file for prediction: {X_test}")
        #return self.model.predict(X_test)
        return ['Cucumber, Carrot, Garlic, Butter, Toast']


if __name__ == '__main__':
    df_test = get_data('path_to_test_data')
    predictor = Predictor()
    y_pred = predictor.predict(df_test)

    print(f"Prediction: {y_pred}")
