from sklearn.ensemble import RandomForestRegressor

class DemandForecastModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)
        print("Model trained successfully with provided data.")

    def predict(self, X):
        return self.model.predict(X)