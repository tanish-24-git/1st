import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    required_columns = ['Historical_Sales', 'Promotion', 'Day_of_Week', 'Month', 'Product_ID', 'Demand']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check numeric columns
    numeric_columns = ['Historical_Sales', 'Promotion', 'Day_of_Week', 'Month', 'Demand']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                raise ValueError(f"Column '{col}' must be numeric")

    features = ['Historical_Sales', 'Promotion', 'Day_of_Week', 'Month', 'Product_ID']
    X = df[features]
    y = df['Demand']
    
    X = pd.get_dummies(X, columns=['Product_ID'], drop_first=True)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, scaler