import pandas as pd

def engineer_features(data):
    # Example: Adding a feature that represents the tenure in years
    if 'Tenure' in data.columns:
        data['TenureYears'] = data['Tenure'] // 12
    return data
