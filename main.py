import data_preprocessing as dp
import feature_engineering as fe
import model_training as mt
import model_evaluation as me

# Load the data
data = dp.load_data('customer_churn_data.csv')

# Feature engineering
data = fe.engineer_features(data)

# Define features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Preprocess data
preprocessor = dp.preprocess_data(X)

# Split data
X_train, X_test, y_train, y_test = mt.split_data(X, y)

# Get models
stacked_model = mt.get_models()

# Get pipeline
pipeline = mt.get_pipeline(preprocessor, stacked_model)

# Hyperparameter tuning
grid_search = mt.hyperparameter_tuning(pipeline, X_train, y_train)

# Evaluate model
me.evaluate_model(grid_search, X_test, y_test)
