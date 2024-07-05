import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def get_models():
    base_models = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ]

    stacked_model = StackingClassifier(
        estimators=base_models,
        final_estimator=RandomForestClassifier(random_state=42)
    )

    return stacked_model

def get_pipeline(preprocessor, model):
    clf = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    return clf

def hyperparameter_tuning(pipeline, X_train, y_train):
    param_grid = {
        'classifier__rf__n_estimators': [100, 200],
        'classifier__rf__max_depth': [None, 10, 20],
        'classifier__gb__n_estimators': [100, 200],
        'classifier__gb__learning_rate': [0.01, 0.1, 0.2],
        'classifier__final_estimator__n_estimators': [100, 200],
        'classifier__final_estimator__max_depth': [None, 10, 20]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search
