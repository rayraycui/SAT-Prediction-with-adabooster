import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

def convert_percentage_to_float(df, columns):
    for column in columns:
        if column in df and df[column].dtype == object:
            df[column] = df[column].str.rstrip('%').astype('float') / 100.0

training_set = pd.read_csv('~/project/TSA/SAT_FOR_TSA/Training.csv')
testing_set = pd.read_csv('~/project/TSA/SAT_FOR_TSA/Testing.csv')

percentage_columns = ['Percent White', 'Percent Black', 'Percent Hispanic', 'Percent Asian']
convert_percentage_to_float(training_set, percentage_columns)
convert_percentage_to_float(testing_set, percentage_columns)

training_set['AverageScore'] = training_set[['Average Score (SAT Math)', 'Average Score (SAT Reading)', 'Average Score (SAT Writing)']].mean(axis=1)
testing_set['AverageScore'] = testing_set[['Average Score (SAT Math)', 'Average Score (SAT Reading)', 'Average Score (SAT Writing)']].mean(axis=1)

training_set.dropna(subset=['Borough', 'AverageScore'] + percentage_columns, inplace=True)
testing_set.dropna(subset=['Borough', 'AverageScore'] + percentage_columns, inplace=True)

X_train = training_set[['Borough','Student Enrollment'] + percentage_columns]
y_train = training_set['AverageScore']
X_test = testing_set[['Borough','Student Enrollment'] + percentage_columns]
y_test = testing_set['AverageScore']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), percentage_columns),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), ['Borough'])
    ])
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)
model.fit(X_train_processed, y_train)

predictions = model.predict(X_test_processed)

accurate_predictions = sum(abs(actual - predicted) <= 50 for actual, predicted in zip(y_test, predictions))
total_predictions = len(predictions)
percentage_accuracy = (accurate_predictions / total_predictions) * 100

print(f"Percentage Accuracy (Within ±50 Points): {percentage_accuracy:.2f}%")

N = 10  
print("First 10 Predictions and Their Accuracy:")
for i in range(min(N, len(predictions))):
    actual = y_test.iloc[i]
    predicted = predictions[i]
    is_accurate = abs(actual - predicted) <= 50
    accuracy_text = "Accurate" if is_accurate else "Not Accurate"
    print(f"Prediction: {predicted:.2f}, Actual: {actual:.2f}, {accuracy_text}")
