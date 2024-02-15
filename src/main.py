import configparser
from dataloader import final_preprocessing
from pathlib import Path
import pandas as pd
import xgboost as xgb


# Load configuration
config = configparser.ConfigParser()
config.read('../config.ini')

# Paths
data_path = Path(config['PATHS']['data_path'])
relationship_data_path = Path(config['PATHS']['relationship_data_path'])
output_path = config['PATHS']['output_path']

# Parameters
drop_time = int(config['PARAMETERS']['drop_time'])
churn_time = int(config['PARAMETERS']['churn_time'])
recent_to = int(config['PARAMETERS']['recent_to'])
day_diff_to = int(config['PARAMETERS']['day_diff_to'])
total_days_to = int(config['PARAMETERS']['total_days_to'])
test_drop_time = int(config['PARAMETERS']['test_drop_time'])

# Data preprocessing
train_df = final_preprocessing(data_path, drop_time, churn_time, recent_to, day_diff_to, total_days_to)
test_df = final_preprocessing(data_path, test_drop_time, churn_time, recent_to, day_diff_to, total_days_to)

relationship_df = pd.read_excel(relationship_data_path)
relationship_encoded = pd.get_dummies(relationship_df, columns=['quali_relation'])

train_df = train_df.merge(relationship_encoded, on='client_id', how='left')
test_df = test_df.merge(relationship_encoded, on='client_id', how='left')

# Model training
X_train = train_df.drop('churn', axis=1)
y_train = train_df['churn']
X_test = test_df.drop('churn', axis=1)
y_test = test_df['churn']

model = xgb.XGBClassifier(colsample_bytree=0.7, learning_rate=0.1, max_depth=4, min_child_weight=4, n_estimators=200) # Best parameters from Grid search
model.fit(X_train, y_train)

# Predictions and output
test_df['predicted_churn'] = model.predict(X_test)
churned_clients = test_df[test_df['churn']==1]['client_id']
churned_clients.to_csv(output_path, index=False)
