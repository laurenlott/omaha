import pandas as pd
import numpy as np
from numpy.random import randn
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import pickle
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# Load the dataset and handle missing values
pbp = pd.read_csv('/Users/lauren/Downloads/epa_plays.csv', index_col=0)
pbp = pbp.dropna(subset=['quarter', 'down', 'distance', 'yards_to_go', 'play_type', 'regular_play_type', 'seconds_left_in_quarter'])

# Taking out the rows that a play did not occur in
pbp = pbp[pbp.no_play != 1]

pbp = pbp[['quarter', 'down', 'distance', 'yards_to_go', 'play_type', 'regular_play_type', 'seconds_left_in_quarter']]

pbp.loc[pbp['play_type'] != 'S']

X = pbp[['quarter', 'down', 'distance', 'yards_to_go', 'seconds_left_in_quarter']]
y = pbp['regular_play_type']

# XGBoost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Label encoding for target variable y
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Creating an XGBoost classifier
model = XGBClassifier()

# Fitting the classifier to the training data
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)

# Calculating and printing the accuracy score
accuracy = metrics.accuracy_score(y_test, y_pred)
print("XGBoost Accuracy:", accuracy)

# Save the XGBoost model to a file on your desktop
desktop_path = "/Users/lauren/Desktop/Projects"

with open(desktop_path + "/model.pickle", 'wb') as f:
    pickle.dump(model, f)

# Saving the XGBoost Model
# Saving model to a file
with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)

# Loading the saved model
with open('model.pickle', 'rb') as f:
    loaded_model = pickle.load(f)
    
# Modify the label encoding to map 'R' to 0 and 'P' to 1
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Create a dictionary to map the encoded labels back to the original labels
label_map = {0: 'R', 1: 'P'}

# Get the column names in the same order as the training data
user_input_cols = X_train.columns.tolist()

# Function to get the user's input and make predictions
def get_user_input(quarter, down, distance, yards_to_go, seconds_left_in_quarter):
    # Making a prediction using the loaded model
    user_input = [[quarter, down, distance, yards_to_go, seconds_left_in_quarter]]

    # Create a DataFrame with user input and use it for prediction
    user_input_df = pd.DataFrame(user_input, columns=user_input_cols)
    prediction_encoded = loaded_model.predict(user_input_df)[0]
    
    # Reversing label encoding and using the label_map
    prediction = label_map[prediction_encoded]

    # Returning prediction as the output
    return prediction

# Create the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Play Prediction App"),
    html.Div([
        html.Label("Quarter:"),
        dcc.Input(id="quarter-input", type="number", value=1),
    ]),
    html.Div([
        html.Label("Down:"),
        dcc.Input(id="down-input", type="number", value=1),
    ]),
    html.Div([
        html.Label("Distance:"),
        dcc.Input(id="distance-input", type="number", value=10),
    ]),
    html.Div([
        html.Label("Yards to Go:"),
        dcc.Input(id="yards-to-go-input", type="number", value=5),
    ]),
    html.Div([
        html.Label("Seconds Left in Quarter:"),
        dcc.Input(id="seconds-left-input", type="number", value=300),
    ]),
    html.Br(),
    html.Button("Predict", id="predict-button", n_clicks=0),
    html.Br(),
    html.Div(id="output")
])

# Callback to handle prediction
@app.callback(
    Output("output", "children"),
    [Input("predict-button", "n_clicks")],
    [dash.dependencies.State("quarter-input", "value"),
     dash.dependencies.State("down-input", "value"),
     dash.dependencies.State("distance-input", "value"),
     dash.dependencies.State("yards-to-go-input", "value"),
     dash.dependencies.State("seconds-left-input", "value")]
)
def predict_play(n_clicks, quarter, down, distance, yards_to_go, seconds_left_in_quarter):
    prediction = get_user_input(quarter, down, distance, yards_to_go, seconds_left_in_quarter)
    return f"Model Prediction: {prediction}"

if __name__ == '__main__':
    app.run_server(debug=False)


