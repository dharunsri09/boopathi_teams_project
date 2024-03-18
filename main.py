import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor

# Step 1: Prepare the data
# Load Input dataset
df = pd.read_csv('btc.csv')

print("=======\nDataset\n=======\n",df)

# Filter data for Bitcoin Cryptocurrency
df = df[df['Cryptocurrency'] == 'Bitcoin']
print("=============\nBitcoin Dataset\n=============\n",df)

# Filter data for Dash Cryptocurrency
#df = df[df['Cryptocurrency'] == 'Dash']
#print("=============\nDash Dataset\n=============\n",df)

# Filter data for Litecoin Cryptocurrency
#df = df[df['Cryptocurrency'] == 'Litecoin']
#print("=============\nLitecoin Dataset\n=============\n",df)

# Filter data for Bitcoin-Cash Cryptocurrency
#df = df[df['Cryptocurrency'] == 'Bitcoin-Cash']
#print("=============\nBitcoin-Cash Dataset\n=============\n",df)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Calculate sentiment score for each CoinDeskTweets using TextBlob library
df['Sentiment'] = df['CoinDeskTweets'].apply(lambda x: TextBlob(x).sentiment.polarity)

print("\n=================================================\nDataset after Sentiment Score Calculation\n=================================================\n",df)

# Step 2: Train machine learning classifiers
# Select features and target
X = df[['Sentiment']]
y = df['Closing Price (in $)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#fasfadfasdf
# Printing training dataset
print("\nTraining Dataset:")
print(X_train,"\n")
print(y_train)

# Printing testing dataset
print("\nTesting Dataset:")
print(X_test,"\n")
print(y_test,"\n")

def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'Model': model_name, 'MSE': mse, 'MAE': mae, 'R2': r2}, y_test, y_pred

# Predict stack price based on sentiment using Linear Regression
lr_results, lr_actual, lr_predicted = evaluate_model(LinearRegression(), 'LSTM')

# Compare models using performance metrics
results_df = pd.DataFrame([lr_results])

# Plotting the comparison
plt.figure(figsize=(10, 6))
metrics = ['MSE', 'MAE', 'R2']
for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    plt.bar(results_df['Model'], results_df[metric])
    plt.title(metric)
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Displaying the comparison table
print(results_df)

# Displaying actual and predicted values for each model
models = ['Linear Regression']
for model_name, actual, predicted in zip(models, [lr_actual], [lr_predicted]):
    print(f"\n{model_name} Results:")
    results = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    print(results)


# Step 4: Visualize the results
# Plot Date vs Predicted Stock Price for each Cryptocurrency
plt.figure(figsize=(10, 6))
for Cryptocurrency in df['Cryptocurrency'].unique():
    df_Cryptocurrency = df[df['Cryptocurrency'] == Cryptocurrency]
    plt.plot(df_Cryptocurrency['Date'], df_Cryptocurrency['Closing Price (in $)'], label=Cryptocurrency)
plt.title('Date vs Predicted Stock Price')
plt.xlabel('Date')
plt.ylabel('Closing Price (in $)')
plt.legend()
plt.show()

# Plot Date vs Sentiment for each Cryptocurrency
plt.figure(figsize=(10, 6))
for Cryptocurrency in df['Cryptocurrency'].unique():
    df_Cryptocurrency = df[df['Cryptocurrency'] == Cryptocurrency]
    plt.bar(df_Cryptocurrency['Date'], df_Cryptocurrency['Sentiment'], label=Cryptocurrency)
plt.title('Date vs Sentiment')
plt.xlabel('Date')
plt.ylabel('Sentiment')
plt.legend()
plt.show()
