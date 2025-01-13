import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

# Simulated dataset: Financial news with sentiment labels
data = {
    "news": [
        "The company reports record profits this quarter.",
        "Market downturn leads to massive sell-offs.",
        "New product launch expected to boost revenues.",
        "CEO resigns amid controversy, stock plunges.",
        "Shares up as tech sector shows strong growth."
    ],
    "sentiment": [1, 0, 1, 0, 1]
}

# Convert to DataFrame
news_df = pd.DataFrame(data)

# Split into train/test sets
X = news_df['news']
y = news_df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)

# Convert text to sequences and pad them
max_length = 50
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# Build the LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=max_length),
    LSTM(64, return_sequences=True),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_test_pad, y_test),
    epochs=10,
    batch_size=32,
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Custom accuracy calculation
y_pred_nn = (model.predict(X_test_pad) > 0.5).astype("int32")
accuracy = np.mean(y_pred_nn.flatten() == y_test)
print(f"Custom Test Accuracy: {accuracy * 100:.2f}%")

# Classification report
print(classification_report(y_test, y_pred_nn, target_names=["Negative", "Positive"]))

# Fetch stock price data
stock_data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()
stock_data = stock_data.dropna()

# Generate sentiment signals for stock data
new_news = [
    "Stocks rally after positive earnings reports.",
    "Economic data suggests prolonged recession.",
    "Tech sector continues strong growth.",
    "Company faces lawsuits over data breach."
]
new_news_seq = tokenizer.texts_to_sequences(new_news)
new_news_pad = pad_sequences(new_news_seq, maxlen=max_length, padding='post', truncating='post')
predicted_sentiments = (model.predict(new_news_pad) > 0.5).astype("int32")

# Simulate daily sentiment (randomly aligning for demonstration purposes)
stock_data['Sentiment'] = np.random.choice([1, -1], size=len(stock_data))

# Define trading strategy
stock_data['Strategy'] = stock_data['Sentiment'] * stock_data['Daily Return']
stock_data['Cumulative Market Return'] = (1 + stock_data['Daily Return']).cumprod()
stock_data['Cumulative Strategy Return'] = (1 + stock_data['Strategy']).cumprod()

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Cumulative Market Return'], label='Market Return', color='blue')
plt.plot(stock_data['Cumulative Strategy Return'], label='Strategy Return', color='green')
plt.title('Cumulative Returns')
plt.legend()
plt.show()
