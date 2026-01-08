import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model(input_file, model_save_path, seq_length=12, epochs=20, batch_size=32):
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # Assuming 'Vehicle_Count' is the target column
    data = df['Vehicle_Count'].values.reshape(-1, 1)
    
    # Normalize the data
    print("Normalizing data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    # seq_length = 12 (12 * 15min = 3 hours lookback)
    print(f"Creating sequences with length {seq_length}...")
    X, y = create_sequences(data_scaled, seq_length)
    
    # Split into train and test sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Build LSTM Model
    print("Building model...")
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    # Added 'mae' to metrics for "Accuracy" visualization
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    # Train
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    # Plot Training Metrics (Loss & MAE)
    plt.figure(figsize=(12, 5))

    # Subplot 1: Loss (MSE)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
    plt.title('Model Loss (Mean Squared Error)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Accuracy (MAE)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train Error (MAE)', color='orange')
    plt.plot(history.history['val_mae'], label='Val Error (MAE)', color='red')
    plt.title('Model Accuracy (Mean Absolute Error)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Training metrics plot saved to 'training_metrics.png'")
    
    # Evaluate
    print("Evaluating model...")
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss (MSE): {loss}")
    print(f"Test MAE: {mae}")
    
    # Save model
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
    
    # Predictions for visualization
    predictions = model.predict(X_test)
    predictions_inverse = scaler.inverse_transform(predictions)
    y_test_inverse = scaler.inverse_transform(y_test)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inverse, label='True Value')
    plt.plot(predictions_inverse, label='Predicted Value', alpha=0.7)
    plt.title('Traffic Density Prediction (Test Set)')
    plt.xlabel('Time Step')
    plt.ylabel('Vehicle Count')
    plt.legend()
    plt.savefig('prediction_results.png')
    print("Prediction plot saved to 'prediction_results.png'")
    
    # Save scaler for future inference
    import joblib
    joblib.dump(scaler, 'scaler.gz')
    print("Scaler saved to 'scaler.gz'")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, '..', 'processed_traffic_data.csv')
    model_path = 'traffic_lstm_model.h5'
    
    train_model(input_path, model_path)
