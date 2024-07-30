import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

file_path = 'CloudWatch_Traffic_Web_Attack.csv'  
data = pd.read_csv(file_path)

data['creation_time'] = pd.to_datetime(data['creation_time'])
data['end_time'] = pd.to_datetime(data['end_time'])

data['duration'] = (data['end_time'] - data['creation_time']).dt.total_seconds()

features = ['bytes_in', 'bytes_out', 'duration', 'protocol', 'response.code', 'dst_port']

data['protocol'] = data['protocol'].astype('category').cat.codes
data['response.code'] = data['response.code'].astype('category').cat.codes

# Normalize the features
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

X = data[features]

iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X)
data['anomaly'] = iso_forest.predict(X)
data['anomaly'] = data['anomaly'].apply(lambda x: 1 if x == -1 else 0)

input_dim = X.shape[1]
encoding_dim = 5

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

reconstructions = autoencoder.predict(X)
mse = np.mean(np.power(X - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 95)  
data['anomaly_autoencoder'] = [1 if e > threshold else 0 for e in mse]

# Evaluation
print("Isolation Forest Report")
print(classification_report(data['anomaly'], iso_forest.predict(X), zero_division=1))

print("Autoencoder Report")
print(classification_report(data['anomaly_autoencoder'], [1 if e > threshold else 0 for e in mse], zero_division=1))

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['bytes_in'], c=data['anomaly'], cmap='coolwarm')
plt.title('Anomalies Detected by Isolation Forest')
plt.xlabel('Index')
plt.ylabel('Bytes In')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['bytes_in'], c=data['anomaly_autoencoder'], cmap='coolwarm')
plt.title('Anomalies Detected by Autoencoder')
plt.xlabel('Index')
plt.ylabel('Bytes In')
plt.show()

# Distribution of packet sizes
plt.figure(figsize=(10, 6))
plt.hist(data['bytes_in'], bins=50, alpha=0.7, label='Bytes In')
plt.hist(data['bytes_out'], bins=50, alpha=0.7, label='Bytes Out')
plt.title('Distribution of Packet Sizes')
plt.xlabel('Packet Size')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Time intervals between packets
data['time_interval'] = data['creation_time'].diff().dt.total_seconds().fillna(0)
plt.figure(figsize=(10, 6))
plt.plot(data['creation_time'], data['time_interval'])
plt.title('Time Intervals Between Packets')
plt.xlabel('Time')
plt.ylabel('Time Interval (seconds)')
plt.show()

# Number of sessions per protocol
protocol_counts = data['protocol'].value_counts()
plt.figure(figsize=(10, 6))
protocol_counts.plot(kind='bar')
plt.title('Number of Sessions per Protocol')
plt.xlabel('Protocol')
plt.ylabel('Count')
plt.show()
