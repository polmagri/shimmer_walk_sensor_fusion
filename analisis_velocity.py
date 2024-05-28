import csv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

# Liste per memorizzare i dati importati
imported_timestamp1 = []
imported_velocity_norm1 = []
imported_timestamp2 = []
imported_velocity_norm2 = []

# Importa i dati dal primo file CSV
with open('velocità.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Salta l'intestazione
    for row in reader:
        imported_timestamp1.append(float(row[0]))
        imported_velocity_norm1.append(float(row[1]))

# Importa i dati dal secondo file CSV
with open('velocità2.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Salta l'intestazione
    for row in reader:
        imported_timestamp2.append(float(row[0]))
        imported_velocity_norm2.append(float(row[1]))

# Plot dei dati importati dal primo file
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(imported_timestamp1, imported_velocity_norm1, color='blue', label='Velocity Norm (File 1)')
plt.title('Velocity Norm (File 1)')
plt.xlabel('Timestamp 1')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()

# Plot dei dati importati dal secondo file
plt.subplot(1, 2, 2)
plt.plot(imported_timestamp2, imported_velocity_norm2, color='red', label='Velocity Norm (File 2)')
plt.title('Velocity Norm (File 2)')
plt.xlabel('Timestamp 2')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()




# Interpolazione dei dati del primo file
f1 = interp1d(imported_timestamp1, imported_velocity_norm1, kind='linear', fill_value='extrapolate')

# Interpolazione dei dati del secondo file
f2 = interp1d(imported_timestamp2, imported_velocity_norm2, kind='linear', fill_value='extrapolate')

# Definizione di un intervallo di tempo comune
start_time = max(imported_timestamp1[0], imported_timestamp2[0])
end_time = min(imported_timestamp1[-1], imported_timestamp2[-1])
common_timestamp = np.linspace(start_time, end_time, num=1000)

# Calcolo della velocità interpolata
interpolated_velocity_norm1 = f1(common_timestamp)
interpolated_velocity_norm2 = f2(common_timestamp)

# Plot dei dati interpolati
plt.plot(common_timestamp, interpolated_velocity_norm1, color='blue', label='Interpolated Velocity Norm (File 1)')
plt.plot(common_timestamp, interpolated_velocity_norm2, color='red', label='Interpolated Velocity Norm (File 2)')
plt.title('Interpolated Velocity Norm')
plt.xlabel('Timestamp')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()
plt.show()

# Somma delle velocità interpolate dei due piedi
total_velocity = interpolated_velocity_norm1 + interpolated_velocity_norm2

# Applica una media mobile alle velocità totali
window_size = 200  # Dimensione della finestra della media mobile
smoothed_total_velocity = uniform_filter1d(total_velocity, size=window_size)

# Plot della velocità totale e della media mobile
plt.plot(common_timestamp, total_velocity, color='blue', label='Total Velocity')
plt.plot(common_timestamp, smoothed_total_velocity, color='green', label=f'Smoothed Total Velocity (Window Size={window_size})')
plt.title('Total Velocity and Smoothed Total Velocity')
plt.xlabel('Timestamp')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()
plt.show()

# Applica il filtro di Savitzky-Golay alla velocità totale
smoothed_total_velocity = savgol_filter(total_velocity, window_length=251, polyorder=5)

# Plot della velocità totale e della velocità filtrata
plt.plot(common_timestamp, total_velocity, color='blue', label='Total Velocity')
plt.plot(common_timestamp, smoothed_total_velocity, color='green', label='Smoothed Total Velocity (Savitzky-Golay)')
plt.title('Total Velocity and Smoothed Total Velocity')
plt.xlabel('Timestamp')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()
plt.show()





