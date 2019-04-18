import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.fftpack import fft, fftfreq, ifft
import numpy as np
import wavio


datos = read("handel.wav")
frate = datos[0]
data = datos[1]

print(f'El archivo tiene una frecuencia de muestreo de {frate}')
# Cálculo del eje X, correspondiente al tiempo del grafico del data
n = len(data)  # Numero de intervalos
dt = 1 / frate  # Delta tiempo entre cada intervalo
largo = n * dt  # Tiempo total para generar n puntos de espaciado dt
vector_tiempo = np.linspace(0, largo, num=n)  # Generación del vector de tiempos

print(f'El tiempo entre cada punto es {dt}, el tiempo total es {largo}')
# Obtensión de la transformada
fft_out = fft(data)
frq = fftfreq(n, dt)  # Vector de frecuencias para la transformada (Eje X)


fft_norm = fft_out
#fft_norm = list(map(lambda x: x/n, fft_out))  # Se deberia escalar por n o por n/2?


deltaFrq = 0.1 # Al cambiar esto se seleccionan las frecuencias al rededor de la frecuencia de magnitud máxima

maxFrqIndex = np.argmax(fft_norm) # Índice frecuencia máxima
filterfrq = frq[maxFrqIndex] # Frecuencia máxima
limFrqInf = filterfrq - deltaFrq
limFrqSup = filterfrq + deltaFrq

print(f'La frecuencia central de filtrado es {filterfrq} de indice {maxFrqIndex}, se seleccionan las frecuencias entre {limFrqInf} y {limFrqSup} Hz')

# Filtración del % de frecuencias al rededor del máximo

fft_fixed = list(map(lambda x: 0 if (frq[x[0]] < limFrqInf or frq[x[0]] > limFrqSup ) else x[1], enumerate(np.abs(fft_norm))))


# Audio original
plt.figure(1)
plt.plot(vector_tiempo, data,'blue')
plt.title('Audio original')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
#plt.show()

# Transformada de Fourier
plt.figure(2)
plt.plot(frq, np.abs(fft_norm), 'r')

plt.title('Transformada de Fourier')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|F(w)|')
#plt.show()

# Audio transformada inversa
fft_inversa = ifft(fft_norm).real
plt.figure(3)
plt.plot(vector_tiempo, fft_inversa)
plt.title('Audio transformada inversa')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
#plt.show()

# Transformada truncada
plt.figure(4)
plt.plot(frq, np.abs(fft_fixed), 'g')
plt.title('Transformada de Fourier truncada')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|F(w)|')
#plt.show()

# Audio transformada inversa truncada
plt.figure(5)
fftinv = ifft(fft_fixed).real
plt.plot(vector_tiempo, fftinv, 'purple')
plt.title('Audio transformada inversa truncada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.show()

# write('handel_fftinv_forma1.wav', rate, fftinv) # Método 1 escritura en .wav
wavio.write("handel_inversa_fixed.wav", fftinv, frate, sampwidth=3) # Metodo 2 escritura en .wav

print('Programa finalizado con éxito')
