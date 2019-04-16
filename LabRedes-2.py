import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.fftpack import fft, fftfreq, ifft
import numpy as np
import wavio


datos = read("handel.wav")
rate = datos[0]
data = datos[1]

# Cálculo del eje x, correspondiente al tiempo del grafico del data

n = len(data)  # Numero de intervalos
dt = 1 / rate  # Delta tiempo
largo = n * dt  # aqui saco hasta que largo se llega para generar n puntos de espaciado dt
vector_tiempo = np.linspace(0, largo, num=n) # se genera el vector de 0 a largo con n intervalos así cada intervalo queda con espaciado dt

# Obtensión de la transformada
fft_out = fft(data)
fft_norm = list(map(lambda x: x / n, fft_out))  # Se deberia escalar por n o por n/2?

frq = fftfreq(n, dt)  # frecuencias de la transformada (eje x)


# Obtension de la transformada filtrada
maxfrq = np.max(fft_norm)
filterfrq = np.abs(maxfrq * 0.15)

print('La frecuencia de filtrado es')
print(filterfrq)


# FILTRACIÓN DEL 15% DE LAS FRECUENCIAS

fft_fixed = list(map(lambda x: x if (x > filterfrq) else 0.0, np.abs(fft_norm)))


# AUDIO ORIGINAL
plt.figure(1)
plt.plot(vector_tiempo, data,'blue')
plt.title('Datos del audio original')
plt.xlabel('Tiempo [s]')
# plt.ylabel('aiuda')
plt.show()

# Transformada de Fourier

plt.figure(2)
plt.plot(frq, np.abs(fft_norm), 'r')

plt.title('Transformada de Fourier del audio')
plt.xlabel('Frecuencia [Hz]')
#plt.ylabel('amplitud')
plt.show()

# Transformada truncada
plt.figure(3)
plt.plot(frq, np.abs(fft_fixed), 'g')
plt.title('Transformada de Fourier truncada')
plt.xlabel('Frecuencia [Hz]')
#plt.ylabel('amplitud')
plt.show()

# AUDIO DESPUES DE TRUNCADO
plt.figure(4)
fftinv = ifft(fft_fixed)
fftinv2 = fftinv.real


plt.plot(vector_tiempo, fftinv, 'purple')
plt.title('Inversa de la transformada de Fourier truncada')
plt.xlabel('Tiempo [s]')
#plt.ylabel('aiuda')
plt.show()

write('handel_fftinv_forma1.wav', rate, fftinv2)
wavio.write("handel_inv_forma2.wav", fftinv, rate, sampwidth=3)

print('Programa finalizado con éxito')
