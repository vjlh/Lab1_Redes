import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.fftpack import fft, fftfreq, ifft, fftn
import numpy as np
import wavio


datos = read("handel.wav")
srate = datos[0]
data = datos[1]

print(f'El archivo tiene una frecuencia de muestreo de {srate}')
# Cálculo del eje X, correspondiente al tiempo del grafico del data
n = len(data)  # Numero de intervalos
dt = 1 / srate  # Delta tiempo entre cada intervalo
largo = n * dt  # Tiempo total para generar n puntos de espaciado dt
vector_tiempo = np.linspace(0, largo, num=n)  # Generación del vector de tiempos
print(f'Se toman {n} puntos en el archivo de entrada')
print(f'El tiempo entre cada punto es {dt} [s], el tiempo total es {largo} [s]')
# Obtensión de la transformada
fft_out = fft(data)
frq = fftfreq(n, dt)  # Vector de frecuencias para la transformada (Eje X)

# fft_norm = list(map(lambda x: x/n, fft_out))  # No se requiere una normalización
fft_norm = fft_out

deltaFrq = 1 # Al cambiar esto se seleccionan las frecuencias al rededor de la frecuencia de magnitud máxima
deltaFrq2 = 0.1 # Descomentar para prueba 2

maxFrqIndex = np.argmax(fft_norm) # Índice frecuencia máxima

filterfrq = frq[maxFrqIndex] # Frecuencia máxima
limFrqInf = filterfrq - deltaFrq
limFrqSup = filterfrq + deltaFrq
limFrqInf2 = filterfrq - deltaFrq2
limFrqSup2 = filterfrq + deltaFrq2

diffFrec = srate/n # Espacio diferencial del vector de frecuencias
puntos = 2*deltaFrq//diffFrec # Número de puntos tomados en el intervalo de filtrado
puntos2 = 2*deltaFrq2//diffFrec # Número de puntos tomados en el intervalo de filtrado

print(f'La frecuencia central de filtrado es {filterfrq} [Hz]')
print(f'Se seleccionan {puntos} frecuencia/s en un rango de {deltaFrq}[Hz] y {puntos2} frecuencia/s en un rango de {deltaFrq2} [Hz]')
print(f'Se seleccionan las frecuencias entre {limFrqInf} y {limFrqSup} [Hz] para la prueba 1')
print(f'Se seleccionan las frecuencias entre {limFrqInf2} y {limFrqSup2} [Hz] para la prueba 2')
# Filtración de las frecuencias al rededor del máximo
fft_fixed = list(map(lambda x: 0 if (frq[x[0]] < limFrqInf or frq[x[0]] > limFrqSup ) else 0.01*x[1], enumerate(np.abs(fft_norm))))
fft_fixed2 = list(map(lambda x: 0 if (frq[x[0]] < limFrqInf2 or frq[x[0]] > limFrqSup2 ) else 0.01*x[1], enumerate(np.abs(fft_norm))))

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

# Transformada truncada 1 [Hz]
plt.figure(4)
plt.plot(frq, np.abs(fft_fixed), 'g')
plt.title(f'Transformada de Fourier truncada {deltaFrq} [Hz]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|F(w)|')
#plt.show()

# Transformada truncada 0.1 [Hz]
plt.figure(5)
plt.plot(frq, np.abs(fft_fixed2), 'g')
plt.title(f'Transformada de Fourier truncada {deltaFrq2} [Hz]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|F(w)|')
#plt.show()

# Audio transformada inversa truncada prueba 1
plt.figure(6)
fftinv = ifft(fft_fixed).real
plt.plot(vector_tiempo, fftinv, 'purple')
plt.title(f'Audio transformada inversa truncada {deltaFrq} [Hz]')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')

# Audio transformada inversa truncada prueba 2
plt.figure(7)
fftinv2 = ifft(fft_fixed2).real
plt.plot(vector_tiempo, fftinv2, 'purple')
plt.title(f'Audio transformada inversa truncada {deltaFrq2} [Hz]')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')

plt.show()


# Audio transformada inversa sin filtrar
wavio.write("handel_inversa_original.wav", fft_inversa, srate, sampwidth=3) # Metodo 2 escritura en .wav

# Audio luego del filtrado
wavio.write("handel_inversa_fixed.wav", fftinv, srate, sampwidth=3) # Metodo 2 escritura en .wav
wavio.write("handel_inversa_fixed2.wav", fftinv2, srate, sampwidth=3) # Metodo 2 escritura en .wav

# En caso de que falle la escritura de archivo descomentar
# write('handel_inversa_original.wav', rate, fftinv) # Método 1 escritura en .wav
# write('handel_inversa_fixed.wav', rate, fftinv) # Método 1 escritura en .wav
# write('handel_inversa_fixed2.wav', rate, fftinv2) # Método 1 escritura en .wav

print('Programa finalizado con éxito')
