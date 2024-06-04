import tkinter as tk
from tkinter import Label, Entry, Button, StringVar, OptionMenu
import matplotlib.pyplot as plt
import os
import cv2
import random
from natsort import natsorted
import numpy as np
import math

# Parámetros iniciales
intervalo = [2, 8]  # Intervalo de ejemplo, se sobrescribe con la UI
Resolución = 0.05
poblacion_inicial = 10
cantidad_generaciones = 20
posibilidad_de_cruza = 50
posibilidad_de_mutacion_individuo = 10
posibilidad_de_mutacion_gen = 60
limite_poblacion = 30

# Estadísticas para graficar
maximos = []
minimos = []
generaciones = []
promedios = []

# Directorio para guardar las imágenes
carpeta_imagenes = 'Imagenes'

# Función objetivo
def f(x):
    return x * np.cos(x)  # Función objetivo x * cos(x)

# Utilidad para convertir binario a decimal
def bin_to_decimal(binario):
    return int(binario, 2)

# Calcula el rango del intervalo
def calculo_rango(intervalo):
    a, b = intervalo
    return b - a

# Calcula los bits necesarios
def calcular_bits(resultado_rango, Resolución):
    num_saltos = (resultado_rango / Resolución) + 1
    n = 1
    while not (2**(n - 1) <= num_saltos <= 2**n):
        n += 1
    return n

# Calcula Delta X
def Delta_x(resultado_rango, puntos_bits):
    return resultado_rango / (2**puntos_bits - 1)

# Genera la población inicial
def generar_poblacion_inicial(bits, poblacion_inicial):
    poblacion = [''.join(random.choice('01') for _ in range(bits)) for _ in range(poblacion_inicial)]
    print("Población Inicial Creada:")
    print("Ejemplos de individuos iniciales:", poblacion[:3])  # Muestra los primeros 3 para no saturar la consola
    return poblacion

# Ajusta la población si excede el límite
def ajustar_poblacion(poblacion, limite_poblacion):
    if len(poblacion) > limite_poblacion:
        print(f"Ajustando población de {len(poblacion)} a {limite_poblacion}")
        return poblacion[:limite_poblacion]
    return poblacion

# Cruza dos individuos
def cruza(individuo1, individuo2):
    punto = random.randint(1, len(individuo1) - 1)
    hijo1 = individuo1[:punto] + individuo2[punto:]
    hijo2 = individuo2[:punto] + individuo1[punto:]
    print(f"Cruzando: {individuo1} y {individuo2} en punto {punto}")
    print(f"Obtenidos: {hijo1}, {hijo2}")
    return hijo1, hijo2

# Muta un individuo
def mutacion(individuo, prob_mutacion_gen):
    individuo_mutado = ''.join(bit if random.randint(1, 100) > prob_mutacion_gen else '1' if bit == '0' else '0' for bit in individuo)
    if individuo != individuo_mutado:
        print(f"Mutado: {individuo} a {individuo_mutado}")
    return individuo_mutado

# Selección y reproducción
def seleccion_y_reproduccion(poblacion, prob_cruza, prob_mutacion_individuo, prob_mutacion_gen, opcion_max_min, puntos_bits):
    print("\nProcesando Selección y Reproducción")
    nueva_poblacion = []
    valores_funcionales = [bin_to_decimal(individuo) * Delta_x(calculo_rango(intervalo), puntos_bits) + intervalo[0] for individuo in poblacion]
    aptitudes = [f(x) for x in valores_funcionales]
    
    print(f"Valores Funcionales de algunos individuos: {valores_funcionales[:3]}")
    print(f"Aptitudes de algunos individuos: {aptitudes[:3]}")
    
    # Ajuste para asegurar no negatividad en aptitudes
    if min(aptitudes) < 0:
        aptitudes = [apt + abs(min(aptitudes)) + 1 for apt in aptitudes]
    
    # Evitar division por cero en total_aptitud
    total_aptitud = sum(aptitudes)
    if total_aptitud == 0:
        prob_seleccion = [1/len(aptitudes)] * len(aptitudes)
    else:
        prob_seleccion = [apt / total_aptitud for apt in aptitudes]

    prob_seleccion = np.clip(prob_seleccion, 0, 1)
    prob_seleccion /= np.sum(prob_seleccion)
    
    seleccionados = np.random.choice(poblacion, size=len(poblacion), p=prob_seleccion)
    print(f"Individuos seleccionados para cruza y mutación: {seleccionados[:3]}")
    
    for i in range(0, len(seleccionados), 2):
        if random.randint(1, 100) < prob_cruza:
            hijo1, hijo2 = cruza(seleccionados[i], seleccionados[i+1])
            nueva_poblacion.append(hijo1)
            nueva_poblacion.append(hijo2)
        else:
            nueva_poblacion.append(seleccionados[i])
            nueva_poblacion.append(seleccionados[i+1])

    nueva_poblacion = [mutacion(individuo, prob_mutacion_gen) if random.randint(1, 100) < prob_mutacion_individuo else individuo for individuo in nueva_poblacion]
    
    print(f"Retornando nueva población de tamaño: {len(nueva_poblacion)}")
    return nueva_poblacion[:len(poblacion)]

# Mostrar gráficas de evolución con puntos de los mejores y peores individuos
def mostrar_graficas(maximos, minimos, promedios, generaciones, mejores, peores):
    for gen in range(len(generaciones)):
        plt.figure(figsize=(12, 6))

        # Graficar la función objetivo f(x) en un rango de valores
        x = np.linspace(intervalo[0], intervalo[1], 400)
        y = f(x)
        plt.plot(x, y, label='f(x)', color='blue')

        # Marcar el mejor y peor individuos en la gráfica para la generación actual
        mejor_x = mejores[gen]
        peor_x = peores[gen]
        mejor_y = f(mejor_x)
        peor_y = f(peor_x)

        plt.scatter(mejor_x, mejor_y, color='green', marker='o', s=100, label='Mejor')
        plt.scatter(peor_x, peor_y, color='red', marker='x', s=100, label='Peor')

        plt.title(f'Evolución del Algoritmo Genético - Generación {gen + 1}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        
        # Guarda cada gráfica con un nombre único
        plt.savefig(f'{carpeta_imagenes}/grafica_generacion_{gen + 1}.png')
        plt.close()

# Función para mostrar el resumen de las gráficas de los mejores, peores y promedios
def mostrar_resumen_graficas(maximos, minimos, promedios, generaciones):
    plt.figure(figsize=(12, 6))
    
    # Graficar el mejor, peor y promedio de aptitudes
    plt.plot(generaciones, maximos, label='Mejor', color='green', marker='o')
    plt.plot(generaciones, minimos, label='Peor', color='red', marker='x')
    plt.plot(generaciones, promedios, label='Promedio', color='blue', marker='s')

    plt.title('Resumen de la Evolución del Algoritmo Genético')
    plt.xlabel('Generación')
    plt.ylabel('Aptitud')
    plt.legend()
    plt.grid(True)
    plt.show()

# Crea un video a partir de imágenes generadas
def crear_video(carpeta_imagenes, nombre_video='video.avi', fps=2):
    imagenes = natsorted([img for img in os.listdir(carpeta_imagenes) if img.endswith(".png")])
    if not imagenes:
        print("No hay imágenes para generar el video.")
        return
    frame = cv2.imread(os.path.join(carpeta_imagenes, imagenes[0]))
    altura, ancho, capas = frame.shape
    video = cv2.VideoWriter(nombre_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (ancho, altura))
    
    for imagen in imagenes:
        video.write(cv2.imread(os.path.join(carpeta_imagenes, imagen)))
    
    cv2.destroyAllWindows()
    video.release()
    print("Vídeo creado y guardado como", nombre_video)

# Función principal del algoritmo genético
def main():
    print("Iniciando el Algoritmo Genético")
    print(f"Intervalo: {intervalo}")
    print(f"Resolución: {Resolución}")
    print(f"Población Inicial: {poblacion_inicial}")
    print(f"Cantidad de Generaciones: {cantidad_generaciones}")
    print(f"Probabilidad de Cruza: {posibilidad_de_cruza}%")
    print(f"Probabilidad de Mutación de Individuo: {posibilidad_de_mutacion_individuo}%")
    print(f"Probabilidad de Mutación de Gen: {posibilidad_de_mutacion_gen}%")
    print(f"Límite de Población: {limite_poblacion}")
    print(f"Función Objetivo: f(x) = x * cos(x)")

    # Verifica si la carpeta de imágenes existe y si no, la crea
    if not os.path.exists(carpeta_imagenes):
        os.makedirs(carpeta_imagenes)
    
    # Limpia las imágenes de ejecuciones previas
    eliminar_imagenes(carpeta_imagenes)

    resultado_rango = calculo_rango(intervalo)
    puntos_bits = calcular_bits(resultado_rango, Resolución)
    Calculo_DeltaX = round(Delta_x(resultado_rango, puntos_bits), 4)
    poblacion = generar_poblacion_inicial(puntos_bits, poblacion_inicial)

    mejores = []
    peores = []

    for gen in range(cantidad_generaciones):
        print(f"\nGeneración {gen + 1}/{cantidad_generaciones}")
        poblacion = seleccion_y_reproduccion(poblacion, posibilidad_de_cruza, posibilidad_de_mutacion_individuo, posibilidad_de_mutacion_gen, opcion_max_min, puntos_bits)
        poblacion = ajustar_poblacion(poblacion, limite_poblacion)
        
        valores_funcionales = [bin_to_decimal(individuo) * Calculo_DeltaX + intervalo[0] for individuo in poblacion]
        aptitudes = [f(x) for x in valores_funcionales]
        maximos.append(max(aptitudes))
        minimos.append(min(aptitudes))
        promedios.append(sum(aptitudes) / len(aptitudes))
        generaciones.append(gen)

        print(f"Mejor aptitud: {maximos[-1]}")
        print(f"Peor aptitud: {minimos[-1]}")
        print(f"Promedio de aptitud: {promedios[-1]}")

        mejor_ind = np.argmax(aptitudes) if opcion_max_min == "maximizar" else np.argmin(aptitudes)
        peor_ind = np.argmin(aptitudes) if opcion_max_min == "maximizar" else np.argmax(aptitudes)
        mejores.append(valores_funcionales[mejor_ind])
        peores.append(valores_funcionales[peor_ind])
    
    mostrar_graficas(maximos, minimos, promedios, generaciones, mejores, peores)
    mostrar_resumen_graficas(maximos, minimos, promedios, list(range(1, cantidad_generaciones + 1)))
    crear_video(carpeta_imagenes)
    print("Proceso completado. Gráficas y vídeo generados.")

# Actualiza las variables globales con los valores de la interfaz
def actualizar_variables_globales():
    global intervalo, Resolución, poblacion_inicial, cantidad_generaciones
    global posibilidad_de_cruza, posibilidad_de_mutacion_individuo, posibilidad_de_mutacion_gen
    global limite_poblacion, opcion_max_min
    
    intervalo = [float(entry_intervalo_min.get()), float(entry_intervalo_max.get())]
    Resolución = float(entry_resolucion.get())
    poblacion_inicial = int(entry_poblacion_inicial.get())
    cantidad_generaciones = int(entry_cantidad_generaciones.get())
    posibilidad_de_cruza = int(entry_posibilidad_cruza.get())
    posibilidad_de_mutacion_individuo = int(entry_posibilidad_mutacion_individuo.get())
    posibilidad_de_mutacion_gen = int(entry_posibilidad_mutacion_gen.get())
    limite_poblacion = int(entry_limite_poblacion.get())
    opcion_max_min = var_opcion_max_min.get()

    print("\nVariables globales actualizadas:")
    print(f"Intervalo: {intervalo}")
    print(f"Resolución: {Resolución}")
    print(f"Población Inicial: {poblacion_inicial}")
    print(f"Cantidad de Generaciones: {cantidad_generaciones}")
    print(f"Probabilidad de Cruza: {posibilidad_de_cruza}%")
    print(f"Probabilidad de Mutación de Individuo: {posibilidad_de_mutacion_individuo}%")
    print(f"Probabilidad de Mutación de Gen: {posibilidad_de_mutacion_gen}%")
    print(f"Límite de Población: {limite_poblacion}")

# Elimina imágenes previas para evitar conflictos con el nuevo video
def eliminar_imagenes(carpeta):
    for archivo in natsorted(os.listdir(carpeta)):
        os.remove(os.path.join(carpeta, archivo))
    print("Imágenes previas eliminadas.")

# Creación de la interfaz gráfica
def crear_interfaz_grafica():
    root = tk.Tk()
    root.title("Algoritmo Genético GUI")

    labels = ["Intervalo (min):", "Intervalo (max):", "Resolución:", "Población Inicial:", 
              "Cantidad de Generaciones:", "Probabilidad de Cruza (%):", 
              "Probabilidad de Mutación Individuo (%):", "Probabilidad of Mutación Gen (%):", 
              "Límite of Población:", "Opción (maximizar/minimizar):"]
    entries = [Entry(root) for _ in labels]
    
    for i, (label, entry) in enumerate(zip(labels, entries)):
        Label(root, text=label).grid(row=i, column=0)
        entry.grid(row=i, column=1)
    
    global entry_intervalo_min, entry_intervalo_max, entry_resolucion, entry_poblacion_inicial
    global entry_cantidad_generaciones, entry_posibilidad_cruza, entry_posibilidad_mutacion_individuo
    global entry_posibilidad_mutacion_gen, entry_limite_poblacion, var_opcion_max_min
    
    entry_intervalo_min, entry_intervalo_max, entry_resolucion, entry_poblacion_inicial, entry_cantidad_generaciones, \
    entry_posibilidad_cruza, entry_posibilidad_mutacion_individuo, entry_posibilidad_mutacion_gen, entry_limite_poblacion = entries[:9]
    
    var_opcion_max_min = StringVar(value="maximizar")
    OptionMenu(root, var_opcion_max_min, "maximizar", "minimizar").grid(row=9, column=1)
    
    Button(root, text="Ejecutar Algoritmo Genético", command=lambda: [actualizar_variables_globales(), main()]).grid(row=10, column=0, columnspan=2)
    
    root.mainloop()

if __name__ == "__main__":
    crear_interfaz_grafica()