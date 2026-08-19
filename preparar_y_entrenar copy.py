"""
Script: preparar_y_entrenar.py
Objetivo: Limpiar el dataset pacientes.csv y prepararlo correctamente
          
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier #Ironicamente, sklearn no se utiliza en el script
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------------------------------------------------------------------
# 1. CARGAR DATOS
# ---------------------------------------------------------------------------
def cargar_datos(ruta_csv):
    #crear dataframe con pandas
    df = pd.read_csv(ruta_csv)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


# ---------------------------------------------------------------------------
# 2. EXPLORAR DATOS (antes de limpiar)
# ---------------------------------------------------------------------------
def explorar_datos(df):
    """
    Muestra un resumen del estado actual de los datos:
    - Cuántos valores nulos (vacíos) hay por columna.
    - Cuántas filas están duplicadas
    """
    print("\n--- Exploración de datos ---")
    print("Valores nulos por columna:")
    print(df.isnull().sum())
    print(f"\nFilas duplicadas: {df.duplicated().sum()}")
    print(f"Balance de clases (problema_cardiaco):\n{df['problema_cardiaco'].value_counts()}")


# ---------------------------------------------------------------------------
# 3. LIMPIAR DATOS
# ---------------------------------------------------------------------------
def limpiar_datos(df):
    """
    Corrige los problemas detectados en el dataset:

    a) Valores faltantes (NaN) en 'edad' y 'colesterol':
       En vez de eliminar esas filas (perderíamos datos útiles), se
       rellenan (imputan) con la MEDIANA de cada columna. Se usa la
       mediana en vez del promedio porque es menos sensible a valores
       extremos (outliers) como el colesterol de 564. """
            #Estar pendiente a esto, la mediana puede crear datos iguales con diferente resultado
    """
    b) Filas duplicadas:
       Se eliminan porque un mismo paciente repetido dos veces podría
       hacer que el modelo le dé más peso del que debería a esa fila
       en particular, sesgando el aprendizaje.

    Devuelve un DataFrame limpio, sin nulos y sin duplicados.
    """
    df_limpio = df.copy()

    # a) Rellenar valores faltantes con la mediana de cada columna
    mediana_edad = df_limpio['edad'].median()
    mediana_colesterol = df_limpio['colesterol'].median()
    df_limpio['edad'] = df_limpio['edad'].fillna(mediana_edad)
    df_limpio['colesterol'] = df_limpio['colesterol'].fillna(mediana_colesterol)

    # b) Eliminar conflictos: mismas entradas con diferentes resultados
    # Si dos registros tienen exactamente los mismos valores de entrada
    # ('edad' y 'colesterol') pero etiquetas diferentes, son contradictorios
    # para este modelo. Eliminamos TODAS las filas pertenecientes a esos
    # grupos conflictivos, incluidas las creadas luego de la imputacion de medianas.
    columnas_entrada = ['edad', 'colesterol']
    filas_antes = df_limpio.shape[0]

    etiquetas_por_entrada = (
        df_limpio.groupby(columnas_entrada)['problema_cardiaco']
        .nunique()
    )

    entradas_conflictivas = etiquetas_por_entrada[
        etiquetas_por_entrada > 1
    ].index

    if len(entradas_conflictivas) > 0:
        indice_conflictivo = df_limpio.set_index(
            columnas_entrada
        ).index.isin(entradas_conflictivas)

        conflictos_eliminados = int(indice_conflictivo.sum())

        print("\nEntradas conflictivas encontradas:")
        for entrada in entradas_conflictivas:
            print(f"  edad={entrada[0]}, colesterol={entrada[1]}")

        df_limpio = df_limpio.loc[~indice_conflictivo].copy()
    else:
        conflictos_eliminados = 0
        print("\nNo se encontraron entradas con etiquetas contradictorias.")

    # c) Eliminar filas completamente duplicadas restantes
    duplicados_antes = df_limpio.shape[0]
    df_limpio = df_limpio.drop_duplicates()
    duplicados_eliminados = duplicados_antes - df_limpio.shape[0]

    print("\n--- Limpieza de datos ---")
    print(f"Nulos en 'edad' rellenados con mediana = {mediana_edad}")
    print(f"Nulos en 'colesterol' rellenados con mediana = {mediana_colesterol}")
    print(f"Filas eliminadas por etiquetas contradictorias: {conflictos_eliminados}")
    print(f"Duplicados exactos eliminados después: {duplicados_eliminados}")
    print(f"Filas finales: {df_limpio.shape[0]}")

    return df_limpio


# ---------------------------------------------------------------------------
# 4. PREPARAR DATOS PARA LA RED NEURONAL
# ---------------------------------------------------------------------------
def preparar_datos(df_limpio):
    """
    Deja los datos en el formato que necesita una red neuronal:

    - X: variables predictoras (entradas) -> edad, colesterol
    - y: variable objetivo (salida a predecir) -> problema_cardiaco

    - División train/test: separamos 80% de los datos para ENTRENAR
      el modelo y 20% para EVALUARLO con datos que nunca vio. Esto es
      clave para saber si el modelo realmente aprendió o solo
      "memorizó" los datos de entrenamiento (overfitting).

    - Escalado (StandardScaler): las redes neuronales entrenan mucho
      mejor cuando las variables de entrada están en escalas similares.
      Aquí 'edad' va de ~29 a 77, y 'colesterol' de ~126 a 564: son
      rangos muy distintos. El escalado transforma ambas columnas para
      que tengan media 0 y desviación estándar 1, evitando que la red
      le dé más "importancia" al colesterol solo porque sus números
      son más grandes.

      IMPORTANTE: el escalador se "ajusta" (fit) SOLO con los datos de
      entrenamiento, y luego se aplica (transform) igual a los datos de
      test. Esto evita que información del set de prueba se filtre al
      entrenamiento (data leakage).
    """
    X = df_limpio[['edad', 'colesterol']]
    y = df_limpio['problema_cardiaco']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # stratify=y asegura que la proporción de 0s y 1s sea igual en
    # train y test, manteniendo el balance de clases.

    escalador = StandardScaler()
    X_train_escalado = escalador.fit_transform(X_train)
    X_test_escalado = escalador.transform(X_test)

    print("\n--- Preparación de datos ---")
    print(f"Entrenamiento: {X_train.shape[0]} filas")
    print(f"Prueba (test): {X_test.shape[0]} filas")

    return X_train_escalado, X_test_escalado, y_train, y_test

#Toda esta parte es omitible, la idea de usar sklearn para entrenar una red neural
#directamente en la sesion es inecesario

# ---------------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL DEL SCRIPT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    RUTA_ARCHIVO = "C:\\Users\\juana\\OneDrive\\Documents\\Clase\\Apuntes-Ciencia-de-Datos\\pacientes.csv"  # cambia esto por la ruta de tu archivo

    df = cargar_datos(RUTA_ARCHIVO)
    explorar_datos(df)
    df_limpio = limpiar_datos(df)

    # Guardamos una copia del dataset ya limpio, es el que se usara para pasar a json
    df_limpio.to_csv("C:\\Users\\juana\\OneDrive\\Documents\\Clase\\Apuntes-Ciencia-de-Datos\\pacientes_limpio.csv", index=False)
    print("\nArchivo limpio guardado como 'pacientes_limpio.csv'")

