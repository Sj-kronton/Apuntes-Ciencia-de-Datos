import pandas as pd

# ------------------------------------------------------------
# CONFIGURACIÓN, cargar el csv limpio y la ruta de salida del json
# ------------------------------------------------------------

ARCHIVO_CSV = "C:\\Users\\juana\\OneDrive\\Documents\\Clase\\Apuntes-Ciencia-de-Datos\\pacientes_limpio.csv"
ARCHIVO_JSON = "C:\\Users\\juana\\OneDrive\\Documents\\Clase\\Apuntes-Ciencia-de-Datos\\pacientes_playground.json"


# ------------------------------------------------------------
# CONVERSIÓN CSV → JSON CON Z-SCORE
# ------------------------------------------------------------

#Debido a que los datos de dataser original estan en una pocision MUY sesgada, 
#es imposible usar un json en "bruto" para playground,
#Por lo que se usoo una normalizacion con Z-score para que los datos se
#encuentren en un rango mas cercano a 0, sin perder su valor

def convertir_csv_a_json_zscore(ruta_csv, ruta_json):

    # Leer el CSV limpio
    df = pd.read_csv(ruta_csv)

    print(f"Datos cargados: {len(df)} filas")

    # --------------------------------------------------------
    # Verificar columnas, por si acaso
    # --------------------------------------------------------

    columnas_requeridas = [
        "edad",
        "colesterol",
        "problema_cardiaco"
    ]

    columnas_faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in df.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas en el CSV: {columnas_faltantes}"
        )

    # --------------------------------------------------------
    # Convertir a valores numéricos
    # --------------------------------------------------------

    edad = pd.to_numeric(df["edad"])
    colesterol = pd.to_numeric(df["colesterol"])

    # --------------------------------------------------------
    # Calcular media y desviación estándar
    #
    # ddof=0 utiliza la desviación estándar poblacional.
    # --------------------------------------------------------

    media_edad = edad.mean()
    desviacion_edad = edad.std(ddof=0)

    media_colesterol = colesterol.mean()
    desviacion_colesterol = colesterol.std(ddof=0)

    # --------------------------------------------------------
    # Aplicar Z-SCORE
    #
    # z = (valor - media) / desviación_estándar
    # --------------------------------------------------------

    edad_z = (
        (edad - media_edad)
        / desviacion_edad
    ) * 2

    colesterol_z = (
        (colesterol - media_colesterol)
        / desviacion_colesterol
    ) * 2

    # --------------------------------------------------------
    # Crear estructura compatible con Playground
    #
    # edad              → x
    # colesterol        → y
    # problema_cardiaco → label
    #
    # Playground utiliza:
    # -1 = clase negativa
    # +1 = clase positiva
    # --------------------------------------------------------

    df_playground = pd.DataFrame({
        "x": edad_z,
        "y": colesterol_z,
        "label": df["problema_cardiaco"].map({
            0: -1,
            1: 1
        })
    })

    # --------------------------------------------------------
    # Comprobar etiquetas
    # --------------------------------------------------------

    if df_playground["label"].isnull().any():

        valores_invalidos = (
            df.loc[
                df_playground["label"].isnull(),
                "problema_cardiaco"
            ]
            .unique()
        )

        raise ValueError(
            "Se encontraron valores inválidos en "
            "'problema_cardiaco': "
            f"{valores_invalidos}"
        )

    # --------------------------------------------------------
    # Exportar JSON
    # --------------------------------------------------------

    df_playground.to_json(
        ruta_json,
        orient="records",
        indent=2
    )

    # --------------------------------------------------------
    # Mostrar información de la transformación
    # --------------------------------------------------------

    print("\n--- Normalización Z-Score ---")

    print(f"Edad:")
    print(f"  Media = {media_edad:.4f}")
    print(f"  Desviación estándar = {desviacion_edad:.4f}")

    print(f"\nColesterol:")
    print(f"  Media = {media_colesterol:.4f}")
    print(f"  Desviación estándar = {desviacion_colesterol:.4f}")

    print("\n--- Resultado ---")

    print(
        f"Media de x después de normalizar: "
        f"{df_playground['x'].mean():.6f}"
    )

    print(
        f"Media de y después de normalizar: "
        f"{df_playground['y'].mean():.6f}"
    )

    print(
        f"\nJSON creado correctamente: {ruta_json}"
    )

    print(
        f"Registros exportados: "
        f"{len(df_playground)}"
    )


# ------------------------------------------------------------
# EJECUCIÓN
# ------------------------------------------------------------

if __name__ == "__main__":

    convertir_csv_a_json_zscore(
        ARCHIVO_CSV,
        ARCHIVO_JSON
    )