from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


MODEL_PATH = Path(__file__).parent / "prendas.keras"

# Clases estándar de Fashion-MNIST.
# Sustituya este arreglo si el modelo fue entrenado con otras clases.
CLASS_NAMES = [
    "Camiseta/top",
    "Pantalón",
    "Jersey",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla",
    "Bolso",
    "Botín",
]


st.set_page_config(
    page_title="Predictor de prendas",
    page_icon="👕",
    layout="centered",
)

st.title("👕 Predictor de prendas con TensorFlow")
st.write("Dibuje una prenda o cargue una imagen para realizar una predicción.")

if not MODEL_PATH.is_file():
    st.error(f"No se encontró el modelo: {MODEL_PATH.name}")
    st.info(
        "Guarde o suba el modelo entrenado con el nombre "
        "'prendas.keras' en la misma carpeta que app.py."
    )
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """Convierte una imagen a escala de grises y tamaño 28x28."""
    image = image.convert("L")
    image = ImageOps.fit(
        image,
        (28, 28),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    model_input = image_array

    input_shape = load_model().input_shape

    if len(input_shape) == 4:
        model_input = model_input[..., np.newaxis]

    model_input = np.expand_dims(model_input, axis=0)
    return model_input, image


def predict(image: Image.Image):
    model = load_model()
    model_input, processed_image = prepare_image(image)

    prediction = model.predict(model_input, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index])

    return predicted_index, confidence, processed_image, prediction


input_mode = st.radio(
    "Seleccione el método de entrada:",
    ["Dibujar en el canvas", "Subir una imagen"],
    horizontal=True,
)

image_to_predict = None

if input_mode == "Dibujar en el canvas":
    st.subheader("Dibuje la prenda")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        return_image_data=True,
        key="clothing_canvas",
    )

    if canvas_result.image_data is not None:
        canvas_image = Image.fromarray(
            canvas_result.image_data.astype("uint8"),
            mode="RGBA",
        )
        image_to_predict = canvas_image.convert("L")

else:
    st.subheader("Cargue una imagen")
    uploaded_file = st.file_uploader(
        "Formatos permitidos: PNG, JPG o JPEG",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
        st.image(image_to_predict, caption="Imagen cargada", width=280)


if st.button("Predecir", type="primary"):
    if image_to_predict is None:
        st.warning("Dibuje una imagen o cargue un archivo antes de predecir.")
    else:
        try:
            class_index, confidence, processed_image, probabilities = predict(
                image_to_predict
            )

            st.image(
                processed_image,
                caption="Imagen procesada a 28 x 28 píxeles",
                width=180,
            )

            st.success(
                f"Predicción: **{CLASS_NAMES[class_index]}** "
                f"({confidence:.2%} de confianza)"
            )

            st.subheader("Probabilidades")
            probability_data = {
                CLASS_NAMES[index]: float(probabilities[index])
                for index in range(min(len(CLASS_NAMES), len(probabilities)))
            }
            st.bar_chart(probability_data)

        except Exception as error:
            st.error(f"No fue posible realizar la predicción: {error}")


st.divider()
st.subheader("Instrucciones y recomendaciones")
st.markdown(
    """
- Dibuje una sola prenda sobre el fondo negro.
- Use trazos blancos o claros, similares a los datos de entrenamiento.
- Las imágenes se convierten automáticamente a escala de grises y se redimensionan a **28 x 28 píxeles**.
- Las imágenes cargadas deben ser similares o parecidas a las utilizadas durante el entrenamiento.
- Para obtener mejores resultados, centre la prenda y evite fondos, texto u objetos adicionales.
"""
)