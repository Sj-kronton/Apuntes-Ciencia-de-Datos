from pathlib import Path
import math

import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SCALER_PATH = BASE_DIR / "modelo_estandarizacion.joblib"

HEADER_IMAGE = (
	"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRLQsMAHbrZDTth8V8FB0TDOD_wnWhQFhs3OUgr5nrYwA&s=10"
)
HEALTHY_IMAGE = (
	"https://img.magnific.com/foto-gratis/corazon-rojo-manos-mujer_1098-2719.jpg?semt=ais_hybrid&w=740&q=80"
)
RISK_IMAGE = (
	"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTRooc1_PoMwuNm_Iuz_6fOLAJzJ_PaFnOuqpbnf_eKjA&s=10"
)


@st.cache_resource
def load_scaler():
	return joblib.load(SCALER_PATH)


def forward(x1, x2):
	"""Ejecuta la red neuronal proporcionada y devuelve su salida tanh."""
	a1 = max(0, -0.35 + (-1.2 * x1) + (-1.4 * x2))
	a2 = max(0, 0.45 + (0.26 * x1) + (0.83 * x2))
	a3 = max(0, 1.0 + (1.3 * x1) + (0.98 * x2))
	a4 = max(0, 0.30 + (-1.1 * x1) + (1.1 * x2))
	a5 = max(0, -0.69 + (0.50 * x1) + (0.51 * x2))
	a6 = max(0, -0.39 + (-0.13 * x1) + (-1.4 * x2))
	a7 = max(0, -0.11 + (1.7 * x1) + (0.010 * x2))
	a8 = max(0, 0.60 + (-0.10 * x1) + (2.0 * x2))
	a9 = max(0, 0.27 + (0.077 * a1) + (0.065 * a2) + (0.021 * a3) + (-0.26 * a4) + (0.067 * a5) + (-0.26 * a6) + (0.34 * a7) + (0.35 * a8))
	a10 = max(0, -2.3 + (0.62 * a1) + (-0.36 * a2) + (0.45 * a3) + (0.74 * a4) + (-0.15 * a5) + (1.4 * a6) + (0.93 * a7) + (0.066 * a8))
	a11 = max(0, 0.91 + (0.38 * a1) + (0.043 * a2) + (0.67 * a3) + (0.68 * a4) + (0.45 * a5) + (-0.71 * a6) + (-1.4 * a7) + (-1.2 * a8))
	a12 = max(0, 0.70 + (-1.5 * a1) + (0.26 * a2) + (1.4 * a3) + (0.94 * a4) + (0.97 * a5) + (0.25 * a6) + (-0.86 * a7) + (-0.61 * a8))
	a13 = max(0, 1.2 + (-0.34 * a1) + (-0.83 * a2) + (0.39 * a3) + (0.70 * a4) + (-0.095 * a5) + (-0.23 * a6) + (-0.30 * a7) + (-1.4 * a8))
	a14 = max(0, 1.1 + (-0.76 * a1) + (-0.37 * a2) + (-0.22 * a3) + (-0.90 * a4) + (0.30 * a5) + (0.88 * a6) + (-0.43 * a7) + (-0.091 * a8))
	a15 = max(0, 0.10 + (-0.018 * a9) + (-0.29 * a10) + (-0.41 * a11) + (-0.41 * a12) + (-0.16 * a13) + (-0.40 * a14))
	a16 = max(0, 0.021 + (-0.62 * a9) + (1.2 * a10) + (1.5 * a11) + (0.14 * a12) + (0.20 * a13) + (0.90 * a14))
	a17 = max(0, 2.1 + (-0.13 * a9) + (-1.6 * a10) + (-0.19 * a11) + (-0.57 * a12) + (-0.21 * a13) + (1.3 * a14))
	a18 = max(0, -0.71 + (-0.84 * a9) + (1.7 * a10) + (0.84 * a11) + (-1.2 * a12) + (1.4 * a13) + (-0.21 * a14))
	a19 = max(0, 0.071 + (-0.39 * a15) + (-0.38 * a16) + (-0.45 * a17) + (-0.086 * a18))
	a20 = max(0, 1.4 + (0.021 * a15) + (1.4 * a16) + (-1.8 * a17) + (-1.7 * a18))
	return math.tanh(-1.7 + (0.40 * a19) + (1.6 * a20))


st.set_page_config(page_title="Predicción de problemas cardiacos", page_icon="❤")

st.title("prediccion de problemas cardiacos")
st.image(HEADER_IMAGE, width=520)
st.write(
	"Esta aplicación experimental estima, a partir de la edad y el nivel de "
	"colesterol, si el patrón de entrada se parece a los casos con problemas "
	"cardiacos aprendidos por la red neuronal. No reemplaza una valoración médica."
)

with st.expander("Instrucciones de uso", expanded=True):
	st.write(
		"1. Ingresa una edad entre 20 y 100 años.\n"
		"2. Ingresa el colesterol entre 200 y 400 mg/dL.\n"
		"3. Pulsa **Realizar predicción** y revisa el resultado y las recomendaciones."
	)

with st.form("prediction_form"):
	age = st.number_input("Edad (años)", min_value=20, max_value=100, value=40, step=1)
	cholesterol = st.number_input(
		"Colesterol (mg/dL)", min_value=200, max_value=400, value=220, step=1
	)
	submitted = st.form_submit_button("Realizar predicción")

if submitted:
	scaler = load_scaler()
	scaled_values = scaler.transform([[age, cholesterol]])[0] * 2
	network_output = forward(float(scaled_values[0]), float(scaled_values[1]))
	prediction = 1 if network_output >= 0 else -1
	confidence = ((network_output + 1) / 2 * 100) if prediction == 1 else ((1 - network_output) / 2 * 100)

	st.subheader("Resultado")
	if prediction == -1:
		st.success("no sufrira del corazon")
		st.image(HEALTHY_IMAGE, width=520)
		st.write(
			"Recomendaciones: mantén una alimentación equilibrada, realiza actividad "
			"física regularmente, duerme bien y asiste a controles médicos periódicos."
		)
	else:
		st.error("sufrira problemas del corazon")
		st.image(RISK_IMAGE, width=520)
		st.metric("Porcentaje de la predicción", f"{confidence:.2f}%")
		if age >= 60 and cholesterol >= 240:
			recommendation = "Por tu edad y colesterol elevado, solicita pronto una valoración médica y sigue un plan supervisado para reducir grasas saturadas y aumentar actividad física."
		elif age >= 60:
			recommendation = "Por tu edad, realiza controles cardiovasculares periódicos, mantén actividad física adaptada y consulta cualquier síntoma con un profesional."
		elif cholesterol >= 240:
			recommendation = "El colesterol está elevado: consulta para controlarlo, prioriza fibra y alimentos frescos, reduce grasas saturadas y realiza actividad física."
		else:
			recommendation = "Mantén una alimentación equilibrada, actividad física regular y controles preventivos para disminuir el riesgo cardiovascular."
		st.write(f"Recomendación: {recommendation}")

st.divider()
st.caption("Sierra J.")
st.caption("Esto es un trabajo experimental UNAB 2026")
