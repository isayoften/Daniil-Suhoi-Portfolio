import streamlit as st
import pickle

st.set_page_config(
    page_title='Peritonit predictions',
    page_icon='💊'
)


@st.cache_resource
def load_model():
    with open('SRG_model_knn.pkl', 'rb') as file:
        return pickle.load(file)


model = load_model()

st.write("### Болевой синдром")
pain_syndrome_input = st.selectbox(
    label=" ",
    options=("Есть", "Нет"),
    index=None,
    placeholder="Выберите вариант",
    label_visibility='collapsed'
)

st.write("### Диурез")
diuresis_input = st.number_input(
    label=" ",
    value=None,
    placeholder="Введите диурез",
    min_value=800,
    max_value=2100,
    step=100,
    label_visibility='collapsed'
)
st.caption("Диурез от 900 мл до 2000 мл")

st.write("### Объем выпота")
effusion_input = st.number_input(
    label=" ",
    value=None,
    placeholder="Введите объём выпота",
    min_value=0,
    max_value=600,
    step=100,
    label_visibility='collapsed'
)
st.caption("Выпот от 50 мл до 500 мл")

if pain_syndrome_input == "Есть":
    pain = 1
else:
    pain = 0

if st.button("Предсказать"):
    if all([pain_syndrome_input, diuresis_input, effusion_input]):
        try:
            proba = model.predict_proba([[pain, diuresis_input, effusion_input]])[0, 1]

            if int(proba * 100) == 100:
                st.write(f"Вероятность возникновения осложнения 99.9 %")
            elif int(proba * 100) == 0:
                st.write(f"Вероятность возникновения осложнения 0.1 %")
            else:
                st.write(f"Вероятность возникновения осложнения {int(proba * 100)} %")

        except Exception:
            st.write('Ошибка в данных')
    else:
        st.write("Введены не все данные")
