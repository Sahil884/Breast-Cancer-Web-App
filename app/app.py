import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
def main():
    def get_clean_data():
        data = pd.read_csv("data/data.csv")
        data = data.drop(['Unnamed: 32', 'id'], axis=1)
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        return data

    def add_sidebar():
        st.sidebar.header("Cell Nuclei Measurements")
        data = get_clean_data()
        slider_labels = [
            ("Radius (mean)", "radius_mean"),
            ("Texture (mean)", "texture_mean"),
            ("Perimeter (mean)", "perimeter_mean"),
            ("Area (mean)", "area_mean"),
            ("Smoothness (mean)", "smoothness_mean"),
            ("Compactness (mean)", "compactness_mean"),
            ("Concavity (mean)", "concavity_mean"),
            ("Concave points (mean)", "concave points_mean"),
            ("Symmetry (mean)", "symmetry_mean"),
            ("Fractal dimension (mean)", "fractal_dimension_mean"),
            ("Radius (se)", "radius_se"),
            ("Texture (se)", "texture_se"),
            ("Perimeter (se)", "perimeter_se"),
            ("Area (se)", "area_se"),
            ("Smoothness (se)", "smoothness_se"),
            ("Compactness (se)", "compactness_se"),
            ("Concavity (se)", "concavity_se"),
            ("Concave points (se)", "concave points_se"),
            ("Symmetry (se)", "symmetry_se"),
            ("Fractal dimension (se)", "fractal_dimension_se"),
            ("Radius (worst)", "radius_worst"),
            ("Texture (worst)", "texture_worst"),
            ("Perimeter (worst)", "perimeter_worst"),
            ("Area (worst)", "area_worst"),
            ("Smoothness (worst)", "smoothness_worst"),
            ("Compactness (worst)", "compactness_worst"),
            ("Concavity (worst)", "concavity_worst"),
            ("Concave points (worst)", "concave points_worst"),
            ("Symmetry (worst)", "symmetry_worst"),
            ("Fractal dimension (worst)", "fractal_dimension_worst"),
        ]

        input_dict = {}

        for label, key in slider_labels:
            input_dict[key] = st.sidebar.slider(
                label=label,
                min_value=float(0),
                max_value=float(data[key].max()),
                value=float(data[key].mean()),
            )

        return input_dict

    def get_scaled_values(input_dict):
        data = get_clean_data()
        x = data.drop(['diagnosis'], axis=1)
        scaled_dict = {}

        for key, value in input_dict.items():
            max_val = x[key].max()
            min_val = x[key].min()
            scaled_value = (value - min_val) / (max_val - min_val)
            scaled_dict[key] = scaled_value
        return scaled_dict

    def get_radar_chart(inp_data):
        inp_data = get_scaled_values(inp_data)

        categories = ["Radius", "Texture", "Perimeter", "Area",
                      "Smoothness", "Compactness", "Concavity", "Concave Points",
                      "Symmetry", "Fractal Dimension"]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[
                inp_data['radius_mean'], inp_data['texture_mean'], inp_data['perimeter_mean'],
                inp_data['area_mean'], inp_data['smoothness_mean'], inp_data['compactness_mean'],
                inp_data['concavity_mean'], inp_data['concave points_mean'], inp_data['symmetry_mean'],
                inp_data['fractal_dimension_mean']
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[
                inp_data['radius_se'], inp_data['texture_se'], inp_data['perimeter_se'], inp_data['area_se'],
                inp_data['smoothness_se'], inp_data['compactness_se'], inp_data['concavity_se'],
                inp_data['concave points_se'], inp_data['symmetry_se'], inp_data['fractal_dimension_se']
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[
                inp_data['radius_worst'], inp_data['texture_worst'], inp_data['perimeter_worst'],
                inp_data['area_worst'], inp_data['smoothness_worst'], inp_data['compactness_worst'],
                inp_data['concavity_worst'], inp_data['concave points_worst'], inp_data['symmetry_worst'],
                inp_data['fractal_dimension_worst']
            ],
            theta=categories,
            fill='toself',
            name='Worst Value'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True
        )

        return fig

    def add_predictions(input_data):
        model = pickle.load(open("model/model.pkl", "rb"))
        scaler = pickle.load(open('model/scaler.pkl', 'rb'))

        input_array = np.array(list(input_data.values())).reshape(1, -1)
        input_array_scaled = scaler.transform(input_array)

        prediction = model.predict(input_array_scaled)

        st.subheader("Cell Cluster Prediction")
        st.write("The cell cluster is :")

        if prediction[0] == 0:
            st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
        else:
            st.write("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)

        st.write(f"Probabilty of being Benign: <span class='B'>{model.predict_proba(input_array_scaled)[0][0]}</span>", unsafe_allow_html=True)
        st.write(f"Probabilty of being Malignant: <span class='M'>{model.predict_proba(input_array_scaled)[0][1]}</span>", unsafe_allow_html=True)

        st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")




    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    inp_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. "
                 "This app predicts using machine learning model whether a breast mass is benign or malignant based on the measurements it recieves from your cytosis lab. "
                 "You can also update the measurement manually by usung the sliders in the sidebar. ")
        col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(inp_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(inp_data)


if __name__ == "__main__":
    main()