import shap
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from data_processing import add_weather_data, SmartNormalizerDF, target_col, ok_val, important_columns
from catboost import CatBoostClassifier
import os
import warnings


st.set_page_config(
    page_title="✨ Advanced Data Analysis ✨",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

def main():
    st.markdown("<h1 style='text-align: center;'>📊 Advanced Data Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    # Create sidebar menu with larger text
    st.sidebar.markdown("<h2>Navigation Menu</h2>", unsafe_allow_html=True)
    menu = st.sidebar.selectbox("Select Section", ["Data", "Prediction", "Analysis"])
    
    if menu == "Data":
        st.markdown("<h2>📥 Load and View Data</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.markdown("<h3>📋 Raw Data:</h3>", unsafe_allow_html=True)
                st.dataframe(df, height=300)
                st.session_state['data'] = df

                df = add_weather_data(df).drop(columns=["message_timestamp", "physical_part_id"])
                target = df[target_col] == ok_val
                df = df.drop(columns=[target_col])
                df = df[important_columns]

                smart_normalizer = SmartNormalizerDF(two_col=True)
                smart_normalizer.fit(df)
                df = smart_normalizer.transform(df)

                model = CatBoostClassifier()
                model.load_model("app/best_model.cbm")

                st.session_state['model'] = model

                predictions = model.predict_proba(df)[:, 1]
                df['status'] = predictions
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                    help="Click to download the processed data with predictions"
                )

                st.markdown("<h3>🔄 Processed Data with Predictions:</h3>", unsafe_allow_html=True)
                st.markdown("<div style='font-size: 1.2em'>The 'status' column shows the probability of failure - higher values indicate greater risk</div>", unsafe_allow_html=True)
                
                st.markdown("<h4>📊 Data Sorted by Risk Level (Lowest to Highest):</h4>", unsafe_allow_html=True)
                st.dataframe(df.sort_values(by='status', ascending=True), height=400, use_container_width=True)
                
                st.info("💡 The table above shows all your data points sorted by risk level. " 
                       "Lower values in the 'status' column indicate lower risk of failure.")

                # Add row selection by ID
                row_id = st.number_input("Enter Row ID to analyze:", min_value=0, max_value=len(df)-1, value=0)
                if st.button("Analyze Selected Row"):
                    # Store selected row in session state
                    st.session_state['selected_row'] = df[df.index == row_id]
                    # Switch to Analysis tab
                    menu = "Prediction"
                    # Force rerun to update the page
                    st.rerun()
                
                
    elif menu == "Prediction":
        if 'data' not in st.session_state:
            st.warning("⚠️ Please load data in the 'Data' tab first")
        else:
            st.markdown("<h2>🔮 Prediction</h2>", unsafe_allow_html=True)
            selected_row = st.session_state['selected_row']
            st.write("Selected Row Data:")
            st.write(selected_row)

            model = st.session_state['model']

            def st_shap(plot, height=None):
                shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                components.html(shap_html, height=height)

            # Get feature importance explanation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(selected_row)
            
            st.write("Feature Importance Analysis:")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st.pyplot(shap.plots.waterfall(shap_values[0]))
            #st_shap(shap.plots.waterfall(shap_values[0]), height=500)

    else:  # Analysis
        if 'data' not in st.session_state:
            st.warning("⚠️ Please load data in the 'Data' tab first")
        else:
            st.markdown("<h2>📊 Data Analysis</h2>", unsafe_allow_html=True)
            df = st.session_state['data']
            
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h3>📈 Scatter Plot Analysis</h3>", unsafe_allow_html=True)
                    x_col = st.selectbox("Select X Axis", numeric_cols)
                    y_col = st.selectbox("Select Y Axis", numeric_cols)
                    
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
                    fig.update_layout(title_font_size=24)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("<h3>📊 Distribution Analysis</h3>", unsafe_allow_html=True)
                    hist_col = st.selectbox("Select Column for Histogram", numeric_cols)
                    fig = px.histogram(df, x=hist_col, title=f"Distribution of {hist_col}")
                    fig.update_layout(title_font_size=24)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Not enough numeric columns to create plots")

if __name__ == "__main__":
    main()