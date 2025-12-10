
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
import plotly.graph_objects as go

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from src.preprocessing import DataPreprocessor

st.set_page_config(
    page_title="🎬 Movie Revenue Prediction",
    page_icon="🎬",
    layout="wide"
)

MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# @st.cache_resource
def load_resources():
    """Load model and preprocessor with caching"""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = DataPreprocessor.load_preprocessor(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

def safe_predict(model, X):
    """
    Safe prediction that bypasses feature name validation
    Works with both XGBoost and sklearn models
    """
    # Ensure X is pure numpy array
    X = np.asarray(X, dtype=np.float64)
    
    # For XGBoost models, use the internal booster directly
    if hasattr(model, 'get_booster'):
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        return model.get_booster().predict(dmatrix)
    else:
        # For sklearn models
        return model.predict(X)

def main():
    st.title("🎬 Movie Revenue Prediction AI")
    st.markdown("### Predict Box Office Revenue using Advanced Machine Learning")
    
    model, preprocessor = load_resources()
    
    if model is None or preprocessor is None:
        st.warning("Please ensure trained models exist in 'models/' directory.")
        return

    st.sidebar.header("📝 Movie Details")
    
    with st.sidebar:
        title = st.text_input("Movie Title", "My Awesome Movie")
        
        budget = st.number_input(
            "Budget (USD)", 
            min_value=1000, 
            value=50000000, 
            step=1000000,
            format="%d"
        )
        
        runtime = st.number_input(
            "Runtime (minutes)", 
            min_value=10, 
            max_value=300, 
            value=120
        )
        
        popularity = st.slider(
            "Popularity Score", 
            min_value=0.0, 
            max_value=2000.0, 
            value=50.0
        )
        
        vote_average = st.slider(
            "Average Rating (0-10)", 
            min_value=0.0, 
            max_value=10.0, 
            value=6.5
        )
        
        vote_count = st.number_input(
            "Vote Count", 
            min_value=0, 
            value=1000
        )
        
        release_date = st.date_input("Release Date")
        
        available_genres = [
            'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
            'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 
            'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 
            'TV Movie', 'Thriller', 'War', 'Western'
        ]
        selected_genres = st.multiselect(
            "Genres", 
            available_genres,
            default=['Action', 'Adventure']
        )
        
        overview = st.text_area(
            "Plot Overview",
            "A group of heroes embark on a dangerous journey to save the world from an ancient evil.",
            height=150
        )
        
        predict_btn = st.button("🚀 Predict Revenue", use_container_width=True)

    if predict_btn:
        with st.spinner("🤖 AI is analyzing the movie..."):
            
            input_data = {
                'id': [0], 
                'title': [title],
                'budget': [budget],
                'popularity': [popularity],
                'runtime': [runtime],
                'release_date': [release_date.strftime('%Y-%m-%d')],
                'vote_average': [vote_average],
                'vote_count': [vote_count],
                'genres': ['|'.join(selected_genres)],  # Pipe-separated format
                'overview': [overview],
                'production_companies': [''],  # Add empty production companies
                'original_language': ['en'],  # Add default language
                'status': ['Released'],  # Add default status
                'revenue': [0] 
            }
            
            df_input = pd.DataFrame(input_data)
            
            try:
                X_processed, _ = preprocessor.transform(df_input)
                
                # Use safe_predict to bypass feature name validation
                log_pred = safe_predict(model, X_processed)
                revenue_pred = np.expm1(log_pred)[0]
                
                st.success("Analysis Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="💰 Predicted Revenue",
                        value=f"${revenue_pred:,.2f}",
                        delta=f"{((revenue_pred - budget)/budget)*100:.1f}% ROI"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="💸 Budget",
                        value=f"${budget:,.2f}"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("### 📊 Financial Outlook")
                
                fig = go.Figure(data=[
                    go.Bar(name='Budget', x=['Financials'], y=[budget], marker_color='#ef553b'),
                    go.Bar(name='Predicted Revenue', x=['Financials'], y=[revenue_pred], marker_color='#00cc96')
                ])
                
                fig.update_layout(
                    barmode='group',
                    height=400,
                    template='plotly_white',
                    title_text="Budget vs Predicted Revenue"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ROI Analysis
                roi = revenue_pred - budget
                if roi > 0:
                    st.balloons()
                    st.markdown(f"### 🎉 Profit: ${roi:,.2f}")
                else:
                    st.markdown(f"### ⚠️ Loss: ${abs(roi):,.2f}")
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.error("Please check if the input format matches the training data.")

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using basic Streamlit & XGBoost")

if __name__ == "__main__":
    main()
