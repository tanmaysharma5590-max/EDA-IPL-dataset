import streamlit as st
import pandas as pd
import pickle

# ---------------------------------------------------------
# Load trained pipeline (must exist as pipe.pkl in same folder)
# ---------------------------------------------------------
pipe = None
pipe_load_error = None
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except Exception as e:
    pipe_load_error = str(e)

teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title='IPL Win Predictor', page_icon='🏏', layout='centered')

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #d4f5e0 0%, #e8f9ef 100%);
}

.main .block-container {
    background-color: #ffffff;
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    max-width: 700px;
}

h1 {
    color: #1a2332;
    font-weight: 700;
}

label {
    color: #2d3748 !important;
    font-weight: 500 !important;
}

div[data-baseweb="select"] > div,
.stNumberInput input {
    background-color: #eef1f6 !important;
    border-radius: 10px !important;
    border: none !important;
}

div.stButton > button {
    background: linear-gradient(90deg, #16a34a, #15803d);
    color: white;
    font-weight: 600;
    font-size: 18px;
    padding: 0.75rem 0;
    width: 100%;
    border: none;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(22,163,74,0.3);
    transition: 0.2s;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #15803d, #166534);
    transform: translateY(-1px);
}

.result-box {
    margin-top: 1.5rem;
    padding: 1.25rem;
    border-radius: 14px;
    text-align: center;
    font-size: 20px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

if pipe_load_error:
    st.error(f"Could not load model (pipe.pkl): {pipe_load_error}")

st.markdown("<h1>🏏 Match Situation</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Batting Team (chasing)', sorted(teams))
with col2:
    bowling_team = st.selectbox('Bowling Team', sorted(teams))

city = st.selectbox('Host City', sorted(cities))

col3, col4 = st.columns(2)
with col3:
    target = st.number_input('Target Score', min_value=0, step=1, value=180)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1, value=10.0)

col5, col6 = st.columns(2)
with col5:
    score = st.number_input('Current Score', min_value=0, step=1, value=100)
with col6:
    wickets_out = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1, value=2)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button('🎯 Predict Win Probability'):
    if batting_team == bowling_team:
        st.error("Batting and bowling team can't be the same.")
    elif pipe is None:
        st.error("Model file 'pipe.pkl' not found. Place it in the same folder as this script.")
    else:
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets_out
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        loss_prob = result[0][0]
        win_prob = result[0][1]

        st.markdown(f"""
        <div class="result-box" style="background-color:#dcfce7; color:#15803d;">
            {batting_team}: {round(win_prob * 100)}% to win
        </div>
        <div class="result-box" style="background-color:#fee2e2; color:#b91c1c;">
            {bowling_team}: {round(loss_prob * 100)}% to win
        </div>
        """, unsafe_allow_html=True)
