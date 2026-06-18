import streamlit as st
import os
import pandas as pd
import random
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(page_title="DeepShield: DPO Annotator", layout="wide")
FAKE_DIR = "/Users/torshamajumder/vector_db/extracted_images/fake/"  
SAVE_FILE = "/Users/torshamajumder/vector_db/preferences.csv"

# --- STATE MANAGEMENT ---
if 'pair' not in st.session_state:
    st.session_state.all_fakes = [f for f in os.listdir(FAKE_DIR) if f.endswith(('.jpg', '.png'))]
    st.session_state.pair = random.sample(st.session_state.all_fakes, 2)

# --- FUNCTIONS ---
def save_preference(winner, loser):
    new_data = pd.DataFrame([[winner, loser]], columns=["winner", "loser"])
    if os.path.exists(SAVE_FILE):
        df = pd.read_csv(SAVE_FILE)
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(SAVE_FILE, index=False)
    else:
        new_data.to_csv(SAVE_FILE, index=False)
    
    # Get new pair
    st.session_state.pair = random.sample(st.session_state.all_fakes, 2)

# --- UI ---
st.title("🛡️ DeepShield: Human Preference Curation")
st.write("Which of these deepfakes is **more convincing / harder to detect**?")

col1, col2 = st.columns(2)

img_a_path = os.path.join(FAKE_DIR, st.session_state.pair[0])
img_b_path = os.path.join(FAKE_DIR, st.session_state.pair[1])

with col1:
    st.image(Image.open(img_a_path), use_column_width=True)
    if st.button("Left is more convincing (Winner)", use_container_width=True):
        save_preference(st.session_state.pair[0], st.session_state.pair[1])
        st.rerun()

with col2:
    st.image(Image.open(img_b_path), use_column_width=True)
    if st.button("Right is more convincing (Winner)", use_container_width=True):
        save_preference(st.session_state.pair[1], st.session_state.pair[0])
        st.rerun()

st.divider()
if os.path.exists(SAVE_FILE):
    count = len(pd.read_csv(SAVE_FILE))
    st.write(f"Total Preferences Captured: **{count}**")
    st.caption("Aim for 100-200 for a solid DPO alignment pilot.")