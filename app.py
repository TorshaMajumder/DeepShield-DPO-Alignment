import timm
import torch
from groq import Groq 
from PIL import Image
import torch.nn as nn
import streamlit as st
from torchvision import transforms

# --- MODEL DEFINITION ---
class DeepfakeSentinel(nn.Module):
    def __init__(self):
        super().__init__()
        # Ensure this matches your Kaggle backbone!
        self.backbone = timm.create_model('vit_small_patch16_224_dino', pretrained=True, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# --- PRE-PROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- UI SETUP ---
st.set_page_config(page_title="DeepShield: Aligned Detection", layout="wide")
st.title("🛡️ DeepShield: AI-Aligned Deepfake Sentinel")
st.markdown("This system uses **DPO (Direct Preference Optimization)** to align detection with human-perceived risk.")

# Sidebar for model loading
with st.sidebar:
    st.header("Configuration")
    
    # GROQ KEY AS VISIBLE TEXT
    groq_key = st.text_input("Groq API Key", type="password")
    
    st.divider()
    st.subheader("Model Registry")
    sft_path = st.file_uploader("Upload SFT Model (.pth)", type="pth")
    dpo_path = st.file_uploader("Upload DPO Model (.pth)", type="pth")


if sft_path and dpo_path:
    # Load Models
    @st.cache_resource
    def load_models(s_p, d_p):
        sft_m = DeepfakeSentinel()
        sft_m.load_state_dict(torch.load(s_p, map_location='cpu'))
        sft_m.eval()
        
        dpo_m = DeepfakeSentinel()
        dpo_m.load_state_dict(torch.load(d_p, map_location='cpu'))
        dpo_m.eval()
        return sft_m, dpo_m

    sft_model, dpo_model = load_models(sft_path, dpo_path)

    # --- INFERENCE ---
    uploaded_file = st.file_uploader("Choose an image to scan...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Target Image", width=400)
        
        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            sft_logit = sft_model(input_tensor).item()
            dpo_logit = dpo_model(input_tensor).item()

        # Conversion logic
        sft_prob = torch.sigmoid(torch.tensor(sft_logit)).item()
        dpo_prob = torch.sigmoid(torch.tensor(dpo_logit)).item()

        # Display Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("SFT Baseline Score", f"{sft_prob*100:.1f}%")
        with col2:
            st.metric("DPO Aligned Risk", f"{dpo_prob*100:.1f}%")
        with col3:
            shift = (dpo_prob - sft_prob) * 100
            st.metric("Sensitivity Shift", f"{shift:+.1f}%")


        # --- THE "RESEARCH" INSIGHT ---
        st.divider()
        st.subheader("🤖 AI Security Analyst")
        
        if st.button("Generate Forensic Report"):
            if groq_key:
                try:
                    client = Groq(api_key=groq_key)
                    with st.spinner("Generating Analyst Report..."):
                        prompt = f"""
                        Analyze these Deepfake Detection results for a security audit:
                        - Baseline SFT Probability: {sft_prob*100:.1f}%
                        - DPO-Aligned Risk Score: {dpo_prob*100:.1f}%
                        - Detected Shift: {shift:+.1f}%
                        
                        The Direct Preference Optimization (DPO) model was aligned by a human expert to prioritize high-fidelity, convincing deepfakes.
                        Interpret what this shift means regarding the 'Convincingness' and 'Threat Level' of this specific image.
                        """
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.info(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Groq Error: {e}")
            else:
                st.warning("Please enter your Groq API Key in the sidebar.")
else:
    st.info("Awaiting model weights (.pth) for analysis.")