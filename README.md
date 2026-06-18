# 🛡️ DeepShield: Aligned Deepfake Detection via DPO

**DeepShield** is a deepfake detection system that moves beyond simple binary classification. By utilizing **Direct Preference Optimization (DPO)** and **Self-Supervised Learning (SSL)**, this project aligns model "suspicion" with human-perceived risk, specifically targeting the detection of high-fidelity, seamless manipulations that traditional models often miss.

### 🔬 Research Objective
Standard deepfake detectors are prone to high false-positive rates on authentic media and low sensitivity toward "convincing" fakes. This project implements a **Human-in-the-Loop (HITL)** alignment strategy to:
1.  **Increase sensitivity** toward high-fidelity manipulations.
2.  **Reduce false positives** on authentic, "noisy" images.
3.  **Calibrate model confidence** based on human expert rankings.

### 🛠️ Technical Architecture
*   **Feature Extraction:** Utilizes a frozen **DINOv2 (ViT-Small)** backbone, leveraging its self-supervised pre-training to capture fine-grained spatial and geometric inconsistencies.
*   **Stage 1: Supervised Fine-Tuning (SFT):** Established a baseline by training a classification head on the DFDC [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge) dataset.
*   **Stage 2: Direct Preference Optimization (DPO):** Aligned the model using a manually curated dataset of **106 human preference pairs**. The model was optimized using a KL-constrained log-ratio loss to shift the decision boundary toward "convincing" fakes.

### 📈 Experimental Results (Empirical Proof)
The alignment phase yielded significant improvements in model calibration and specificity:

| Sample Type | SFT Baseline | **DPO Aligned (Ours)** | Impact |
| :--- | :--- | :--- | :--- |
| **High-Threat Fake** | 72.5% | **97.5%** | **Increased Sensitivity (+24.9%)** |
| **Obvious/Glitched Fake** | 93.4% | 77.1% | Better Calibration |
| **Authentic Real** | 60.2% | **9.6%** | **Reduced False Positives (-50.6%)** |

**Key Finding:** DPO not only sharpened the detection of seamless fakes but also taught the model to distinguish between "artifact noise" and "authentic human textures," drastically reducing false alarms on real images.

### 💻 System Demo
The included **Streamlit Console** allows users to:
*   Compare the **Sensitivity Shift** between the SFT baseline and the DPO-aligned model.
*   Generate **Forensic Reports** via **Groq (Llama-3.1)** to interpret the "Suspicion Delta" in plain English.

### 🚀 Setup & Usage
1. **Clone & Install:**
   ```bash
   git lfs install
   git clone https://github.com/TorshaMajumder/DeepShield-DPO-Alignment.git
   pip install -r requirements.txt
   ```
2. **Run Inference:**
   Place your `.pth` weights in the `/models` directory and launch:
   ```bash
   streamlit run app.py
   ```

---

## 🚀 Live Demo

<div align="center">
  <video src="YOUR_DRAGGED_LINK_HERE" width="100%" autoplay loop muted playsinline></video>

  <br/>


</div>


---

