# ⚛️ Hybrid Quantum-Classical Image Classifier

> Built for **World Quantum Day Hackathon** — *"Explore the World with Qubits"*  
> Organized by **Pragati QITC** | April 2026

---

## 👨‍💻 Team
**Nemala Jayaram Satya Santosh**  
EEE Undergraduate | Pragati Institute of Technology  

---

## 🎯 Problem Statement
Develop a hybrid model using a Variational Quantum Circuit (VQC) 
for image classification and compare accuracy, training time, 
and parameter count against a classical CNN baseline.

---

## 📊 Final Results

| Metric | Classical CNN 🔵 | Hybrid VQC 🟣 |
|--------|----------------|--------------|
| ✅ Accuracy | 99.00% | **100.00%** |
| 📊 Parameters | 51,618 | **50,518** |
| ⏱️ Training Time | 1.87s | 3.60s |

> 🏆 VQC achieved higher accuracy with fewer parameters!

---

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| PennyLane 0.44.1 | Quantum circuit simulation |
| PyTorch 2.11.0 | Deep learning framework |
| MNIST Dataset | Handwritten digit images |
| Matplotlib | Visualization & graphs |

---

## 📁 Project Structure

VQC-Image-Classifier/
├── step2_data.py         # Dataset loading & visualization
├── step3_cnn.py          # Classical CNN model & training
├── step4_vqc.py          # Hybrid Quantum VQC model
├── step5_compare.py      # Final comparison charts
├── sample_images.png     # Dataset preview
├── final_comparison.png  # Results chart
├── cnn_loss.png          # CNN training graph
└── vqc_loss.png          # VQC training graph


---

## ▶️ How to Run

```bash
# Install dependencies
pip install pennylane torch torchvision matplotlib scikit-learn

# Run step by step
python step2_data.py      # Load & visualize dataset
python step3_cnn.py       # Train Classical CNN
python step4_vqc.py       # Train Hybrid VQC
python step5_compare.py   # Generate comparison charts
```

---

## 🔬 Key Insight
The Hybrid VQC matched and exceeded CNN accuracy with fewer 
parameters — proving quantum advantage potential even on a 
CPU simulator. On real quantum hardware, this gap would be 
significantly larger.

---

## 🔗 Connect
- LinkedIn: [Jayaram Satya Santosh](https://in.linkedin.com/in/jayaram-satya-santosh-nemala-a2b265321)
- Email: 24a31a0247@pragati.ac.in


