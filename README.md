# **NLP Intent Parser — End-to-End Industrial Command Understanding**

This project implements a complete **intent parsing system** for technician-style instructions in industrial and microgrid environments.
It takes raw natural language like:

> “check the inverter temperature and update the power limit to 20%”

…and outputs structured actions:

```json
{
  "intent": "update_parameter",
  "target": "inverter",
  "parameter": {
    "name": "power_limit",
    "value": "20%"
  }
}
```

The system benchmarks **three NLP modeling families**:

1. **TF-IDF + Linear SVM** — baseline intent classification
2. **LSTM / BiLSTM** — multi-output classification (intent + target + parameter)
3. **DistilBERT Token Classifier** — end-to-end structured extraction

The goal: build a clean, production-style pipeline that demonstrates how traditional ML, classical deep learning, and modern transformers differ in capability and performance.

---

## **📌 Key Features**

### **✓ Synthetic Dataset Generation**

Because real technician logs are private and inconsistent, the project builds a **controlled, balanced synthetic dataset** covering:

- 10+ intents
- 15+ equipment targets
- 20+ parameter types
- Numeric, categorical, and percentage values

Flexible enough to extend or adapt to real operational logs later.

---

### **✓ Multi-Model Benchmarking**

Each model solves the same set of tasks:

- **Intent classification**
- **Target identification**
- **Parameter extraction**

This allows for direct comparison between:

| Model             | Strengths                                      | Weaknesses                          |
| ----------------- | ---------------------------------------------- | ----------------------------------- |
| **TF-IDF + SVM**  | Fast, simple, solid baseline                   | No structured extraction            |
| **LSTM / BiLSTM** | Multi-output, learns patterns                  | Needs hand-engineered preprocessing |
| **DistilBERT**    | Best overall generalisation; robust extraction | Heavier, slower on CPU              |

---

### **✓ End-to-End Parsing Demo**

A final pipeline demonstrates how raw text becomes structured output:

- Tokenisation
- Transformer inference
- Slot grouping
- Value extraction
- JSON-like final output

This is the most production-like part of the system and the highlight of the project.

---

### **✓ Clear, Modular Notebook Structure**

The notebook is intentionally structured so that:

- Students can follow it
- Recruiters can evaluate it quickly
- Engineers can adapt it to production

Sections include:

1. Imports & Setup
2. Dataset Generation
3. Preprocessing
4. TF-IDF Baseline
5. LSTM / BiLSTM Models
6. DistilBERT Model
7. Unified Evaluation
8. End-to-End Demo
9. Model Comparison Summary

---

## **🏗 Architecture Overview**

```
Raw Instruction
        ↓
   Preprocessing
        ↓
  Three Model Families
        ↓
Unified Evaluation Layer
        ↓
 End-to-End Parser Demo
        ↓
 Structured JSON Output
```

---

## **🧪 Example Usage**

Run the final cell in the notebook:

```python
parse_instruction("reset the inverter frequency to 50hz")
```

Output:

```json
{
  "intent": "reset",
  "target": "inverter",
  "parameter": {
    "name": "frequency",
    "value": "50hz"
  }
}
```

---

## **📊 Model Performance Summary**

While scores vary depending on the exact synthetic dataset, trends are consistent:

- **TF-IDF + SVM** performs well for _intent only_
- **BiLSTM** improves multi-output performance
- **DistilBERT** dominates structured extraction

Transformers are recommended if you want:

- robustness to grammar changes
- better understanding of technician lingo
- high accuracy on parameter extraction

---

## **⚙️ Installation**

```bash
pip install -r requirements.txt
```

Or install notebook dependencies manually:

```
transformers
tensorflow
torch
pandas
numpy
scikit-learn
tqdm
matplotlib
```

---

## **📁 Repository Structure**

```
.
├── e2e_intent_parser.ipynb
├── README.md
└── data/

```

---

## **🎯 Why This Project Matters**

Industrial environments need interpretable AI — not black boxes.

Technicians speak in short, imperative commands with variable structure.
This project shows how to build an AI system that can:

- parse real operator instructions
- extract actionable parameters
- integrate into predictive maintenance and control systems
- adapt across equipment types

It’s both a **portfolio piece** and a **template** for real deployments.

---

## **🚀 Next Steps**

Planned improvements:

- Add CRF layer on top of DistilBERT
- Add error-recovery heuristics for incomplete queries
- Export a standalone Python library (`intent_parser/`)
- Production API (FastAPI + lightweight ONNX model)
