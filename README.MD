# WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models

🔧 **WorkflowLLM** is a data-centric framework designed to augment large language models (LLMs) for workflow orchestration across diverse APIs. Leveraging **WorkflowBench**, a meticulously curated dataset, WorkflowLLM supports over **106,763 samples** spanning **1,503 APIs** across **83 applications in 28 categories**.

---

## 🌐 What's New

- **[2024/10/29]** The WorkflowBench dataset is live, empowering WorkflowLLM with the capability to orchestrate workflows across thousands of real-world APIs. Discover the structured training and evaluation scripts provided in this repository to enable end-to-end API orchestration.
- **Improved Data Conversion**: Added tools for converting `.plist` or `.shortcut` files into Python format for easy integration into ML workflows.

---

## 🚀 Overview

WorkflowLLM was built with the goal of advancing workflow orchestration in LLMs by providing:

- A **high-quality instruction-tuning dataset**, WorkflowBench, to train and evaluate models in tool-use capability.
- **Training scripts** for model fine-tuning, built for robustness and scalability.
- **Evaluation scripts** to assess model performance and enhance real-world applicability.

---

## 📂 Dataset
Please download our dataset using the following link: [Google Drive](https://drive.google.com/file/d/1ybvkAL6vU2IIMK0X_N1nsWFmcc7KWs_r/view?usp=sharing) and unzip it into `./data/`.

- **Seed Data**: [./data/seed_data.json](./data/seed_data.json)
- **Synthesized Data**: [./data/synthesized_data.json](./data/synthesized_data.json)

---

## 🔧 Environment Setup

- Python **3.8** is recommended.
- All dependencies are listed in `./requirements.txt` for ease of installation.

```bash
pip install -r requirements.txt
```

---
## 🛠 Data Preprocessing

To convert `.plist` or `.shortcut` data into Python format, use the following script:

```bash
python ./preprocess/Convert_ShortCut_to_Python.py --folder_path {folder_path}
```

Replace {folder_path} with the path to your folder containing `.plist` or `.shortcut` files.

---

## 🚅 Getting Started
### Training the Model
Use the following command to start training:

```bash
sh ./scripts/train.sh {BASE_MODEL_PATH} {DATA_PATH}
```

Replace {BASE_MODEL_PATH} and {DATA_PATH} with the appropriate paths.

### Inference
To run inference, use:

```bash
sh ./scripts/infer.sh {LOAD_PATH}
```
For more detailed training configurations, refer to `./code/train_bf16.py`.

