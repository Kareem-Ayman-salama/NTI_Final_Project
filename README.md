# NTI_Final_Project
Home Credit Default Risk
### **Description for GitHub Repository**

Here’s a detailed and engaging description for your GitHub repository:

---

# **Home Credit Default Risk**

📊 **Predict Credit Default Risk using Machine Learning**

Welcome to the **Home Credit Default Risk** project! This repository showcases a complete end-to-end pipeline for predicting credit default risk using structured data provided by Home Credit. The goal is to identify potential loan defaulters to assist financial institutions in minimizing risk while maximizing customer satisfaction.

---

## **🚀 Features**

- **Data Loading & Preprocessing:**
  - Efficient loading of multiple datasets.
  - Comprehensive preprocessing: handling missing values, encoding categorical features, and scaling numeric features.
  
- **Feature Aggregation & Engineering:**
  - Advanced aggregation techniques for POS, credit card, and installment payments.
  - Intelligent merging of datasets for holistic feature representation.

- **Machine Learning Pipeline:**
  - Supports multiple models: XGBoost, Random Forest, Logistic Regression.
  - Hyperparameter flexibility for tuning models.
  - Robust evaluation using metrics like ROC-AUC and confusion matrix.

- **Visualization:**
  - Feature importance.
  - Correlation matrices.
  - ROC and precision-recall curves.

- **Streamlit Application:**
  - Interactive UI for exploring the pipeline.
  - Upload datasets, preprocess data, train models, and evaluate performance.

---

## **📂 Project Structure**

```plaintext
├── data/                   # Raw and processed datasets
├── src/                    # Core scripts for pipeline
│   ├── load_data.py        # Data loading
│   ├── join.py             # Aggregation and merging
│   ├── preprocessing.py    # Data preprocessing
│   ├── train_model.py      # Model training
│   ├── evaluate_model.py   # Model evaluation
│   └── visualize.py        # Visualizations
├── main.py                 # Pipeline orchestration
├── streamlit_app.py        # Streamlit interactive app
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
```

---

## **📈 Data Workflow**

1. **Load Data**: Load multiple datasets like application, credit card, and POS balance data.
2. **Join & Aggregate**: Combine datasets and engineer new features.
3. **Preprocess**: Handle missing values, encode features, and scale data.
4. **Train Models**: Experiment with XGBoost, Random Forest, and Logistic Regression.
5. **Evaluate & Visualize**: Assess model performance and visualize insights.

---

## **🔧 How to Use**

### **1. Run Locally**
```bash
# Clone the repository
git clone https://github.com/yourusername/home-credit-default-risk.git
cd home-credit-default-risk

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```

### **2. Launch Streamlit App**
```bash
# Run Streamlit app
streamlit run streamlit_app.py
```

---

## **🛠️ Built With**

- **Python**
- **Pandas** and **NumPy**: Data manipulation.
- **Scikit-learn**: Preprocessing and evaluation.
- **XGBoost**: Advanced machine learning.
- **Matplotlib** and **Seaborn**: Data visualization.
- **Streamlit**: Interactive app.

---

## **💡 Future Enhancements**

- Incorporate advanced models (e.g., CatBoost, LightGBM).
- Add automated hyperparameter tuning.
- Extend visualization capabilities.
- Include time-series analysis for sequential datasets.

---

## **📄 License**

This project is licensed under the MIT License.

---

## **🤝 Contributing**

Contributions are welcome! If you’d like to improve the project or fix a bug:
1. Fork the repository.
2. Create your feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add feature-name'`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

---

## **📬 Contact**

For any questions, feel free to reach out via kareem202119883@gmail.com or create an issue.

