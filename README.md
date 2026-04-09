# Customer-Churn-Prediction
# 📉 Telco Customer Churn Prediction (ANN)

Customer churn happens when clients stop doing business with a company. This repository implements a Deep Learning solution using an **Artificial Neural Network (ANN)** to predict churn based on customer demographics, service usage, and account information.

---

## 🛠️ The Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow & Keras
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Metrics:** Scikit-Learn (Classification Report, Confusion Matrix)

---

## 🧠 Project Workflow

### 1. Exploratory Data Analysis (EDA)
* Identified key drivers of churn, such as **Tenure**, **Monthly Charges**, and **Contract Type**.
* Visualized the distribution of churners vs. non-churners across different demographics (gender, senior citizen status).

### 2. Data Preprocessing & Cleaning
* **Handling Missing Values:** Cleaned the `TotalCharges` column by removing empty strings and converting it to a numeric format.
* **Label Encoding:** Converted categorical variables with two values (e.g., 'Yes'/'No') into binary (1/0).
* **One-Hot Encoding:** Applied `get_dummies` for categorical features with multiple levels (e.g., Internet Service type, Payment Method).
* **Feature Scaling:** Used **MinMaxScaler** to normalize features like tenure and charges to a range of 0-1, ensuring the Neural Network converges efficiently.

### 3. ANN Architecture
Built a deep neural network with the following structure:
* **Input Layer:** Matches the number of processed features.
* **Hidden Layers:** Two dense layers with **ReLU** activation for non-linear pattern recognition.
* **Output Layer:** A single neuron with **Sigmoid** activation to output the probability of churn (0 to 1).
* **Compilation:** Optimized using the **Adam** optimizer and `binary_crossentropy` loss function.

---

## 📊 Performance Evaluation
Since churn is often an imbalanced classification problem, the model was evaluated using a comprehensive **Classification Report**:
* **Precision:** Accuracy of positive predictions.
* **Recall:** Ability to find all churners.
* **F1-Score:** The harmonic mean of precision and recall.
* **Confusion Matrix:** Visualized to see the count of True Positives vs. False Positives.

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Manaswi9123/Python-DataScience-Fundamentals.git](https://github.com/Manaswi9123/Python-DataScience-Fundamentals.git)

2. **Install Dependencies:**
   pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
   
3. **Execute:**
      Open Customer Churn Prediction.ipynb in Jupyter or VS Code to see the full analysis and training process.
