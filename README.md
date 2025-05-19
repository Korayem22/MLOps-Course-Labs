# Bank Customer Churn Prediction

This project focuses on predicting customer churn using various machine learning models, tracked and evaluated through MLflow. The models were trained on a publicly available dataset from Kaggle and evaluated using common classification metrics: **accuracy**, **precision**, **recall**, and **F1-score**.

---

##  Dataset

The dataset used in this project is available here:  
[Bank Customer Churn Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/data)

Please download the dataset manually and place it in the appropriate path as expected by the code.

---

##  Project Structure

```

.
├── log\_model.py       # MLflow model logging script
├── models.py          # Model definitions (LogReg, SVM, RF, XGBoost)
├── train.py           # Preprocessing and training pipeline
├── requirements.txt   # Python dependencies
└── README.md          # Project overview and instructions

````

---

##  Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
````
---

##  Models Evaluated

The following models were trained and compared:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Gradient Boosting (XGBoost)

Each model was evaluated on:

* **Accuracy**
* **F1-score**
* **Precision**
* **Recall**

---

##  Best Model

Based on the MLflow evaluation dashboard:

### **Gradient Boosting (XGBoost)** achieved the best results:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.77  |
| F1-score  | 0.77  |
| Precision | 0.78  |
| Recall    | 0.75  |

>  This model consistently outperformed others across all metrics and is recommended for deployment.

---

##  Running the Project

1. Make sure the dataset is downloaded from Kaggle and placed correctly.
2. Install required libraries:

```bash
pip install -r requirements.txt
```

3. Run the training pipeline:

```bash
python train.py
```

4. Start MLflow to track and view metrics:

```bash
mlflow ui
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to view the MLflow dashboard.

---

##  Notes

* This branch of the repository is dedicated to **research purposes only**.
* Experiments compare multiple models including Random Forest, SVM, and GBoost.
* MLflow is used to log metrics, parameters, and model artifacts.
