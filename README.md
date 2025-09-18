# MLOps Assignment 1

**Author:** RANA AHMED
**Repo:** `mlops-assignment-1`

---

## 1. Overview / Objective

This project demonstrates a basic MLOps workflow:

* GitHub for version control.
* Train multiple ML models on the Iris dataset.
* Log experiments (parameters, metrics, artifacts) with **MLflow**.
* Register the best model in the **MLflow Model Registry** and promote it to **production**.
* Provide reproducible instructions so anyone can run the notebook and reproduce the results.

---

## 2. Problem statement & dataset

**Problem:** Classify Iris flowers into one of three species using measured sepal/petal dimensions.
**Dataset:** Iris (built into `scikit-learn`): 150 samples, 4 numeric features:

* `sepal length (cm)`
* `sepal width (cm)`
* `petal length (cm)`
* `petal width (cm)`

---

## 3. Models trained & selection

Trained three classic models:

* **Logistic Regression** (`max_iter=200`)
* **Random Forest** (`n_estimators=100, random_state=42`)
* **Support Vector Machine (SVM)** (`kernel='linear'`)

**Summary of results (test set):**

| Model               | Accuracy |
| ------------------- | :------: |
| Logistic Regression |    1.0   |
| Random Forest       |    1.0   |
| SVM (linear)        |    1.0   |

> Note: Iris is an easy/linearly separable dataset; all three models achieved perfect accuracy on the chosen split. Random Forest was selected as the model to register.

---

## 4. MLflow logging & model registration

* Each model training run was wrapped with `mlflow.start_run()` and logged:

  * `mlflow.log_param()` for hyperparameters
  * `mlflow.log_metric()` for accuracy
  * `mlflow.sklearn.log_model()` to store the trained model
* MLflow UI was used to compare runs (Experiments tab).
* Best model (Random Forest) was registered in the Model Registry as **`Best_Model_RF`** and promoted to **production** (alias `production`).

---

## 5. Screenshots

### Part 3 – Model Registry

### Part 4 – Production Alias

### Part 5 – Experiment Comparison

Attached in the document file for submission

---

## 6. Project structure

```
mlops-assignment-1/
├── notebooks/
│   └── part2_model_training.ipynb   # main notebook: train, log, register
├── models/                          # saved models (joblib .pkl)
├── results/                         # confusion matrix & classification report
├── images/                          # screenshots for report
└── README.md
```

---

## 7. Environment & required packages

**Recommended:** Conda with Python 3.10

```bash
conda create -n mlops python=3.10 -y
conda activate mlops
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn matplotlib joblib mlflow
```

If not using conda:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

pip install numpy pandas scikit-learn matplotlib joblib mlflow
```

---

## 8. How to reproduce

1. Clone the repo:

   ```bash
   git clone https://github.com/YOUR-USERNAME/mlops-assignment-1.git
   cd mlops-assignment-1
   ```

2. Activate environment (see section 7).

3. Open notebook:
   `notebooks/part2_model_training.ipynb`

4. Start MLflow UI:

   ```bash
   mlflow ui --backend-store-uri file:///c:/full/path/to/mlops-assignment-1/mlruns
   ```

   Default: [http://127.0.0.1:5000](http://127.0.0.1:5000)

5. Run notebook cells → models will be trained, logged, and registered.

6. In MLflow UI → check **Experiments** and **Models** tabs.

7. Load production model and test:

   ```python
   import mlflow
   mlflow.set_tracking_uri("http://127.0.0.1:5000")
   loaded_model = mlflow.sklearn.load_model("models:/Best_Model_RF@production")
   print(loaded_model.predict([[5.1, 3.5, 1.4, 0.2]]))
   ```

---

## 9. Deliverables checklist

* [x] Code + Notebook
* [x] Results (confusion matrix, report)
* [x] MLflow runs logged
* [x] Model registered & promoted to production
* [x] README with problem, dataset, model comparison, registry, and screenshots


