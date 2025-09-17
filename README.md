# MLOps Assignment 1 — Final README (draft, screenshots to add later)

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

|               Model | Accuracy |
| ------------------: | :------: |
| Logistic Regression |    1.0   |
|       Random Forest |    1.0   |
|        SVM (linear) |    1.0   |

> Note: Iris is an easy/linearly separable dataset; all three models achieved perfect accuracy on the chosen split. Random Forest was selected as the model to register.

---

## 4. MLflow logging & model registration (what was done)

* Each model training run was wrapped with `mlflow.start_run()` and logged:

  * `mlflow.log_param()` for hyperparameters
  * `mlflow.log_metric()` for accuracy (and optionally precision, recall, f1)
  * `mlflow.sklearn.log_model()` to store the trained model as an artifact
* MLflow UI was used to compare runs (Experiments tab).
* Best model (Random Forest) was registered in the Model Registry as **`Best_Model_RF`** and promoted to **production** (alias `production`).

---

## 5. Screenshots (placeholders)

*(Replace these placeholders with actual screenshots in the `images/` folder and update paths)*

* **Part 2 – Training**
  `![Part2_Training](images/part2_training.png)`
  *Notebook cell showing dataset shape and printed accuracies.*

* **Part 3 – MLflow Experiments**
  `![Part3_Experiments](images/part3_experiments.png)`
  *MLflow Experiments view showing the three runs.*

* **Part 4 – Model Registry**
  `![Part4_ModelRegistry](images/model_registry.png)`
  *Model Registry listing `Best_Model_RF` and versions.*

* **Part 4 – Production Alias**
  `![Part4_Production](images/production_alias.png)`
  *Version 2 showing alias `production`.*

* **Part 5 – Comparison**
  `![Part5_Comparison](images/experiments_comparison.png)`
  *MLflow Compare view (accuracy metric comparison).*

---

## 6. Project structure

```
mlops-assignment-1/
├── data/                  # (optional) datasets
├── notebooks/
│   └── part2_model_training.ipynb   # main notebook: train, log, register
├── src/                   # (optional) helper scripts
├── models/                # saved models (joblib .pkl)
├── results/               # plots, confusion matrices, csvs
├── images/                # screenshots for report (place here)
└── README.md
```

---

## 7. Environment & required packages

**Recommended:** use Conda with Python 3.10 (this avoids NumPy / pandas compatibility issues).

**Conda (recommended):**

```bash
conda create -n mlops python=3.10 -y
conda activate mlops
conda install pip -y
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn matplotlib joblib mlflow
```

**Or pip-only (if not using conda):**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install numpy pandas scikit-learn matplotlib joblib mlflow
```

**Notes / Known issues:**

* Use Python 3.10 for best binary compatibility with `numpy==1.x` and `pandas`.
* If you see NumPy/Pandas import errors, recreate the conda env with Python 3.10 and re-install packages.
* If `conda` is not in VS Code terminal, use Anaconda Prompt or configure VS Code to use the conda environment interpreter.

---

## 8. How to reproduce (step-by-step)

1. Clone the repo:

   ```bash
   git clone https://github.com/YOUR-USERNAME/mlops-assignment-1.git
   cd mlops-assignment-1
   ```

2. Create & activate environment (see section 7).

3. Open `notebooks/part2_model_training.ipynb` in VS Code (or Jupyter).

4. **Set tracking URI** (important): in notebook cell (before logging/registration) you can set:

   ```python
   import mlflow
   mlflow.set_tracking_uri("file:///c:/full/path/to/mlops-assignment-1/mlruns")
   ```

   * NOTE: If you plan to run MLflow UI against a different directory or the server, set `--backend-store-uri` when starting the UI (see next step).

5. Start MLflow UI in a terminal (point it at the same mlruns used by the notebook):

   ```bash
   # Example: start UI and point to project mlruns
   mlflow ui --backend-store-uri file:///c:/full/path/to/mlops-assignment-1/mlruns --default-artifact-root file:///c:/full/path/to/mlops-assignment-1/mlruns
   ```

   * Default UI address: `http://127.0.0.1:5000`

6. In the notebook, run cells in order:

   * Load dataset, split, train models (LogReg, RF, SVM)
   * Evaluate and print metrics
   * Save models to `/models` (joblib)
   * Log each model to MLflow with `mlflow.start_run()` and `mlflow.sklearn.log_model(...)`

7. Refresh MLflow UI → go to **Experiments** → you should see runs. Use **Compare** to visualize metrics.

8. Register the best model (Random Forest) from notebook (example):

   ```python
   import mlflow
   mlflow.set_tracking_uri("http://127.0.0.1:5000")
   model_name = "Best_Model_RF"
   with mlflow.start_run(run_name="Register Random Forest"):
       mlflow.sklearn.log_model(rf_model, artifact_path="rf_model", registered_model_name=model_name)
   ```

   * Or register from MLflow UI by selecting a run → click **Register model**.

9. Promote the registered model to production:

   * In UI: **Models** → select model → version → **Promote** (or add alias `production`).
   * Or in code use MLflow registry client.

10. Load the production model and test:

    ```python
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    loaded_model = mlflow.sklearn.load_model("models:/Best_Model_RF@production")
    import numpy as np
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    print(loaded_model.predict(sample))
    ```

---

## 9. Commands to push everything to GitHub

```bash
git add .
git commit -m "Complete: training, MLflow logging, model registration details"
git push origin main
```

---

## 10. Deliverables checklist (for submission)

* [x] Public GitHub repo `mlops-assignment-1` with code & notebook
* [x] `models/` folder with saved models (`.pkl`)
* [x] MLflow runs logged (Experiments)
* [x] Model registered in MLflow Registry (`Best_Model_RF`)
* [ ] Report with screenshots (`images/` folder + `MLOps_Assignment_Report_RANA_AHMED.docx`) ← **to add tomorrow**
* [ ] README final (this file) including embedded screenshots ← **to finalize after you upload screenshots**

---

## 11. Final notes & next steps (what we will do tomorrow)

* You will capture the key screenshots listed in section 5 and save them in `images/`.
* Upload the report document (`MLOps_Assignment_Report_RANA_AHMED.docx`) here for me to review.
* I will then:

  * Insert the screenshots in README and polish wording.
  * Produce the final README version ready to push.
  * Provide a short submission-ready PDF if you want.

---

## 12. Contact

If anything breaks while reproducing these steps, paste the terminal/error output here and I’ll guide you line-by-line.

---

**Done for now.**
Paste this content into your `README.md` file, save, commit and push (`git add README.md && git commit -m "Add Part 4 README draft" && git push`).
Tomorrow, upload screenshots and the report file and I’ll update the README and finalize everything.
