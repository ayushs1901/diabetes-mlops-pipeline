# Diabetes Prediction MLOps Pipeline

A small end-to-end machine-learning application that serves a previously trained
Pima Indians Diabetes Random Forest model through a FastAPI API. The repository
contains the model artifact, API, input validation, automated tests, Docker
packaging, and a GitHub Actions workflow.

> **Important:** This is an educational demonstration, not a medical device.
> A prediction is not a diagnosis and must not be used to make healthcare
> decisions. The model's training data, performance, and limitations must be
> evaluated before any real-world use.

## 1. Project overview

### Problem and objective

The project demonstrates how to serve a binary classification model that
predicts whether a Pima Indians Diabetes Database example belongs to the
diabetes class. The model was trained outside this API project, in Colab or
Jupyter as described by the project owner. The API loads the existing
`models/model.joblib`; it does not train a model.

The practical objectives are to:

- Keep a fitted model separate from the serving code.
- Validate incoming measurements with Pydantic.
- Return predictions from a REST API.
- Test API behavior automatically.
- Package the service and its model in a Docker image.
- Run tests and an image build in GitHub Actions.

### What the application does

At startup, `main.py` loads the serialized artifact and extracts its fitted
scikit-learn pipeline. `POST /predict` accepts eight named features, orders
them to match the model's training columns, and returns a binary class and
human-readable label. `GET /` and `GET /health` provide basic service checks.

### Technologies

| Technology | Use |
|---|---|
| Python | Application and tests |
| FastAPI | HTTP API and OpenAPI documentation |
| Pydantic | Request validation |
| pandas | Build a one-row, named-column prediction input |
| scikit-learn | Median imputation and Random Forest classification |
| joblib | Load the saved model artifact |
| pytest and FastAPI `TestClient` | Automated API tests |
| Docker | Package the API, dependencies, and model |
| GitHub Actions | Test and build the Docker image on pushes and pull requests |

### Intended use and limitations

This repository is a college MLOps practical and a demonstration of model
serving. It is not a clinical system. It does not provide authentication,
authorization, rate limiting, patient-data storage, model monitoring,
drift-detection, or clinical validation. The API only checks that all required
fields are present and non-negative (and that selected numeric fields are
finite); it does not enforce medical plausibility ranges.

## 2. End-to-end MLOps workflow

The model artifact and API are present in this repository. A training notebook,
dataset file, training script, and evaluation report are **not** present, so
the detailed training steps shown below are based on the project owner's
description rather than independently reproducible repository code.

```mermaid
flowchart LR
    A[Kaggle Pima dataset<br/>source/data file not in repository]
    B[Colab or Jupyter<br/>training reported by project owner]
    C[Preprocessing<br/>median imputation is in saved pipeline]
    D[Train/test split<br/>details not available]
    E[Random Forest training<br/>100 trees; random_state 42]
    F[Evaluation<br/>metrics not available]
    G[model.joblib<br/>pipeline plus metadata]
    H[FastAPI<br/>load and serve model]
    I[pytest<br/>12 API tests]
    J[Docker image<br/>API plus model]
    K[GitHub repository]
    L[GitHub Actions<br/>tests then Docker build]

    A --> B --> C --> D --> E --> F --> G --> H
    H --> I
    H --> J
    G --> J
    I --> K --> L
    J --> L
```

In the repository's CI workflow, GitHub checks out the code, installs the
pinned Python requirements, runs pytest, and builds the Docker image. The
workflow does not publish the image to a container registry or deploy it.

## 3. Dataset and feature information

### What is verified

- The project owner identifies the data as the **Pima Indians Diabetes
  Database from Kaggle**.
- The saved model metadata records eight input feature names and the target
  display names `No Diabetes` and `Diabetes`.
- The current repository does not contain a dataset CSV, acquisition script,
  or training notebook. Therefore the exact Kaggle source URL, dataset
  version, number of rows and columns, raw column names, and train/test split
  cannot be independently verified from this repository.
- A commonly referenced Kaggle page could not be verified during preparation
  of this README, so no Kaggle URL is presented as confirmed.

### Model input features

The following meanings describe the standard Pima dataset terminology. The
API and saved model use these exact case-sensitive names and order:

| Feature | Meaning |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose measurement |
| `BloodPressure` | Diastolic blood pressure measurement |
| `SkinThickness` | Triceps skin-fold thickness measurement |
| `Insulin` | Serum insulin measurement |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree-function score |
| `Age` | Age in years |

The actual measurement units and measurement protocol are not recorded in the
model artifact or API code.

### Target and class labels

The artifact contains `target_names == ["No Diabetes", "Diabetes"]`; the
classifier's classes are `[0, 1]`. The API maps class `0` to
`not_diabetic` and class `1` to `diabetic`. The artifact does not store a
source target-column name. The dataset file and training code are absent, so
the raw target column name and its original encoding cannot be confirmed here.

### Preprocessing and missing values

The saved fitted pipeline has two steps:

1. `SimpleImputer(strategy="median", missing_values=np.nan)`.
2. `RandomForestClassifier`.

The imputer was fitted as part of the saved pipeline, so it is applied during
prediction as well as being stored with the classifier. Its recorded setting
handles NaN values. The repository does **not** show whether the training
notebook converted zero measurements to missing values. The API accepts zero
and sends it to the model unchanged; it does not treat zero as missing.

### Training details that are not available

The repository does not include evidence for the number of dataset rows, a
train/test split ratio, a split random seed, cross-validation, or evaluation
metrics such as accuracy, precision, recall, F1-score, or ROC-AUC. Do not
quote training metrics from this README. The Random Forest's `random_state=42`
is verified; it is **not** evidence of the train/test split seed.

## 4. Machine-learning model and artifact

### Algorithm

The saved classifier is a scikit-learn `RandomForestClassifier` inside a
scikit-learn `Pipeline`. A Random Forest combines predictions from multiple
decision trees. This can model nonlinear relationships and feature
interactions without feature scaling. These are general properties of the
algorithm, not a claim that a particular training experiment outperformed
other models.

### Verified model settings

The saved classifier has 100 fitted trees and these recorded settings:

| Setting | Saved value | Meaning |
|---|---:|---|
| `n_estimators` | `100` | Number of decision trees |
| `criterion` | `gini` | Split-quality measure |
| `max_features` | `sqrt` | Candidate features considered at each split |
| `max_depth` | `None` | No explicit maximum tree depth |
| `min_samples_split` | `2` | Minimum samples required to split an internal node |
| `min_samples_leaf` | `1` | Minimum samples required at a leaf |
| `bootstrap` | `True` | Trees are trained on bootstrap samples |
| `class_weight` | `None` | No class weighting is configured |
| `random_state` | `42` | Seed for the Random Forest's randomness |

The artifact also records the fitted `SimpleImputer` with median strategy.
The complete saved parameter values can be inspected with
`model.named_steps["classifier"].get_params()`. The fitted artifact shows the
resulting parameter values, but cannot prove whether each value was explicitly
chosen during training or left at its library default.

### Evaluation

No evaluation report, held-out predictions, or metric values are included in
the repository. Consequently, model accuracy and other training/evaluation
metrics are **unknown and not verified**. The API tests verify that the
application returns a valid class-shaped response; they do not measure model
quality.

### What `model.joblib` contains

Inspection of the existing artifact found a serialized dictionary with these
keys:

- `model`: the fitted scikit-learn pipeline (`SimpleImputer` followed by
  `RandomForestClassifier`).
- `feature_names`: the eight ordered input feature names listed above.
- `target_names`: `["No Diabetes", "Diabetes"]`.
- `sklearn_version`: `"1.6.1"`.

The model reports eight input features, the expected `feature_names_in_`, and
100 fitted tree estimators. The API loads the dictionary and uses its `"model"`
entry. The model file is data produced by joblib; it is not retrained by the
API, test suite, Docker build, or CI workflow.

### Compatibility and artifact safety

`requirements.txt` pins `scikit-learn==1.6.1`, which matches the version
recorded in the artifact. It also pins `joblib==1.5.1` and the other direct
application/test dependencies. Python 3.11 is used in the Docker image and
GitHub Actions; the project was also tested locally with Python 3.13.14.

Joblib uses Python serialization mechanisms. Only load a model artifact from
a trusted source; loading an untrusted artifact can execute code. Do not
replace or retrain `models/model.joblib` as part of API startup.

## 5. Verified project structure

This is the project tree; the local `.venv`, Python bytecode, and pytest cache
are excluded from Git:

```text
diabetes-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── models/
│   └── model.joblib
├── .gitignore
├── Dockerfile
├── main.py
├── README.md
├── requirements.txt
└── test_main.py
```

| Path | Purpose |
|---|---|
| `models/model.joblib` | Existing fitted pipeline and model metadata |
| `main.py` | Loads the artifact, defines request validation, and implements API routes |
| `test_main.py` | Endpoint and input-validation tests |
| `requirements.txt` | Pinned Python dependencies |
| `Dockerfile` | Builds an image containing the service and model |
| `.github/workflows/ci.yml` | Runs tests and builds the image in GitHub Actions |
| `.gitignore` | Excludes the virtual environment and generated Python/test files |
| `README.md` | Project documentation |

There is no training notebook, raw dataset, training script, or evaluation
report in the verified project tree. The API cannot be used to reproduce
training from raw data.

## 6. Windows setup and installation

The GitHub repository is:
`https://github.com/ayushs1901/diabetes-mlops-pipeline`

Open PowerShell in the folder where you want the project, then run:

```powershell
git clone https://github.com/ayushs1901/diabetes-mlops-pipeline.git
cd diabetes-mlops-pipeline
```

These commands download the repository and move into its directory. Create a
virtual environment using the Python launcher:

```powershell
py -3 -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation because of the execution policy, either use
the Python executable directly as shown below or follow your institution's
approved PowerShell policy guidance. Do not change a machine-wide execution
policy just for this project.

Install the exact pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify Python and the key packages:

```powershell
python --version
python -c "import fastapi, joblib, pandas, sklearn; print('FastAPI', fastapi.__version__); print('joblib', joblib.__version__); print('pandas', pandas.__version__); print('scikit-learn', sklearn.__version__)"
```

The API and tests import the model at startup, so keep `models\model.joblib`
at its expected location. Do not place `.venv` inside `models`.

## 7. FastAPI application

### Model loading and prediction flow

1. `MODEL_PATH` is resolved relative to `main.py`, making it independent of the
   shell's current directory.
2. `joblib.load` reads `models/model.joblib`.
3. Since this artifact is a dictionary, the application extracts
   `saved_artifact["model"]`.
4. Pydantic validates a JSON request against `PredictionInput`.
5. The endpoint builds a one-row pandas DataFrame with the exact model feature
   names and ordering.
6. `model.predict` runs the fitted imputer/classifier pipeline.
7. The integer class is converted to a label and returned as JSON.

### Endpoints

#### `GET /`

Returns a simple application message.

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/
```

Response, HTTP `200 OK`:

```json
{"message": "Diabetes prediction API"}
```

#### `GET /health`

Reports that the API is responding and whether the model object is loaded.
This is a basic process/model-load check, not a complete readiness check for
all dependencies or downstream services.

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

Response, HTTP `200 OK`:

```json
{"status": "ok", "model_loaded": true}
```

#### `POST /predict`

Accepts all eight features in a JSON body. `Pregnancies` and `Age` are
non-negative integers. The other six features are non-negative finite numbers.
Unknown extra fields are rejected.

Example request:

```powershell
$body = @{
    Pregnancies = 2
    Glucose = 120
    BloodPressure = 70
    SkinThickness = 20
    Insulin = 79
    BMI = 32.0
    DiabetesPedigreeFunction = 0.47
    Age = 33
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/predict `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

The live local API check for this example returned HTTP `200 OK` and:

```json
{"prediction": 0, "label": "not_diabetic"}
```

The predicted class depends on the loaded artifact; the response example is
not a medical conclusion.

For a valid request, the endpoint returns HTTP `200 OK` with `prediction` as
`0` or `1` and `label` as `not_diabetic` or `diabetic`. Missing fields,
negative values, invalid types, non-finite floating-point values, and extra
fields are rejected by FastAPI/Pydantic with HTTP `422 Unprocessable Entity`.

### Start the development server

From the project directory with the virtual environment active:

```powershell
python -m uvicorn main:app --reload
```

The API listens at `http://127.0.0.1:8000`. Interactive Swagger UI is at
`http://127.0.0.1:8000/docs`; the alternative ReDoc page is at
`http://127.0.0.1:8000/redoc`. Stop the development server with `Ctrl+C`.

## 8. Automated testing

`pytest` discovers and runs the functions in `test_main.py`.
`fastapi.testclient.TestClient` sends requests directly to the FastAPI
application, so these tests do not need a separate server process.

The test file contains 12 test cases:

| Test(s) | What is checked |
|---|---|
| `test_home` | `GET /` returns HTTP 200 and the exact message JSON |
| `test_health` | `GET /health` returns HTTP 200 and indicates the model is loaded |
| `test_predict` | A valid sample returns HTTP 200, class `0` or `1`, and one of the two allowed labels |
| `test_predict_rejects_negative_feature` | Each of the eight features is parameterized with a negative value and must return HTTP 422 |
| `test_predict_rejects_missing_feature` | A request missing `Glucose` must return HTTP 422 |

Run the suite from the project directory:

```powershell
python -m pytest -v
```

**Verified result in this workspace:** `12 passed` (one non-failing
Starlette/AnyIO deprecation warning). GitHub Actions also ran the same test
suite successfully for commit
[`279c586`](https://github.com/ayushs1901/diabetes-mlops-pipeline/commit/279c58698051c3f4d15c16ad066996b6547bfd7f).

The tests cover API contract and basic request validation. They do not prove
clinical accuracy, model fairness, generalization, or correctness on a real
population.

## 9. Docker

The `Dockerfile`:

1. Starts from `python:3.11-slim`.
2. Sets `/app` as its working directory.
3. Installs the pinned dependencies.
4. Copies `main.py` and `models/model.joblib` into the image.
5. Exposes port `8000`.
6. Starts Uvicorn on `0.0.0.0:8000`.

Build the image from the project root:

```powershell
docker build -t diabetes-mlops-api:local .
```

Run it in the background:

```powershell
docker run -d --name diabetes-mlops-api -p 8000:8000 diabetes-mlops-api:local
```

Check the endpoints using the PowerShell examples in the API section. View
container logs and stop/remove the container with:

```powershell
docker logs diabetes-mlops-api
docker stop diabetes-mlops-api
docker rm diabetes-mlops-api
```

Docker must be installed and its engine running for local Docker commands.
**Verified locally:** the image built as `diabetes-mlops-api:local`, and a
container returned successful responses from `/`, `/health`, and `/predict`.
The temporary test container was stopped and removed after the checks.
GitHub Actions also verified that the image builds on its runner.

## 10. GitHub repository and continuous integration

The project is published at
[`ayushs1901/diabetes-mlops-pipeline`](https://github.com/ayushs1901/diabetes-mlops-pipeline).

The workflow file `.github/workflows/ci.yml` is named `CI` and is triggered by:

- A push to the `main` branch.
- A pull request.

It uses `ubuntu-latest` and performs these steps:

1. Check out the repository (`actions/checkout@v4`).
2. Set up Python 3.11 (`actions/setup-python@v5`).
3. Install `requirements.txt`.
4. Run `python -m pytest -v`.
5. Build `diabetes-mlops-api:ci` with `docker build`.

The workflow does not push an image to Docker Hub or another registry.

**Verified CI result:** the push-triggered run for commit
`279c58698051c3f4d15c16ad066996b6547bfd7f` completed with conclusion
`success`, including both the pytest and Docker build steps. See the
[GitHub Actions run](https://github.com/ayushs1901/diabetes-mlops-pipeline/actions/runs/40632558147).

For later changes, GitHub displays workflow results in the repository's
**Actions** tab. A green check means the configured steps completed; it does
not mean that the model is clinically validated.

## 11. Viva preparation: questions and answers

**Why keep training separate from the API?**

The API serves a fixed fitted artifact. Separating training from inference
avoids retraining on startup, reduces startup work, and makes the serving
artifact explicit and repeatable. This repository does not include the
training notebook, so training itself must be explained from the Colab/Jupyter
work used to create the artifact.

**What does the preprocessing pipeline do?**

It imputes NaN values using the median learned by `SimpleImputer`, then passes
the transformed features to the Random Forest. Putting both steps in the
pipeline ensures inference uses the fitted imputation statistics.

**Why must feature ordering be exact?**

The estimator associates each column position with a feature learned during
training. The API constructs the DataFrame using the saved eight feature
names in order, reducing the risk of feeding values into the wrong feature
positions.

**What is the difference between HTTP 200 and HTTP 422 here?**

HTTP 200 indicates that a request passed validation and the endpoint returned
a response. HTTP 422 means the request body could not be validated, such as
when a feature is missing or negative.

**What does `random_state=42` mean?**

It controls the Random Forest's pseudorandom operations so the estimator's
training behavior can be repeatable given the same data and software
environment. It does not identify the data split seed.

**Why is `scikit-learn` pinned?**

The serialized artifact records version 1.6.1. Pinning that version aligns
the runtime with the version used when the artifact was saved and reduces
compatibility surprises.

**What is not possible to claim from this repository alone?**

The raw dataset row count and version, target source-column name, data split,
training notebook steps, evaluation metrics, and model performance cannot be
verified because the dataset, notebook, and evaluation report are not in the
repository.

## 12. Known gaps and possible future improvements

- Add the training notebook or a reproducible training pipeline, if permitted
  by the course requirements, without retraining during API startup.
- Record the verified dataset URL/version, target column, preprocessing
  decisions, split strategy, and evaluation metrics alongside the model.
- Add tests for invalid types, non-finite values, extra fields, and explicit
  label-to-class mappings.
- Add deployment-specific controls such as authentication, request logging
  with privacy safeguards, monitoring, and model/data drift checks before
  considering any production use.
- Define and validate domain-appropriate input ranges with a domain expert;
  non-negative validation alone does not establish that a measurement is
  plausible.
