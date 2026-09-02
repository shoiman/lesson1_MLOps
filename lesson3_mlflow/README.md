# MLflow practical example: Telco customer churn

This project is a step-by-step guide to using MLflow for a real-world machine learning task. We use a Telco Customer Churn dataset (about 100,000 records) to show how MLflow handles everything from initial experiments to model deployment.

## What's inside

We've organized the project into a series of notebooks and scripts that cover different parts of the MLflow ecosystem:

- **`01_data_exploration.ipynb`**: The basics. We'll do some EDA, train a baseline Logistic Regression model, and log our first run.
- **`02_experiment_tracking.ipynb`**: Comparing different models (Random Forest, XGBoost) and using nested runs to keep hyperparameter searches organized.
- **`03_model_registry.ipynb`**: How to version models, use aliases like `champion`, and load them back for predictions.
- **`04_autologging.ipynb`**: Using MLflow's autologging feature to save time, and how to mix it with custom manual metrics.
- **`05_remote_mlflow.ipynb`**: Moving from a local SQLite setup to a full remote stack with PostgreSQL and MinIO (using Docker).
- **`src/` scripts**: CLI versions of the training and registration logic for when you're ready to move out of notebooks.
- **`MLproject`**: A file that lets you run the whole pipeline reproducibly with `mlflow run`.

## Getting started

### 1. Set up your environment

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the notebooks

Start Jupyter and go through the notebooks in order. Each one builds on the last:

```bash
jupyter notebook notebooks/
```

- **Notebook 01**: Baseline model and basic logging.
- **Notebook 02**: Hyperparameter tuning and run comparisons.
- **Notebook 03**: Working with the Model Registry.
- **Notebook 04**: Autologging patterns.
- **Notebook 05**: Remote tracking with Docker.

### 3. Open the MLflow UI

Once you've logged some runs, you can view them by running:

```bash
mlflow ui --port 5000
```

Then head to [http://127.0.0.1:5000](http://127.0.0.1:5000) to see your experiments.

## Running on a remote server (Notebook 05)

If you're working in a team, you'll want a central MLflow server. I've included a `docker-compose.yml` that sets up a production-like environment (PostgreSQL for metadata and MinIO for artifacts).

To start the stack:

```bash
docker-compose up -d
```

Notebook 05 will show you how to point your client at this server (`http://localhost:5000`). You can also use it from the CLI:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/train.py --model-type rf
```

## Using the CLI and MLflow projects

You can run the training scripts directly or use `mlflow run` for better reproducibility.

### Using `mlflow run` (recommended)

This uses the `MLproject` definition to run scripts in a consistent environment. I recommend using `--env-manager=local` to use your existing virtual environment:

```bash
# Train a Random Forest
mlflow run . --env-manager=local

# Train XGBoost with specific params
mlflow run . --env-manager=local -e train -P model_type=xgb -P params='{"learning_rate": 0.1}'

# Register the best model based on a metric
mlflow run . --env-manager=local -e register -P experiment_name=telco-churn -P metric=roc_auc
```

### Running scripts directly

If you don't want to use MLflow Projects:

```bash
# Train
python src/train.py --model-type xgb --params '{"n_estimators": 200}'

# Register the best version
python src/register_model.py --metric roc_auc

# Predict on new data
python src/predict.py --version 1 --output predictions.csv
```

## The dataset

We're using the **Telco Customer Churn** dataset (100k rows). It includes demographic info, service details, and account data. You can find full column descriptions in [`datadictionary.md`](datadictionary.md).

## Core MLflow concepts we cover

- **Experiments & runs**: How to group and track your work.
- **Params & metrics**: Logging what goes in and what comes out.
- **Artifacts**: Saving models, plots, and data files.
- **Model registry**: Versioning models and using aliases like `champion`.
- **Autologging**: Letting MLflow handle the logging for you.
- **Remote tracking**: Sharing runs with a team via PostgreSQL and S3/MinIO.
- **Projects**: Making your code reproducible for others.
