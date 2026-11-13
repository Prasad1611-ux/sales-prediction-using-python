# sales-prediction-using-python
A Python project that explores advertising spend across TV, radio, and newspaper channels, then compares Linear Regression and Random Forest models to predict product sales. Includes a Jupyter notebook, visualizations, and model evaluation.
# Sales Prediction Using Python

This project predicts product sales from advertising spend across TV, radio, and newspaper channels. It includes a runnable Python script and a Jupyter notebook for exploration and model comparison.

## Project files

- `main.py` — loads the dataset, performs exploratory analysis, trains models, reports test metrics, and plots residuals.
- `sales_prediction.ipynb` — step-by-step notebook version of the analysis.
- `data/Advertising.csv` — the Advertising dataset used in the project.

## Requirements

Python 3.9 or later is recommended. Install the required packages with:

```bash
python -m pip install pandas scikit-learn matplotlib seaborn jupyter
```

## Run the Python script

From the project directory, run:

```bash
python main.py
```

The script loads `data/Advertising.csv`. If the file is absent, it downloads the dataset from the public [Advertising.csv source](https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv).

The script prints data checks and descriptive statistics, displays exploratory plots, trains Linear Regression and Random Forest models, compares MAE/RMSE/R² on a held-out test set, and plots residuals for the model with the lower test RMSE.

To open the notebook:

```bash
jupyter notebook sales_prediction.ipynb
```

## Dataset and features

The dataset contains 200 observations and four columns:

| Column | Description |
| --- | --- |
| `tv` | TV advertising spend |
| `radio` | Radio advertising spend |
| `newspaper` | Newspaper advertising spend |
| `sales` | Product sales target |

The analysis includes missing-value checks, descriptive statistics, a pairplot, individual sales-versus-channel scatter plots, and a correlation heatmap.

## Example test-set results

Using an 80/20 train/test split with `random_state=42`, the models produced:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Random Forest | 0.662 | 0.791 | 0.980 |
| Linear Regression | 1.461 | 1.782 | 0.899 |

On this split, TV had the highest Random Forest feature importance. The linear model's largest coefficient by absolute value was for radio. These are predictive associations from this dataset, not causal estimates or guarantees of future performance.
