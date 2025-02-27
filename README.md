# Movie Rating Prediction

## Overview
This project aims to predict movie ratings using machine learning techniques. The notebook covers data preprocessing, feature engineering, model training, and evaluation to improve prediction accuracy.

## Dataset
The dataset includes various movie-related features such as genres, directors, actors, and other metadata that contribute to rating prediction.

## Technologies & Libraries Used
The following Python libraries were used:

- `numpy` (Numerical computing)
- `pandas` (Data manipulation and analysis)
- `matplotlib`, `seaborn` (Data visualization)
- `sklearn.model_selection` (Train-test split)
- `sklearn.preprocessing.OneHotEncoder` (Feature Encoding)
- `sklearn.ensemble.RandomForestRegressor` (Machine Learning Model)
- `sklearn.metrics` (Model Evaluation: MAE, MSE, R²)

## Usage Instructions
1. Open Google Colab.
2. Upload the `MovieRatingPrediction.ipynb` file.
3. Run the notebook cell by cell.
4. Ensure you have the required libraries installed using:
   ```python
   !pip install numpy pandas matplotlib seaborn scikit-learn
   ```
5. Follow the steps in the notebook to preprocess data, train the model, and evaluate performance.

## Model Training & Evaluation
- The dataset is split into **training** and **testing** sets.
- A **Random Forest Regressor** is used for rating prediction.
- The model is evaluated using **Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score**.

## Results & Findings
- Data visualization helps understand trends and correlations in movie ratings.
- One-hot encoding improves categorical feature handling.
- The trained model provides rating predictions with a certain level of accuracy.

## Author
This project was implemented on **Google Colab** as part of a movie rating prediction initiative.

## License
This project is open-source and available for further improvements and modifications.
