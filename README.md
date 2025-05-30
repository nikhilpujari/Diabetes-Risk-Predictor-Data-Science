# Diabetes Risk Predictor

## Project Overview
The **Diabetes Risk Predictor** is a machine learning project designed to predict diabetes risk based on data from the NHANES (1988-2018) dataset. It uses dietary, questionnaire, and demographic data to train models, including Random Forest, Logistic Regression, XGBoost, Neural Networks, and an ensemble VotingClassifier. The project includes data cleaning, feature engineering, model training, evaluation, and visualization of results to identify individuals at high risk of diabetes.

## Dataset
The dataset is sourced from Kaggle (`nguyenvy/nhanes-19882018`) and includes:
- **Questionnaire Data** (`questionnaire_clean.csv`): Contains diabetes diagnosis (DIQ010) and physical activity data (e.g., PAQ625, PAQ655, PAQ706).
- **Dietary Data** (`dietary_clean.csv`): Includes nutritional intake (e.g., DRXTFIBE, DRXTCARB, DRXTCHOL).
- **Demographic Data** (`demographics_clean.csv`): Includes age, sex, and education level (e.g., DMDEDUC2, DMDEDUC3).

The datasets are merged on the `SEQN` (unique identifier) column to create a comprehensive dataset for analysis.

## Prerequisites
To run the project, ensure you have the following Python libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scikit-posthocs`
- `xgboost`
- `kagglehub`

You can install them using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-posthocs xgboost kagglehub
```

## Project Structure
The project is implemented in a Jupyter Notebook (`Diabetes Risk Predictor Colab.ipynb`) and includes the following steps:
1. **Data Loading**: Downloads and loads the NHANES dataset using `kagglehub`.
2. **Data Preprocessing**:
   - Merges questionnaire, dietary, and demographic datasets on `SEQN`.
   - Checks for missing columns and handles missing values by filtering valid `DIQ010` values (1.0: Yes, 2.0: No, 3.0: Borderline) and filling remaining NaNs with 0.
   - Removes rows with invalid (zero or negative) dietary values.
3. **Feature Engineering**:
   - Creates a `Fiber_to_Carb_Ratio` feature to analyze dietary patterns.
   - Selects 18 features for modeling: 14 nutrient intake features (e.g., Niacin, SaturatedFat, Carbs), 2 engineered features (Unsat_to_Sat_Fat_Ratio, Fat_to_Calorie_Ratio), and 2 demographic features (Age, Sex_encoded).
4. **Exploratory Data Analysis**:
   - Generates a correlation heatmap to visualize the top 30 correlated features.
   - Creates histograms to compare fiber intake across diabetes risk groups.
5. **Model Training**:
   - Trains Random Forest, Logistic Regression, XGBoost, and Neural Network (MLPClassifier) models.
   - Combines models into a VotingClassifier ensemble with soft voting.
6. **Model Evaluation**:
   - Evaluates models using accuracy, precision, recall, F1-score, ROC AUC, and confusion matrices.
   - Plots ROC curves for all models.
7. **Risk Prediction**:
   - Predicts diabetes risk probabilities for patients.
   - Outputs the top 20 highest-risk patients based on the ensemble model.

## Key Results
- **Dataset Shape**: After preprocessing, the dataset contains 746,800 rows and 61 columns.
- **Model Performance**:
  - **Random Forest**: Accuracy: 0.93, ROC AUC: 0.7656
  - **Logistic Regression**: Accuracy: 0.93, ROC AUC: 0.7229
  - **XGBoost**: Accuracy: 0.93, ROC AUC: 0.7679
  - **Neural Network**: Accuracy: 0.93, ROC AUC: 0.7471
  - **Ensemble**: Accuracy: 0.93, ROC AUC: 0.7741
- **Top Risk Patients**: The ensemble model identifies the top 20 patients with the highest diabetes risk scores.

## Visualizations
- **Correlation Heatmap**: Displays the top correlated features to identify key predictors.
- **Histograms**: Show fiber intake distribution across diabetes risk groups (Yes, No, Borderline).
- **ROC Curves**: Compare model performance for diabetes risk prediction.

## Deployment
The project is hosted on GitHub at [https://github.com/nikhilpujari/diabetes-predictor.git](https://github.com/nikhilpujari/diabetes-predictor.git). To deploy and run the project locally or on a cloud platform, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nikhilpujari/diabetes-predictor.git
   cd diabetes-predictor
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure a `requirements.txt` file is created with all dependencies listed (e.g., `pandas`, `numpy`, `scikit-learn`, etc.).

4. **Run the Notebook Locally**:
   - Install Jupyter Notebook:
     ```bash
     pip install jupyter
     ```
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open `Diabetes Risk Predictor Colab.ipynb` and execute the cells.

5. **Run on Google Colab**:
   - Upload `Diabetes Risk Predictor Colab.ipynb` to Google Colab.
   - Update the dataset path if necessary, as Colab may require mounting Google Drive or uploading the dataset manually.
   - Execute the notebook cells in the Colab environment.

6. **Optional: Deploy as a Web Application**:
   - Convert the model to a deployable format (e.g., save the trained ensemble model using `joblib` or `pickle`).
   - Create a web interface using a framework like Flask or Streamlit.
   - Deploy the application on a platform like Heroku, AWS, or Google Cloud Platform.
   - Example for saving the model:
     ```python
     import joblib
     joblib.dump(ensemble_model, 'diabetes_risk_model.pkl')
     ```

## Usage
1. **Download the Dataset**:
   - Use `kagglehub.dataset_download("nguyenvy/nhanes-19882018")` to download the dataset.
2. **Run the Notebook**:
   - Open `Diabetes Risk Predictor Colab.ipynb` in Jupyter Notebook or Google Colab.
   - Ensure the required libraries are installed.
   - Execute the cells sequentially to preprocess data, train models, and generate results.
3. **Interpret Results**:
   - Review the classification reports and ROC curves to assess model performance.
   - Check the `top_risk_patients` DataFrame for high-risk individuals.

## Notes
- The project uses `DIQ010` (diabetes diagnosis) as the target variable, mapped to binary values (1: Diabetes, 0: No Diabetes) for modeling.
- The dataset contains significant missing values, particularly in columns like `ALQ120Q` (551,052 missing) and `PAQ625` (689,193 missing), which are handled during preprocessing.
- The ensemble model uses soft voting to combine predictions from Random Forest, Logistic Regression, and XGBoost for improved performance.

## Future Improvements
- Incorporate additional features (e.g., BMI, physical activity) to improve model accuracy.
- Experiment with hyperparameter tuning for individual models.
- Address class imbalance (e.g., using SMOTE) to improve recall for the diabetes class.
- Explore additional ensemble techniques, such as stacking.

## License
This project is for educational and research purposes only. The NHANES dataset is publicly available via Kaggle under their respective terms.

## Contact
For questions or contributions, please contact the project maintainer via the GitHub repository: [https://github.com/nikhilpujari/diabetes-predictor.git](https://github.com/nikhilpujari/diabetes-predictor.git).