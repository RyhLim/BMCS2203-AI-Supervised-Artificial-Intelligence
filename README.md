# Crime Prediction Project (Malaysia)

Project Overview: 
This project applies **Supervised Machine Learning** to predict and explore crime patterns in Malaysia.

It has two main objectives:
1. **Classification**
    - This predicts crime severity levels (Low, Medium, High) for a given context.
2. **Regression**
    - This predicts future crime counts based on state, district, category, type, and date.

Dataset:
- Source: Official Malaysia Government Crime Dataset
- URL: https://data.gov.my/data-catalogue/crime_district

The system is developed using:
    - Jupyter Notebook - for training 
    - Scikit-learn - for modeling and preprocessing
    - Streamlit - for interactive web application 

Implemented Models:
- Random Forest 
- Linear Regression
- Support Vector Machine (SVM) - SVC 

The folder structure:
crime_prediction_project(latest)/
|
|___crime_district.csv # Malaysia crime dataset 
|___train_malaysia.ipynb # Final training notebook
|___app.py # Streamlit user interface
|___artifacts/
|  |__best_regression_model.joblib
|  |__best_classification_model.joblib
|  |__preprocessor.joblib
|  |__evaluation_summary.json
|  |__best_classification_model.joblib
|  |__best_regression_model.joblib
|  |__classes.json
|  |__dropdown_options.json
|  |__evaluation_summary.joblib
|  |__feature_names.json
|  |__label_classes.json
|  |__label_encoder_classes.json
|  |__label_encoder.joblib
|  |__regression_LinearRegression.joblib`  
|  |__regression_RandomForest.joblib
|  |__classification_LogisticRegression.joblib
|  |__classification_RandomForest.joblib
|  |__classification_SVM.joblib
|  |__streamlit_snippet.txt
|___README.md 

1. Install dependencies
```bash
conda create -n crimepred python=3.9
conda activate crimepred
pip install -r requirements.txt

2. Run training notebook
Open Jupyter Notebook and execute train_malaysia.ipynb to regenerate models and artifacts.

3. Run Streamlit app
streamlit run app.py

## System Flow
1. **Data Preprocessing**  
   - Clean dataset, parse dates, encode categorical features, scale numerical data.  
2. **Model Training**  
   - Regression: Linear Regression, Random Forest, (optionally) XGBoost.  
   - Classification: Logistic Regression, Random Forest.  
   - Best models selected by metrics.  
3. **Artifact Saving**  
   - Pipelines and best models saved to `/artifacts` for reuse.  
4. **Streamlit App**  
   - User can select features (state, district, category, type, date).  
   - Predict crime counts (regression).  
   - Predict severity class (classification).  
   - Explore crime trends by state, category, and time.

App features:
Dataset Preview: View dataset and visualize trends.
Predict Crime Count: Regression task, selectable models.
Predict Severity: Classification task, selectable models.
Model Selection: Sidebar dropdown lets you choose which regression and classification models to use.

Results
Regression evaluated using RMSE and R².
Classification evaluated using Accuracy, Precision, Recall, and F1.
Sidebar shows evaluation metrics for each model, with best models highlighted.

References:
Scikit-learn: https://scikit-learn.org/
Streamlit: https://streamlit.io/
Pandas: https://pandas.pydata.org/