# Rental_NY_Prediction
A data science project to explore and analyze NYC rental data, build a predictive pricing model, and derive actionable insights for temporary rental platforms. This repository includes exploratory data analysis, machine learning models, and a comprehensive report with recommendations for business strategies.

The project is divided into three main stages: data preprocessing, exploratory data analysis (EDA), and model training.  This repository includes Jupyter Notebooks for each stage, along with the trained model and the processed dataset.

## Data Description

The dataset contains information on rental listings in NYC.  Key features include:

* **id:** Unique identifier for each listing.
* **nome:** Listing title.
* **host_id:** Unique identifier for the host.
* **host_name:** Name of the host.
* **bairro_group:** Neighborhood group (borough).
* **bairro:** Specific neighborhood.
* **latitude:** Latitude coordinate.
* **longitude:** Longitude coordinate.
* **room_type:** Type of accommodation (e.g., Entire home/apt, Private room, Shared room).
* **price:**  price in USD.
* **minimo_noites:** Minimum nights required for booking.
* **numero_de_reviews:** Number of reviews.
* **ultima_review:** Date of the last review.
* **reviews_por_mes:** Number of reviews per month.
* **calculado_host_listings_count:** Number of listings hosted by the same host.
* **disponibilidade_365:** Number of days available for booking in a year.


## Project Structure

The project is organized as follows:

* **notebooks/:** Contains Jupyter Notebooks for each stage of the project:
    * `01_preprocess_explore.ipynb`: Data cleaning, handling missing values, etc.
    * `02_EDA.ipynb`: Exploratory data analysis, including visualizations and descriptive statistics.
    * `03_prediction_model.ipynb`: Model building, evaluation, and selection.
* **models/:** Stores trained machine learning models (`.pkl` files).
* **data/:** Contains the dataset (raw and cleaned).

## Libraries Used
The following Python libraries were used in this project:

- category-encoders
- folium
- ipywidgets
- matplotlib
- numpy
- pandas
- plotly
- scikit-learn
- scipy
- seaborn

## Transparent Data Analysis: EDA and Report Details in Jupyter Notebooks

For enhanced transparency and reproducibility, the complete data analysis process, including exploratory data analysis (EDA) and model training, is meticulously documented across three Jupyter Notebooks:

* `01_preprocess_explore.ipynb`: Details data preprocessing, including handling missing values and initial exploratory analysis.
* `02_EDA.ipynb`: Presents in-depth EDA, featuring descriptive statistics and insightful visualizations with accompanying interpretations.
* `03_prediction_model.ipynb`: Covers model building, evaluation, feature importance, and model selection, providing explanations for each choice made and the results obtained.

This modular approach allows for a thorough understanding of each stage of the analysis.

## Data Preprocessing

To prepare the dataset for model training, several preprocessing steps were performed. These included identifying and handling missing values (primarily using KNN imputation for critical features) and applying data transformations to improve model accuracy and reliability.

A comprehensive description of the data preprocessing steps, including handling missing values and applying transformations, is provided in the Jupyter Notebook `01_preprocess_explore.ipynb`.  

## EDA:  Driving Actionable Investment Insights

The EDA, documented in `02_EDA.ipynb`, focuses on generating actionable insights for investors in the NYC rental market. Key questions addressed to guide investment decisions include:

* Where is the most attractive location to buy a rental property?
* How do minimum nights and annual availability affect rental income?
* Are there discernible patterns in the names of high-value neighborhoods?

Detailed visualizations and analyses provide evidence-based answers to these questions.
## Model Training and Evaluation

This project employed several **regression models** to predict rental prices.  The following machine learning algorithms were trained and compared:

* Random Forest Regression
* Linear Regression
* Gradient Boosting Regression
* Ridge Regression
* Lasso Regression
* Elastic Net Regression
* Decision Tree Regression
* K-Nearest Neighbors Regression (KNN)
* AdaBoost Regression

Model performance was assessed using the following metrics:

* **Root Mean Squared Error (RMSE)**
* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **R-squared (R²)**
* **Proportion of Variance Explained (PVE)**

These metrics were used to compare model performance and guide the selection of the best-performing model for predicting rental prices. A detailed analysis of the results is available in the `03_prediction_model.ipynb`
  ## Getting Started

1. **Clone the Repository:** Clone this repository to your local machine using:  `git clone https://github.com/EvelynLopesSS/Rental_NY_Prediction`
