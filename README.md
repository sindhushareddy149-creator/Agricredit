Project Title  : Intelligent Credit Scoring System for Sustainable Finance

Project Description

The Agricultural Credit Risk Assessment System is a Machine Learning-based web application developed to assess the creditworthiness of farmers for agricultural loan approval. The system evaluates multiple factors including repayment history, soil health, weather conditions, and market conditions to generate a credit risk score and classify farmers into Low, Medium, or High risk categories.
The application also incorporates Explainable Artificial Intelligence (XAI) to provide transparency in predictions by showing how each factor contributes to the final risk assessment.

Project Structure
The project is organized into the following components:

• Web Application
	app.py
	index.html

• Machine Learning Module
	model_training.py
	AgriCredit_Model.pkl

• Dataset Calculation Module
	past_repayment_score.py
	soil_score.py
	weather_score.py
	market_score.py
	dataset_merge.py

• Datasets
	Soil_Dataset.xlsx
	Farmer_Dataset.xlsx
	Weather_Dataset.xlsx
	Market_Dataset.xlsx
	Final_Dataset_Training.xlsx
	Final_Dataset_Full.xlsx

File Description

1. app.py
	Flask backend application.
	Handles user input, prediction requests, and result generation.
	Connects to MongoDB for data storage and retrieval.

2. index.html
	Frontend user interface for farmer and banker interactions.

3. model_training.py
	Trains and evaluates machine learning models.
	Generates and saves the final trained model.

4. AgriCredit_Model.pkl
	Saved machine learning model used for prediction.

5. Dataset_Calculations Folder
	past_repayment_score.py : Calculates Past Repayment Score from repayment history.
	soil_score.py : Calculates Soil Health Score using soil nutrient values and soil type.
	weather_score.py : Calculates Weather Score using weather parameters.
	market_score.py : Calculates Market Score using market price information.
	dataset_merge.py : Merges all calculated scores and generates the final datasets.
	
Datasets Used

1. Soil_Dataset.xlsx
	Contains soil nutrient values and soil type information.
2. Farmer_Dataset.xlsx
	Contains farmer loan repayment records and repayment history.
3. Weather_Dataset.xlsx
	Contains weather parameters and yearly weather scores.

4. Market_Dataset.xlsx
	Contains agricultural market price information used for market score calculation.

Project Scope

Crop   : Paddy
Location  : Deverakonda, Telangana
Training Period : 2015–2025
Prediction Period : 2015–2030

 Generated Outputs

1. Final_Dataset_Training.xlsx
	Historical dataset used for machine learning model training.

2. Final_Dataset_Full.xlsx
	Complete dataset used for prediction and analysis.

3. AgriCredit_Model.pkl
	Trained machine learning model.

4. Prediction Results
	Risk Score
	Risk Category (Low Risk / Medium Risk / High Risk)

5. Explainable AI (XAI) Results
	Contribution of Repayment Score
	Contribution of Soil Health Score
	Contribution of Weather Score
	Contribution of Market Score
	Explanation of factors influencing the prediction

 Technologies Used

•	Python
•	Pandas
•	NumPy
•	Scikit-Learn
•	XGBoost
•	Flask
•	MongoDB
•	HTML/CSS

 How to Run

1. Ensure MongoDB is installed and running.

2. Execute the scripts inside the Dataset_Calculations folder to prepare the datasets.

3. Run model_training.py to train the machine learning model.

4. Start the application using:
   python app.py

5. Open the application in a web browser.

Project Outcome

The system predicts agricultural credit risk by generating a risk score and classifying farmers into:
	Low Risk
	Medium Risk
	High Risk
The application also provides Explainable AI insights that clearly show how each factor contributes to the final decision, improving transparency, trust, and interpretability in agricultural credit assessment.
