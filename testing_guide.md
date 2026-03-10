# Vehicle Analytics System - Testing & User Guide

## Overview
The **Vehicle Analytics System** is a Django-based machine learning web application that provides actionable insights into the vehicle market. It utilizes a dummy dataset of vehicle sales to perform Exploratory Data Analysis (EDA) and run predictive models.

The system incorporates three core machine learning functionalities:
1. **Regression Analysis (Random Forest Regressor):** Predicts the estimated selling price of a vehicle based on its manufacturing year, kilometers driven, seating capacity, and the owner's estimated income.
2. **Classification Analysis (Random Forest Classifier):** Categorizes the income level (e.g., Low, Medium, High) of a vehicle owner based on vehicle specifications.
3. **Clustering Analysis (K-Means Clustering):** Segments clients into groups ("Economy", "Standard", "Premium") based on income and estimated vehicle price. It inherently combines the regression output for price prediction to plot the final cluster.

---

## How to Start the Application

To test the application, you first need to ensure the Django development server is running.
1. Open a terminal.
2. Navigate to the project directory:
   ```bash
   cd ~/Downloads/django_ml_project
   ```
3. Activate the Python virtual environment:
   ```bash
   source venv/bin/activate
   ```
4. Start the Django server:
   ```bash
   python manage.py runserver
   ```
5. Open a web browser and go to: `http://127.0.0.1:8000/`

---

## How to Test the Application 

Once the application is loaded in your browser, you can navigate using the left sidebar to test different modules.

### 1. Exploratory Data Analysis (EDA)
- **What it does:** Displays the raw dataset, basic statistical summaries, and a Plotly geographic map showing the concentration of vehicle clients across Rwanda's districts (Exercise 19a).
- **How to test:** 
  - Click on "Data Exploration" in the sidebar.
  - Scroll through the Data Table and Statistical Analysis sections to ensure data is loading correctly.
  - View the "Geographic Distribution" section to confirm the Plotly map renders interactively. You can hover, zoom, and pan around the map.

### 2. Regression Analysis
- **What it does:** Predicts a vehicle's selling price.
- **How to test:**
  - Click on "Regression Analysis" in the sidebar.
  - In the "Input Vehicle Specifications" form, enter some test values (e.g., Year: 2022, Kilometers: 15000, Seats: 5, Income: 60000).
  - Click the **"Predict Market Price"** button.
  - **Expected Result:** The right-hand panel should update from "Waiting..." to display an Estimated Value in dollars. Below, verify that the R² Evaluation Metrics are visible.

### 3. Classification Analysis
- **What it does:** Predicts a vehicle owner's internal income category label.
- **How to test:**
  - Click on "Classification Analysis" in the sidebar.
  - Fill out the form with sample inputs (e.g., Year: 2018, Kilometers: 50000, Seats: 5, Income: 20000).
  - Click the **"Predict Income Category"** button.
  - **Expected Result:** The right-hand panel should update to show the Predicted Category (e.g., "Medium"). Check the Evaluation Metrics section below for model accuracy and test set comparisons.

### 4. Clustering Analysis & Client Segmentation
- **What it does:** Uses a combination of Regression (to predict selling price) and K-Means clustering to place the client into a specific market segment ("Economy", "Standard", "Premium"). It also highlights evaluation metrics heavily refined (Silhouette Score > 0.9) and variance calculations (Exercise 19b).
- **How to test:**
  - Click on "Clustering Analysis" in the sidebar.
  - Enter test data (e.g., Year: 2024, Kilometers: 1000, Seats: 5, Income: 150000).
  - Click **"Run Combined Inference"**.
  - **Expected Result:** The dual-model engine will predict both the value of the vehicle and the assigned Client Cluster badge (e.g., "Premium").
  - Scroll down to "Evaluation Metrics" to verify that the **Silhouette Score** is prominently displayed and properly refined, along with the coefficient of variation details.

---
*End of Document*
