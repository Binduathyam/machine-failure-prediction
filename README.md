# Predictive Maintenance Project

## Overview

This project analyzes a predictive maintenance dataset to understand machine behavior and build a machine learning model that predicts equipment failure (`Target`).

## Dataset

* File: `predictive_maintenance.csv`
* The dataset contains machine sensor readings, operational settings, and failure indicators.

## Work Completed (Day 1)

### Data Loading and Inspection

* Loaded the dataset using Pandas
* Checked dataset shape (rows and columns)
* Verified data types of each column
* Checked missing values and duplicate records
* Reviewed summary statistics of numerical features

### Exploratory Data Analysis (EDA)

Created visualizations to understand the dataset:

* Target class distribution
* Boxplots for:

  * Torque [Nm]
  * Tool wear [min]
  * Air temperature [K]
* Compared feature distributions based on the `Target` variable

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook / VS Code

## Project Goal

To predict machine failure using machine sensor data and improve maintenance planning.
