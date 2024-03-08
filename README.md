# Loan Application Decision ML Model

## Overview

This repository contains a machine learning model for automating loan application decisions. The model is designed to predict whether a loan application should be approved or denied based on various features provided.

## Features

- Utilizes a RandomForest Classifier machine learning algorithm.
- Input features include [ReasonForLoan,ExistingLoans,Bankruptcy,amount,creditScore,etc].
- Outputs a binary decision - approve or deny.
- Limitation: Demographics Data Needed for Customization and Improvements

## Getting Started

### Prerequisites

- Python 3.12.0
- Dependencies: flask, pandas, sklearn(scikit-learn), seaborn, matplotlib, etc.

### Installation

1. Clone the repository: `git clone https://github.com/bayooyetoro/loan-application-ml.git`
2. Install dependencies: `pip install -r requirements.txt`

## Example Use Case

Consider a scenario where a financial institution wants to automate its loan approval process. By using this machine learning model, the institution can significantly speed up the decision-making process and reduce the manual workload. The model takes into account various factors such as income, credit score, and employment status to provide an accurate prediction.

## Contributors

- Bayonle Oyetoro, Ehitare Ehinome, Ojiko Willams.
Model is still experimental and in development mode with the goal of becoming a publicly available model