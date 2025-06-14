🧬 Diabetes Prediction with Naive Bayes, Logistic Regression, and Random Forest 📌 Project Overview This project predicts diabetes diagnoses using the Pima Indians Diabetes Dataset from Kaggle. The focus is on data cleaning, handling missing values, and building a prediction model using Gaussian Naive Bayes. The goal: to create a simple, interpretable, and reasonably accurate medical prediction baseline.

1️⃣ Tools & Libraries Used 🔧 Programming & Modeling Tools: Python (VS Code) – Model development, data preprocessing, evaluation

📦 Python Libraries: pandas – Data loading and manipulation

numpy – Numerical operations and imputation

scikit-learn – Model building, evaluation, and train/test splitting

colorama – Terminal output formatting (for better readability in CLI)

2️⃣ Data Wrangling Process The dataset originally contained zero-values in some columns. A preprocessing pipeline was developed to handle this:

✔ Initial Overview – Prints the dataset shape and first five rows ✔ Zero Replacement – Replaces 0 values in key medical columns with NaN ✔ Missing Value Imputation – Fills NaN values using column-wise mean ✔ Feature Selection – Drops Age to see how the model performs without it ✔ Train-Test Split – Splits data into 50% train / 50% test using random_state=43

3️⃣ Machine Learning Models 🤖 Models Used: Gaussian Naive Bayes, Logistic Regressin, and Random Forest

🔍 Evaluation Metrics: Accuracy

Precision

Recall

F1 Score

A detailed classification report is printed with color-coded terminal output for clarity.


🔗 Connect with me https://www.linkedin.com/in/chris-gundes
