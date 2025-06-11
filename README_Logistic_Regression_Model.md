🧬 Diabetes Prediction with Logistic Regression

📌 Project Overview This project predicts diabetes diagnoses using the Pima Indians Diabetes Dataset from Kaggle. The focus is on building a prediction model to compare with my Gaussian Naive Bayes model which analyzes the same dataset with the hope of a better outcome.

1️⃣ Tools & Libraries Used 🔧 Programming & Modeling Tools: Python (VS Code)

📦 Python Libraries: pandas – Data loading and manipulation

numpy – Numerical operations and imputation

scikit-learn – Model building, evaluation, and train/test splitting

colorama – Terminal output formatting (for better readability in CLI)

2️⃣ Data Wrangling Process

I knew the dataset originally contained zero-values in some columns that needed addressing and recreated the solution (with some minor tweaks) from my GaussianNB model but with more of a focus on comparing the classification reports of both of my models to see which performs better.

✔ Initial Overview – Prints the dataset shape and first five rows ✔ Zero Replacement – Replaces 0 values in key medical columns with NaN ✔ Missing Value Imputation – Fills NaN values using column-wise mean ✔ Feature Selection – Drops Age to see how the model performs without it ✔ Train-Test Split – Splits data into 20% train / 80% test using random_state=42 whereas the GaussianNB model used 50% train/test and random_state=43 in an effort to get the best results.

3️⃣ Machine Learning Model 🤖 Model Used: Logistical Regression

🔍 Evaluation Metrics: Accuracy

Precision

Recall

F1 Score

A detailed classification report is printed with color-coded terminal output for clarity.

4️⃣ Key Results & Insights 📊 Current Accuracy: 77.27% ✅ Pretty decent overall but slightly worse than the GuassianNB 🔍 Model is better at identifying non-diabetic cases (Class 0) than diabetic cases (Class 1) - just like the GaussianNB model.

Metric (Class 1 - Diabetic) Value Precision 0.71 Recall 0.62 F1 Score 0.66

Mean accuracy: 76.96% Std deviation: 3.43%

🔹 Insight: This model is accurate overall (just like the GaussianNB), but it too tends to miss a fair amount diabetic cases. The mean accuracy is slightly higher than the GaussianNB model along with a slightly lower std deviation. This consistently low accuracy leads me to believe the dataset being used isn't enough to build an accurate model but I'll need to do more testing to be sure.

5️⃣ Future Improvements 💡 ✅ Explore different models to verify my hypothesis about needing a larger dataset

🔗 Connect with me https://www.linkedin.com/in/chris-gundes
