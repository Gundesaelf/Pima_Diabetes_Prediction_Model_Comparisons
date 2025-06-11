🧬 Diabetes Prediction with Naive Bayes 📌 Project Overview This project predicts diabetes diagnoses using the Pima Indians Diabetes Dataset from Kaggle. The focus is on data cleaning, handling missing values, and building a prediction model using Gaussian Naive Bayes. The goal: to create a simple, interpretable, and reasonably accurate medical prediction baseline using the Gaussian model (because I thought it sounded cool).

1️⃣ Tools & Libraries Used 🔧 Programming & Modeling Tools: Python (VS Code) – Model development, data preprocessing, evaluation

📦 Python Libraries: pandas – Data loading and manipulation

numpy – Numerical operations and imputation

scikit-learn – Model building, evaluation, and train/test splitting

colorama – Terminal output formatting (for better readability in CLI)

2️⃣ Data Wrangling Process The dataset originally contained zero-values in some columns. A preprocessing pipeline was developed to handle this:

✔ Initial Overview – Prints the dataset shape and first five rows ✔ Zero Replacement – Replaces 0 values in key medical columns with NaN ✔ Missing Value Imputation – Fills NaN values using column-wise mean ✔ Feature Selection – Drops Age to see how the model performs without it ✔ Train-Test Split – Splits data into 50% train / 50% test using random_state=43

3️⃣ Machine Learning Model 🤖 Model Used: Gaussian Naive Bayes – A lightweight probabilistic classifier well-suited for numeric data and baseline medical models.

🔍 Evaluation Metrics: Accuracy

Precision

Recall

F1 Score

A detailed classification report is printed with color-coded terminal output for clarity.

4️⃣ Key Results & Insights 📊 Current Accuracy: 78.65% – 80.00% ✅ Very solid for a simple Naive Bayes model (but also the best-case scenario due to intentional overfitting) 🔍 Model is better at identifying non-diabetic cases (Class 0) than diabetic cases (Class 1)

Metric (Class 1 - Diabetic) Value Precision 0.77 Recall 0.60 F1 Score 0.67

Mean accuracy: 75.65% Std deviation: 3.74%

🔹 Insight: While the model is accurate overall (after giving it the best chance possible), it tends to miss a lot of diabetic cases. This is common in imbalanced datasets and can be improved with sampling or threshold tuning. I got a little carried away trying to get higher scores and ended up fixating on improving accuracy and not recall.

5️⃣ Future Improvements 💡 ✅ Explore different test sizes to optimize generalization and retain focus on improving recall, not accuracy.

🔄 Try oversampling (e.g. SMOTE) to balance classes 🧪 Compare with other models: Random Forest, Logistic Regression, etc. 📈 Cross-validate Precision, Recall, and F1 Score

🔗 Connect with me https://www.linkedin.com/in/chris-gundes
