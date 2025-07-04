# Bank Marketing Analysis: Predicting Term Deposit Subscriptions

## Introduction

This project details a comprehensive data analysis initiative focused on a Portuguese bank's marketing campaign dataset. **This analysis was undertaken as a practical test during my one-month internship at Skillfield, given by a mentor, which I successfully completed and received a certificate for**:
![Image Alt](https://github.com/navyaprashanth/Bank-Marketing-Analysis-Term-Deposit-Prediction/blob/ddad2ce1fc0402c8929e8fe7b0a4e718df0ed712/images/Skillfied%20Mentor.png)

The primary objective was to understand customer behavior and predict the likelihood of clients subscribing to a term deposit. By leveraging various data analysis and machine learning techniques, I aimed to transform raw campaign data into actionable insights, enabling the bank to optimize its future marketing strategies and improve conversion rates.

My goal was to develop a predictive model that accurately identifies potential subscribers and to provide the bank with data-driven recommendations for more targeted and efficient campaigns, ultimately leading to enhanced customer engagement and business growth.

## Project Highlights

Here's a quick look at the core aspects and value delivered by this data analysis project:

📊 **In-Depth Customer Behavior Analysis:** Performed extensive Exploratory Data Analysis (EDA) to uncover key demographic, socio-economic, and campaign-related factors influencing customer subscription decisions.

📈 **Predictive Modeling:** Developed and evaluated classification models (Logistic Regression, Random Forest) to predict term deposit subscriptions, with a focus on F1-Score for the positive class due to dataset imbalance.

💡 **Actionable Insights & Recommendations:** Identified top influential features (e.g., call duration, previous campaign outcome, economic indicators) to provide concrete, implementable strategies for optimizing future marketing efforts.

⚙️ **Impact on Campaign Efficiency:** Enabled the bank to potentially improve targeting accuracy, leading to more efficient resource allocation and a higher return on investment for marketing campaigns.

🐍 **Python for End-to-End Analysis:** Utilized Python for data cleaning, preprocessing, feature engineering, model building, and evaluation.

🧠 **AI-Assisted Development:** Leveraged AI (Gemini) for efficient Python code generation and iterative problem-solving, significantly accelerating the project delivery while maintaining focus on analytical depth and insights.

## Project Structure & Files

This project is delivered as a single, fully-solved Jupyter Notebook containing all analysis steps, alongside the raw dataset. All code, outputs, and visualizations are embedded directly within the notebook file.

* **['Bank_Marketing_Inspection_test.ipynb'](https://github.com/navyaprashanth/Bank-Marketing-Analysis-Term-Deposit-Prediction/blob/9dbff7c995e7b30530dfcfb3bbbc268bf654e629/Bank_Marketing_Inspection_test%20(1).ipynb)**: The main Jupyter Notebook file, fully solved, containing all Python code for EDA, data preprocessing, model building, evaluation, and conclusion. All plots and charts generated during the analysis are embedded within this notebook.
* **['bank-marketing.csv'](https://github.com/navyaprashanth/Bank-Marketing-Analysis-Term-Deposit-Prediction/blob/main/bankmarketing.csv)** : The raw dataset used for this analysis.

## Technologies & Skills Used

This project demonstrates proficiency in end-to-end data analysis, from understanding raw data to generating actionable insights and building predictive models.

* 🐍 **Python (in Jupyter Notebook):** Utilized for all stages of data analysis, including:
    * **Pandas & NumPy:** For data manipulation, cleaning, and preprocessing.
    * **Matplotlib & Seaborn:** For data visualization and exploratory data analysis.
    * **Scikit-learn:** For machine learning model development (Logistic Regression, Random Forest), hyperparameter tuning, and model evaluation (classification reports, confusion matrices, ROC AUC).
   
* 🧠 **AI Assistance (Gemini):** Leveraged for efficient Python code generation, debugging, and accelerating the iterative development process. This approach allowed for rapid prototyping and and enabled a concentrated focus on analytical problem-solving and insightful visualization, rather than manual coding complexities, significantly accelerating project delivery.

## Dataset Overview

The analytical insights and predictive models in this project are based on a publicly available dataset from a Portuguese banking institution's direct marketing campaigns.

**'Bank Marketing Dataset':**
* **Source:** UCI Machine Learning Repository.
* **Content:** Contains information on client demographics (age, job, marital, education), socio-economic indicators (euribor3m, employment variation rate, consumer price index), campaign details (contact, month, day_of_week, duration, campaign, pdays, previous, poutcome), and the target variable (`y` - whether the client subscribed to a term deposit).
* **Key Characteristics:** The dataset is highly imbalanced, with a significantly smaller proportion of clients subscribing to a term deposit, which was a critical consideration during model evaluation.

## Analysis & Modeling Breakdown

This project follows a structured approach, progressing from initial data understanding to advanced predictive modeling. All detailed outputs and full visualizations can be found directly within the **['Bank_Marketing_Inspection_test.ipynb'](https://github.com/navyaprashanth/Bank-Marketing-Analysis-Term-Deposit-Prediction/blob/9dbff7c995e7b30530dfcfb3bbbc268bf654e629/Bank_Marketing_Inspection_test%20(1).ipynb)** notebook. Key insights are highlighted below with representative images.

### 1. Exploratory Data Analysis (EDA)
* **Objective:** To understand the dataset's structure, identify data quality issues, and uncover initial patterns and relationships between features and the target variable.
* **Key Insights:**
    * **Target Imbalance:** Confirmed significant class imbalance (`y = 'no'` >> `y = 'yes'`).
    * **Call Duration (`duration`):** Discovered to be a highly influential feature; longer calls strongly correlated with subscription, though its real-world use for *pre-call* prediction is limited.
       ![Image Alt](https://github.com/navyaprashanth/Bank-Marketing-Analysis-Term-Deposit-Prediction/blob/6baad76627508fa0a3f03aa74a74bca1c3475567/images/Call%20Duration%20vs%20Target.png)

    * **Previous Campaign Outcome (`poutcome`):** Clients with a 'success' in previous campaigns were much more likely to subscribe again.
    ![Image Alt](https://github.com/navyaprashanth/Bank-Marketing-Analysis-Term-Deposit-Prediction/blob/69d30c39c1b44767a13f6e4243f0c1842325ac74/images/poutcome%20vs%20target.png)

    * **Socio-economic Factors:** Features like `euribor3m` (interest rates), `emp.var.rate` (employment variation rate), and `nr.employed` showed clear correlations with subscription rates, highlighting the impact of economic conditions.
      ![Image Alt](https://github.com/navyaprashanth/Bank-Marketing-Analysis-Term-Deposit-Prediction/blob/c618dd22a9b59536d5656b97efe5833840fe7f86/images/emp.var.target%20vs.%20Target.png)
  
    * **Contact Month:** Identified specific months (e.g., March, October, September) with higher subscription rates, suggesting seasonality.
      ![Image A lt](https://github.com/navyaprashanth/Bank-Marketing-Analysis-Term-Deposit-Prediction/blob/e7bfcede0fe9acdadb06b1bfab4faa099c6ffbad/images/month%20vs%20target.png)
      
    * **Campaign Contacts:** Customers who subscribed typically had fewer contacts in the current campaign.
    * **Demographics:** Explored relationships between job, marital status, education, and age with subscription behavior.
     ![Image Alt](https://github.com/navyaprashanth/Bank-Marketing-Analysis-Term-Deposit-Prediction/blob/2a96790fead768ca416498f697cc4b8f622612d4/images/job%20vs%20target.png)

### 2. Data Preprocessing & Feature Engineering
* **Objective:** To prepare the raw data for machine learning models, including handling categorical variables, scaling numerical features, and addressing dataset imbalance.
* **Steps:**
    * **Categorical Encoding:** Converted categorical features into numerical representations using techniques like One-Hot Encoding (`pd.get_dummies`).
    * **Numerical Scaling:** Applied StandardScaler to numerical features to ensure they contribute equally to model training.
    * **Handling Imbalance:** Discussed the challenge of class imbalance and considered strategies (e.g., using evaluation metrics suitable for imbalance like F1-score, or techniques like SMOTE if implemented and detailed in the notebook).

### 3.Model Performance

We trained Logistic Regression and Random Forest Classifiers. Given the imbalanced nature of the dataset (fewer positive cases), we focused on metrics like Precision, Recall, and F1-score in addition to overall accuracy.

### Random Forest Classifier

The Random Forest model generally showed strong performance, balancing precision and recall, which is crucial for identifying potential subscribers.

**Confusion Matrix:**
The confusion matrix for the Random Forest Classifier indicates:
* **True Negatives (TN):** 10688 (Correctly predicted 'No' subscriptions)
* **False Positives (FP):** 277 (Incorrectly predicted 'Yes' subscriptions)
* **False Negatives (FN):** 815 (Incorrectly predicted 'No' subscriptions)
* **True Positives (TP):** 577 (Correctly predicted 'Yes' subscriptions)

![Image Alt](https://github.com/navyaprashanth/Bank-Marketing-Analysis-Term-Deposit-Prediction/blob/89136497fb362796af52684955d7c3de7863ef86/images/confusion%20matrix.png)

### 4. Feature Importance Analysis
* **Objective:** To identify which features were most influential in the Random Forest model's predictions, providing insights into drivers of subscription.
* **Key Findings:**
    * `duration` (call duration) was overwhelmingly the most important feature. (Acknowledged its post-call nature for real-world application limitations).
    * `euribor3m` (a key economic indicator), `poutcome_success` (previous campaign success), `age`, and `campaign` (number of contacts in current campaign) were also highly significant.
   ![Image Alt](https://github.com/navyaprashanth/Bank-Marketing-Analysis-Term-Deposit-Prediction/blob/b8bf186ff76b0378e688c6416791b2fa89163b14/images/Top%2015%20features.png)

## Actionable Business Impact & Value Proposition

This project provides **the bank** with tangible, data-driven insights for optimizing its marketing spend and improving term deposit subscription rates.

1.  **Enhanced Marketing Targeting & Efficiency:**
    * **Action:** By focusing campaigns on customer segments identified as more likely to subscribe (e.g., those with successful previous campaign outcomes, specific job types like retired/students, or contacted during high-conversion months like March/October/September), and by optimizing contact frequency.
    * **Impact:** This enables more precise targeting, reducing wasted efforts on unlikely prospects. This can lead to significantly improved conversion rates among targeted groups, translating directly to a more efficient use of marketing budget and increased new term deposit acquisitions.

2.  **Optimized Campaign Timing & Resource Allocation:**
    * **Action:** Leveraging insights from socio-economic indicators and seasonal trends (contact month) to schedule campaigns during more favorable periods.
    * **Impact:** Strategic timing can lead to higher engagement and conversion rates, potentially reducing the cost per acquisition by concentrating resources when customers are most receptive.

3.  **Data-Driven Decision Making:**
    * **Action:** Providing **bank** managers and marketing teams with a clear, data-backed understanding of which factors truly drive subscriptions, moving beyond intuition.
    * **Impact:** This fosters a data-driven culture, enabling faster and more effective adjustments to marketing strategies and product offerings, leading to continuous improvement in business outcomes.

## Conclusion

This project demonstrates my foundational abilities in:

* **Business Acumen:** Translating a real-world business challenge (improving marketing efficiency) into a solvable data analysis problem within a professional internship context.
* **Data Proficiency:** Handling, cleaning, preprocessing, and analyzing complex datasets.
* **Analytical Storytelling:** Extracting actionable insights from data and communicating them effectively.
* **Machine Learning:** Building, evaluating, and interpreting predictive models.
* **Technological Adaptability:** Leveraging Python's ecosystem and **AI-assisted development (Gemini)** for efficient and impactful project execution. This project, completed in a short timeframe (under 2 hours), showcases my ability to rapidly prototype and deliver solutions by efficiently utilizing modern tools.

This project underscores my commitment to data-driven solutions and my potential to contribute meaningfully to data analysis and machine learning roles.
