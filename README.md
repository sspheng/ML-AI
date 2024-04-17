What is Machine Learning?
Machine learning enables a machine to automatically learn from data, improve performance from experiences, and predict things without being explicitly programmed. A machine has the ability to learn if it can improve its performance by gaining more data. There are two categories of Machine Learning:

A. SUPERVISED LEARNING

Supervised Learning involves the data which is labeled and the algorithms learn to predict the output from the input data. Supervised learning is where you have input variables (X) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output. And there are two categories of supervised learning:
Regression: target variable is continious like stock market, house price....
Classification: target variable consists of categories like normal or abnormal, spam or no spam, yes or no...
![Supervise learning](https://github.com/sspheng/Machine-Learning/assets/78303183/6887fc78-9f67-4b2f-a0b4-eb4c8bb11b5c)




STEPs INVOLVED IN SUPERVISED LEARNING

1. First Determine the type of training dataset
   
2. Collect/Gather the labelled training data.
   
3. Split the training dataset into training dataset, test dataset, and validation dataset.
   
4. Determine the input features of the training dataset, which should have enough knowledge so that the model can accurately predict the output.

5.Determine the suitable algorithm for the model, such as support vector machine, decision tree, etc.

6. Execute the algorithm on the training dataset. Sometimes we need validation sets as the control parameters, which are the subset of training datasets.

7. Evaluate the accuracy of the model by providing the test set. If the model predicts the correct output, which means our model is accurate

I. REGRESSION

Regression is a statistical method used in finance, investing, and other disciplines that attempts to determine the strength and character of the relationship between one dependent variable (usually denoted by Y) and a series of other variables (known as independent variables). There are two type of Regressions: Linear Regression and Logistic Regression. let see the project about those regression in notebook:

a) Linear Regression

The notebook 'daily bike rental' data is a practice of linear regression that consist of features:'season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed'and target is 'rental'
Linear regresssion has the form as y = ax + b where y = target, x = feature and a = parameter of model. Now we perform train test split to run the model
train: use train set by fitting
test: make prediction on test set.
With train and test sets, fitted data and tested data are completely different
train_test_split(x,y,test_size = 0.3,random_state = 1)
x: features
y: target variables (normal,abnormal)
test_size: percentage of test size. Example test_size = 0.3, test size = 30% and train size = 70%
random_state: sets a seed. If this seed is same number, train_test_split() produce exact same split at each time
fit(x_train,y_train): fit on train sets
score(x_test,y_test)): predict and give accuracy on test sets
We choose parameter of model(a) according to minimum error function that is lost function
In linear regression we use Ordinary Least Square (OLS) as lost function.
OLS: sum all residuals but some positive and negative residuals can cancel each other so we sum of square of residuals. It is called OLS
Score: Score uses R^2 method that is ((y_pred - y_mean)^2 )/(y_actual - y_mean)^2

* Regularized Regression
As we learn linear regression choose parameters (coefficients) while minimizing lost function. If linear regression thinks that one of the feature is important, it gives high coefficient to this feature. However, this can cause overfitting that is like memorizing in KNN. In order to avoid overfitting, we use regularization that penalize large coefficients.

* Ridge regression: First regularization technique. Also it is called L2 regularization.
Ridge regression lost fuction = OLS + alpha * sum(parameter^2)
alpha is parameter we need to choose to fit and predict. Picking alpha is similar to picking K in KNN. As you understand alpha is hyperparameter that we need to choose for best accuracy and model complexity. This process is called hyperparameter tuning.
What if alpha is zero? lost function = OLS so that is linear rigression :)
If alpha is small that can cause overfitting
If alpha is big that can cause underfitting.

* Lasso regression: Second regularization technique. Also it is called L1 regularization.
Lasso regression lost fuction = OLS + alpha * sum(absolute_value(parameter))
It can be used to select important features of the data.

Linear vs Ridge vs Lasso First impression: Linear Feature Selection: 1.Lasso 2.Ridge Regression model: 1.Ridge 2.Lasso 3.Linear

b. Logistic Regression
The noted book about 'Titanic Data' is an practice example for logistic regression with the target of 'Survived' and 'Not Survived" from the accident.
S – L1) Import Package, Functions and Classes
S – L2) Get Data 
S – L3) Create a model and train it
S – L4) Evaluate the model
S – L5) Improve the model

* Regression Metrics
1. Mean Absolute Error
2. Mean Squared Error
3. R^2


II. CLASSIFICATION

Classification is a type of supervised learning. It specifies the class to which data elements belong to and is best used when the output has finite and discrete values. It predicts a class for an input variable as well. There are two types of Classification 	–  Binomial and Multi Class
Classification Use Cases: To find whether an email received is a spam or ham; To identify customer segments; To find if a bank loan is granted; To identify if a kid will pass or fail in an examination.

* Linear Models

	Logistic Regression: Appropriate regression analysis to conduct when dependent variable is binary
 
	Support Vector Machines: Are supervised learning model with assiciated learning algorithms that analyze data for classification and regression analysis.

* Nonlinear models

	K-nearest Neighbors (KNN): A type of instance - based learning or lazy learning where the fucntion only approximate locally and all computation is deferred until function evaluation.
 
	Kernel Support Vector Machines (SVM):Are supervised learning model with assiciated learning algorithms that analyze data for classification and regression analysis
 
	Naïve Bayes: Are a family of simple " Probabilistic classifiers" based on applying Bayes' Theorem with strong (Naive) independensce assumption between the features
 
	Decision Tree Classification: is the most powerful and popular tool for classification and prediction, it is a flowchart like tree structure, where each internal node denoted a test on an attribute, each branch represent an outcome of the test and each leaf 	node (terminal node) holds a class label.
 
	Random Forest Classification: is the robust machine learning algorithm that can use for variety of tasks including regression and classification. It is ensemble method, meaning that a random forest model is madeup for a large number of small decision trees 	called estimators, which is produce their own predictions. Random forest combined the prediction of the estimators to produce more accurate prediction.

* Classification Metric
1. Classification Accuracy
2. Log Loss
3. Area under ROC Curve
4. Confusion Matrix
5. Classification Report

![Different Linear   Logistic ](https://github.com/sspheng/Machine-Learning/assets/78303183/af71db68-f5f5-4223-ab5b-ee8f27cc01f2)

*** ENSEMBLE TECHNIQUE

1. Bagging
	Also known as Bootstrap Aggregation

2. Boosting 
	Gradient Boosting
	Ada Boosting
	XG Boosting

3. Voting


B. UNSUPERVISED LEARNING

Unsupervised Learning involves the data which is unlabeled and the algorithms learn to inherent structure from the input data. Unsupervised learning is where you only have input data (X) and no corresponding output variables. The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data. These are called unsupervised learning because unlike supervised learning above there is no correct answers and there is no teacher. Algorithms are left to their own devises to discover and present the interesting structure in the data. There are two group of unsupervised learning:

	1. Clustering – A clustering problem is where you want to discover the inherent groupings in the data, such as grouping customers by purchasing behavior.
 
	2. Association – An association rule learning problem is where you want to discover rules that describe large portions of your data, such as people that buy X also tend to buy Y Some popular examples of unsupervised learning algorithms are:
 
	a. k-means for clustering problems
 
	b. KNN (K – Nearest Neighbors)![image](https://github.com/sspheng/Machine-Learning/assets/78303183/87e6c358-bee8-4b9e-ab66-8d993d12ec5d)

* Unsupervised Learning Algorithms are:
1. K-means clustering
2. KNN (k-nearest neighbors)
3. Hierarchal clustering
4. Anomaly detection
5. Neural Networks
6. Principle Component Analysis
7. Independent Component Analysis
8. Apriori algorithm
9. Singular value decomposition

