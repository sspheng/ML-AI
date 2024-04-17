What is Machine Learning?
Machine learning enables a machine to automatically learn from data, improve performance from experiences, and predict things without being explicitly programmed. A machine has the ability to learn if it can improve its performance by gaining more data.

There are two categories of Machine Learning:
A. SUPERVISED LEARNING: Supervised Learning involves the data which is labeled and the algorithms learn to predict the output from the input data. Supervised learning is where you have input variables (X) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output. And there are two categories of supervised learning:
Regression: target variable is continious like stock market, house price....
Classification: target variable consists of categories like normal or abnormal, spam or no spam, yes or no...

We will learn linear and logistic regressions
This daily bike share data is a linear regression so the features are and pelvic_incidence of abnormal
I consider feature is pelvic_incidence and target is sacral_slope
Lets look at scatter plot so as to understand it better
reshape(-1,1): If you do not use it shape of x or y becaomes (210,) and we cannot use it in sklearn, so we use shape(-1,1) and shape of x or y be (210, 1).
Now we have our data to make regression. In regression problems target value is continuously varying variable such as price of house or sacral_slope. Lets fit line into this points.


Linear regression

y = ax + b where y = target, x = feature and a = parameter of model
We choose parameter of model(a) according to minimum error function that is lost function
In linear regression we use Ordinary Least Square (OLS) as lost function.
OLS: sum all residuals but some positive and negative residuals can cancel each other so we sum of square of residuals. It is called OLS
Score: Score uses R^2 method that is ((y_pred - y_mean)^2 )/(y_actual - y_mean)^2

K-NEAREST NEIGHBORS (KNN)¶
KNN: Look at the K closest labeled data points
Classification method.
First we need to train our data. Train = fit
fit(): fits the data, train the data.
predict(): predicts the data
If you do not understand what is KNN, look at youtube there are videos like 4-5 minutes. You can understand better with it.
Lets learn how to implement it with sklearn
x: features
y: target variables(normal, abnormal)
n_neighbors: K. In this example it is 3. it means that Look at the 3 closest labeled data points
Well, we fit the data and predict it with KNN.
So, do we predict correct or what is our accuracy or the accuracy is best metric to evaluate our result? Lets give answer of this questions
Measuring model performance:
Accuracy which is fraction of correct predictions is commonly used metric. We will use it know but there is another problem

As you see I train data with x (features) and again predict the x(features). Yes you are reading right but yes you are right again it is absurd :)


Therefore we need to split our data train and test sets.

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


CROSS VALIDATION¶
As you know in KNN method we use train test split with random_state that split exactly same at each time. However, if we do not use random_state, data is split differently at each time and according to split accuracy will be different. Therefore, we can conclude that model performance is dependent on train_test_split. For example you split, fit and predict data 5 times and accuracies are 0.89, 0.9, 0.91, 0.92 and 0.93, respectively. Which accuracy do you use? Do you know what accuracy will be at 6th times split, train and predict. The answer is I do not know but if I use cross validation I can find acceptable accuracy.
Cross Validation (CV)

K folds = K fold CV.
Look at this image it defines better than me :)
When K is increase, computationally cost is increase
cross_val_score(reg,x,y,cv=5): use reg(linear regression) with x and y that we define at above and K is 5. It means 5 times(split, train,predict)

Regularized Regression¶
As we learn linear regression choose parameters (coefficients) while minimizing lost function. If linear regression thinks that one of the feature is important, it gives high coefficient to this feature. However, this can cause overfitting that is like memorizing in KNN. In order to avoid overfitting, we use regularization that penalize large coefficients.

Ridge regression: First regularization technique. Also it is called L2 regularization.
Ridge regression lost fuction = OLS + alpha * sum(parameter^2)
alpha is parameter we need to choose to fit and predict. Picking alpha is similar to picking K in KNN. As you understand alpha is hyperparameter that we need to choose for best accuracy and model complexity. This process is called hyperparameter tuning.
What if alpha is zero? lost function = OLS so that is linear rigression :)
If alpha is small that can cause overfitting
If alpha is big that can cause underfitting. But do not ask what is small and big. These can be change from problem to problem.
Lasso regression: Second regularization technique. Also it is called L1 regularization.
Lasso regression lost fuction = OLS + alpha * sum(absolute_value(parameter))
It can be used to select important features od the data. Because features whose values are not shrinked to zero, is chosen by lasso regression
In order to choose feature, I add new features in our regression data

Linear vs Ridge vs Lasso First impression: Linear Feature Selection: 1.Lasso 2.Ridge Regression model: 1.Ridge 2.Lasso 3.Linear

ROC Curve with Logistic Regression¶
logistic regression output is probabilities
If probability is higher than 0.5 data is labeled 1(abnormal) else 0(normal)
By default logistic regression threshold is 0.5
ROC is receiver operationg characteristic. In this curve x axis is false positive rate and y axis is true positive rate
If the curve in plot is closer to left-top corner, test is more accurate.
Roc curve score is auc that is computation area under the curve from prediction scores
We want auc to closer 1
fpr = False Positive Rate
tpr = True Positive Rate
If you want, I made ROC, Random forest and K fold CV in this tutorial. https://www.kaggle.com/kanncaa1/roc-curve-with-k-fold-cv/


HYPERPARAMETER TUNING
As I mention at KNN there are hyperparameters that are need to be tuned

For example:
k at KNN
alpha at Ridge and Lasso
Random forest parameters like max_depth
linear regression parameters(coefficients)
Hyperparameter tuning:
try all of combinations of different parameters
fit all of them
measure prediction performance
see how well each performs
finally choose best hyperparameters
This process is most difficult part of this tutorial. Because we will write a lot of for loops to iterate all combinations. Just I am kidding sorry for this :) (We actually did it at KNN part)
We only need is one line code that is GridSearchCV
grid: K is from 1 to 50(exclude)
GridSearchCV takes knn and grid and makes grid search. It means combination of all hyperparameters. Here it is k.

UNSUPERVISED LEARNING¶
Unsupervised learning: It uses data that has unlabeled and uncover hidden patterns from unlabeled data. Example, there are orthopedic patients data that do not have labels. You do not know which orthopedic patient is normal or abnormal.
As you know orthopedic patients data is labeled (supervised) data. It has target variables. In order to work on unsupervised learning, lets drop target variables and to visualize just consider pelvic_radius and degree_spondylolisthesis


KMEANS
Lets try our first unsupervised method that is KMeans Cluster
KMeans Cluster: The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity
KMeans(n_clusters = 2): n_clusters = 2 means that create 2 cluster

