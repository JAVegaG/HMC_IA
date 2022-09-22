# Linear Regression
Linear Regression is a supervised machine learning algorithm, which means that its training is based on a series of inputs to which its outputs are known beforehand.

Regression analysis focuses on modeling the relationship between a dependent variable and one or more independent variables. Therefore, as a result of this modeling, one could predict an expected result given some input. 

Now, when talking about linear regression it is important to understand that it refers to modeling a relationship between one dependent variable and only one independent variable.

In this project, a linear regression algorithm will be implemented; hence, to understand it, it's necessary to interiorize and keep in mind the following:

First, it is necessary to **normalize** the data in order to make learning easier for the model. This means, transforming the original data to guarantee a better performance of the learning algorithm. In this case, the process carried on is called **standardization** and consists of using the mean and standard deviation to transform the original data into equivalent data of zero-mean and standard deviation of one. Mathematically:

$$
\begin{equation}
x' = \frac{x-\overline{x}}{\sigma}
\end{equation}
$$

Then, since this type of regression is linear, the expected output must follow this form:

$$
\begin{equation}
y = \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \dots \theta_n x_n 
\end{equation}
$$

The previous expression can be rewritten as a dot product  between two vectors:

$$
\begin{equation}
\hat{y} = \vec{\theta} \cdot \vec{X}
\end{equation}
$$

Next, to evaluate the performance of the algorithm it is used a **loss function**, which compares the predicted output of the model to the real output. For regression applications, it is very common to use **mean squared error (MSE)** that iterates on every component of the real output vector and the predicted output vector while subtracting from one another and finally dividing the result by the number of components iterated. Mathematically:

$$
\begin{equation}
L_{MSE} = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_{i})^2
\end{equation} 
$$

Later, an **optimization algorithm** is used to adjust the parameters of the model, which would improve the results in the regression. In this example, the **gradient descent** is the algorithm used as shown below, where θ is the parameter to optimize, η is the learning rate, ∇ is the gradient itself, y is the actual value, and L is the loss function:

$$
\begin{equation}
\theta_{t+1} = \theta_{t} - \eta\nabla{L_{MSE}(\hat{y},y)}
\end{equation}
$$

Finally, the partial derivatives of the loss function with respect to the parameters:

$$
\begin{equation}
\frac{\partial L_{MSE}(\hat{y},y)}{\partial \vec{\theta}}
= -
\frac{2}{n} \sum_{i=1}^{n}\vec{X}_i^T \cdot (y_i - \hat{y}_{i})
\end{equation}
$$