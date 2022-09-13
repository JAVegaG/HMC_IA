# Project 2
The binary logistic model is a statistical model that helps estimate the probability of an outcome being true given two possible options. Therefore, it is usually used on simple classification tasks. The goal of this project is to explain how the binary logistic model works through the math behind it, and its Python implementation. For the mathematical part of it, vector notation will be used since it would later enable faster computation time.

In this project, the logistic model will be seen as if it would be implemented on a neural network to simplify the description of its functioning when used for addressing a classification task. Hence, to understand it, it's necessary to interiorize and keep in mind the following equations:

* First, the dot product of the input vector (X) and the weights of the neuron (W) is performed, and the result is added to the neuron's bias (b): 

$$
\begin{equation}
z = \vec{W} \cdot \vec{X} + b
\end{equation}
$$

* Then, the Sigmoid function is used as the activation function (σ) of the neuron. The result of this function would be a value between 1 and 0 corresponding to one of the two classes:

$$
\begin{equation}
\sigma(z) = \hat{y}(z) =
\frac{1}{1+\exp{(-z)}}
\end{equation}
$$

* Next, an optimization algorithm is used to adjust the parameters of the model, which would give better results in the classification. In this example, the gradient descendent is the algorithm used as shown below, where θ is the parameter to optimize (weights or bias), η is the learning rate, ∇ is the gradient itself, y is the actual value, and L is the loss function:

$$
\begin{equation}
\theta_{t+1} = \theta_{t} - \eta\nabla{L(f(x;\theta),y)}
\end{equation}
$$

* Later, the binary cross entropy loss function is used to determine how exact is the prediction of the model comparing it to the real value:

$$
\begin{equation}
L_{CE} = -\frac{1}{m}\sum_{i=1}^{m}{y_i\log{(\hat{y}_i)}}+(1-y_i)\log{(1-\hat{y}_i)}
\end{equation}
$$

* Finally, the partial derivatives of the loss function with respect to the parameters:

$$
\begin{equation}
\frac{\partial L_{CE}(\hat{y},y)}{\partial \vec{W}} = \frac{1}{m}(\hat{y}-y)\vec{X}^T_i
\qquad
\frac{\partial L_{CE}(\hat{y},y)}{\partial \vec{b}} = \frac{1}{m}(\hat{y}-y)
\end{equation}
$$