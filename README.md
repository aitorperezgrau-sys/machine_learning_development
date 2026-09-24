# Machine learning development

This repository contains machine learning algorithms developed as a training exercise. Currently it features a Linear Regression algorithm.

## Linear Regression algorithm

- [Overview](#overview)
- [Implementation notes](#implementation-notes)
- [The cost function](#the-cost-function)
- [Gradient descent](#gradient-descent)
  - [Why gradient descent finds the global minimum](#why-gradient-descent-finds-the-global-minimum)
- [Step implementation](#step-implementation)
  - [Physical meaning of the sign and learning rate](#physical-meaning-of-the-sign-and-learning-rate)
- [Usage](#usage)
- [Results](#results)
  - [Comparison of the numerical derivative against derivative expression](#comparison-of-the-numerical-derivative-against-derivative-expression)
  - [Comparison of the zero gradient convergence method against zero difference between the cost error functions](#comparison-of-the-zero-gradient-convergence-method-against-zero-difference-between-the-cost-error-functions)

---

### Overview
Supervised learning creates a predictive model mapping input features $x$ to target variables $y$. For univariate linear regression, the model is characterized by the parameters $w$, the weight (slope), and $b$ bias (intercept):

$$f_{w,b}(x) = wx + b$$

Linear regression optimizes the parameters $w$ and $b$ to minimize a cost function $J(w, b)$. To achieve this we use the gradient descent algorithm. A visualization of how this algorithm works can be seen in the following graph:

<img width="792" height="773" alt="image" src="https://github.com/user-attachments/assets/0a8147f8-3010-415b-a849-9707c10919dc" />

This plot can be generated using `model.plots.plot_model_and_training_set()`, accessible via the `.plots` attribute of the `linear_regression` class.

### Implementation notes

A modular architecture has been chosen in which a `linear_regression` class performs the gradient descent algorithm, and has associated printing and plotting classes composed.

Vectorization with NumPy, rather than standard Python loops, also ensures a much faster model, this has been validated since, with loops, a model took ~6 seconds to converge, while with the same training set it takes 1 second with NumPy arrays.

### The cost function

The cost error function is given by:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

where:

* $m$: Number of training examples
* $x^{(i)}$: Feature value of the $i$-th sample
* $y^{(i)}$: Target label of the $i$-th sample
* $\hat{y}^{(i)}$: Estimated prediction

This expression quantifies the error by computing the mean squared error: accumulating the squared difference between each estimated value and its target, then averaging over all examples.
Another way of understanding this cost error function is by seeing it as the average of the area of the squares whose side is the difference between the estimated value and the target variable. This can be visualized as follows: 

<img width="862" height="855" alt="image" src="https://github.com/user-attachments/assets/b7ee5e44-65a4-458b-b0ed-6662001e8aec" />

This plot can be generated using `model.plots.plot_error_squares()`, accessible via the `.plots` attribute of the `linear_regression` class. 

### Gradient descent

This linear regression algorithm is implemented using gradient descent, with the option to compute the partial derivatives of the cost function from the definition of the derivative:

$$
f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

which, for $w$ and $b$, gives:

$$
\frac{\partial J}{\partial w} =
\lim_{\epsilon \to 0}
\frac{J(w + \epsilon, b) - J(w, b)}{\epsilon}
$$

$$
\frac{\partial J}{\partial b} =
\lim_{\epsilon \to 0}
\frac{J(w, b + \epsilon) - J(w, b)}{\epsilon}
$$

or, using the partial derivatives with the other variable held constant:

$$
\frac{\partial J(w, b)}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)}
$$

$$
\frac{\partial J(w, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)
$$

#### Why gradient descent finds the global minimum

It should be noted that since the cost error function is a convex (U) function, it has only one minimum, therefore, there is no conflict with possible local minima that don't 
represent the absolute minimum parameters. The absence of local minima can be shown formally using the second-derivative test for a multivariable function: the Hessian of $J(w, b)$.

Hessian matrix:

$$H = \nabla^2 J(w, b) = \left[\begin{array}{cc} \dfrac{\partial^2 J}{\partial w^2} & \dfrac{\partial^2 J}{\partial w \, \partial b} \\\\ \dfrac{\partial^2 J}{\partial b \, \partial w} & \dfrac{\partial^2 J}{\partial b^2} \end{array}\right] = \left[\begin{array}{cc} \dfrac{1}{m} \sum_{i=1}^{m} \left(x^{(i)}\right)^2 & \dfrac{1}{m} \sum_{i=1}^{m} x^{(i)} \\\\ \dfrac{1}{m} \sum_{i=1}^{m} x^{(i)} & 1 \end{array}\right]$$

Hessian determinant:

$$\det(H) = \frac{1}{m} \sum_{i=1}^{m} \left(x^{(i)}\right)^2 - \left( \frac{1}{m} \sum_{i=1}^{m} x^{(i)} \right)^2 = \overline{x^2} - (\bar{x})^2 = \mathrm{Var}(x)$$

By Jensen's inequality, the mean of the squared inputs is always at least as large as the square of the mean, so $\det(H) \geq 0$, with equality only when every $x^{(i)}$ is identical (zero variance in the inputs). The first-order (top-left) minor of $H$, $\frac{1}{m}\sum (x^{(i)})^2$, is positive as long as not every input is exactly 0. So, for any real training set with at least some variation in $x$, both leading principal minors of $H$ are positive, which by Sylvester's criterion means $H$ is positive definite 
Therefore, the critical point is a minimum, and the only one that the function has.

### Step implementation

At each iteration, the parameters $w$ and $b$ step in the opposite direction of the gradient, scaled by the learning rate $\alpha$:

$$
w := w - \alpha \frac{\partial J(w, b)}{\partial w}
$$

$$
b := b - \alpha \frac{\partial J(w, b)}{\partial b}
$$

#### Physical meaning of the sign and learning rate

The gradient of the cost function provides the direction of steepest *increase* at a given point. That is why, in the update rule for $w$ and $b$ at each iteration, we use a negative sign; the model moves in the direction of steepest *descent* instead.

This can also be seen as a restoring sign: when the derivative is negative (i.e., we are to the left of the minimum, $w_{\text{current}} < w_{\text{optimal}}$), subtracting it gives a positive increment, moving us right toward the minimum; when we are to the right of the minimum ($w_{\text{current}} > w_{\text{optimal}}$), the derivative is positive, and subtracting it decreases $w$, moving us left toward the minimum.

The learning rate $\alpha$ scales the size of each update step. It needs to be chosen carefully: too large a value can cause the model to diverge or oscillate around the minimum instead of settling into it, while too small a value makes training extremely slow.


### Usage

A training set is generated from fixed parameters. Thereby, the model can be compared against the baseline $w$ and $b$:

```python
rng = np.random.default_rng(30)

w_true = 3.5
b_true = -10.0

input_features = rng.standard_normal(100) * 10
target_values = w_true * input_features + b_true
```

The `linear_regression` class supports the two ways of computing the gradient, a numerical approximation and the derived form of the cost error function. Then, both can be run on the same data and compared:

```python
numerical_linear_regression_model = linear_regression(input_features, target_values)
expression_linear_regression_model = linear_regression(input_features, target_values)

expression_linear_regression_model.gradient_descent(numerically=False)
numerical_linear_regression_model.gradient_descent(numerically=True)

numerical_linear_regression_model.plots.plot_model_and_training_set()
```

### Results

Executing gradient descent on the dataset with ($w_{\text{true}} = 3.5$, $b_{\text{true}} = -10.0$) confirms that the fitted line maps the underlying linear relationship:

<img width="1008" height="547" alt="image" src="https://github.com/user-attachments/assets/8b7d6b15-23e8-4286-98a7-15b50e683a23" />

This plot can be generated using `model.plots.plot_model_and_training_set()`, accessible via the `.plots` attribute of the `linear_regression` class. The library also includes the printing method .all()
which prints all relevant information of the `linear_regression` class, accessible via the `.print` attribute:

```text
w: 3.4999999913110837
b: -9.999999002157347
number of iterations for convergence: 161385
time for convergence: 3.19501870800741 s
final cost J(w,b): 4.984231649804137e-13
```

Comparing the fitted parameters against the known baseline confirms a proper model implementation:

| Parameter | Model | Real value | Absolute error ($\Delta$) | Relative error |
| :--- | :---: | :---: | :---: | :---: |
| Weight ($w$) | 3.49999999 | 3.5 | $8.69 \times 10^{-9}$ | $2.48 \times 10^{-7}\%$ |
| Bias ($b$) | -9.99999900 | -10.0 | $9.98 \times 10^{-7}$ | $9.98 \times 10^{-6}\%$ |

Having an error of an order of $10^{-13}$ confirms that the model converges to the theoretical point.

#### Comparison of the numerical derivative against derivative expression

To confirm the partial derivatives are implemented correctly, the model was also run with `gradient_descent(numerically=True)`, which estimates the gradient via finite differences instead. 
Both approaches were run on the same data set and converged after the same number of iterations, to nearly identical parameters. 

| Gradient method | Weight ($w$) | Bias ($b$) | Iterations | Time (s) | Cost $J(w,b)$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Numerical derivative | 3.4999999913110837 | -9.999999002157347 | 161,385 | 3.1815 | $4.9842 \times 10^{-13}$ |
| Derivative expression | 3.4999999962991883 | -9.999998998905552 | 161,385 | 2.9661 | $5.0045 \times 10^{-13}$ |
| Relative difference | $1.43 \times 10^{-7}\%$ | $3.25 \times 10^{-8}\%$ | 0% | -6.77% | 0.4054% |

The two methods agree which confirms the gradient expressions match the numerical estimate. 

#### Comparison of the zero gradient convergence method against zero difference between the cost error functions

Convergence in technical terms is achieved when the norm of the gradient is close to zero. However, numerically, an exact zero is generally not reached due to finite numerical precision, so convergence is typically detected when the norm of the gradient falls below a chosen tolerance.
Furthermore, the gradient enters a flat asymptotic region near the minimum where, scaled by the learning rate $\alpha$, it barely alters the values of $w$ and $b$, causing unnecessary iterations.

Therefore, another approach is taking the difference between the cost of the previous iteration and the current one detecting the plateau condition. Nonetheless, this solution is not valid for all error functions, 
since they may have a plateau section without being a minimum. Nevertheless, because the mean squared error has a strictly convex shape, we can assert that it will represent the real minimum. The `linear_regression` class also allows to define the convergence method, with 
'zero_gradient' or 'zero_difference', the comparison between both is as follows:

```python
zero_gradient_linear_regression_model = linear_regression(input_features, target_values)
zero_difference_linear_regression_model = linear_regression(input_features, target_values)

zero_gradient_linear_regression_model.gradient_descent(numerically=False, convergence_method='zero_gradient')
zero_difference_linear_regression_model.gradient_descent(numerically=False, convergence_method='zero_difference')
```

| Convergence criterion | Weight ($w$) | Bias ($b$) | Iterations | Time (s) | Cost $J(w,b)$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Zero gradient | 3.4999999962991883 | -9.999998998905552 | 161,385 | 2.8486 | $5.0045 \times 10^{-13}$ |
| Zero difference | 3.4999629895405104 | -9.989988422639264 | 69,166 | 1.0141 | $5.0052 \times 10^{-5}$ |
| Relative difference | 0.001057% | 0.100206% | -57.14% | -64.40% | 99.999% ($\Delta J = 5.01 \times 10^{-5}$) |

Both methods arrive at essentially the same minimum ($w$ and $b$ match within 0.1% relative difference). The 100% relative difference in $J$ is a result of comparing $10^{-13}$ against $10^{-5}$, 
with an absolute difference of just $5 \times 10^{-5}$. Therefore, using the difference as terminating condition effectively detects the plateau and eliminates more than 92,000 redundant steps.
