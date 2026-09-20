import numpy as np

from model.plots.linear_regression_plots import plots_linear_regression
from model.prints.linear_regression_prints import prints_linear_regression


class linear_regression:

    def __init__(
        self,
        input_variables: np.ndarray,
        output_variables: np.ndarray,
        alpha: float | int = 0.0001,
        w_init: float | int = 0,
        b_init: float | int = 0,
    ) -> None:
        """
        Attributes:
        linear_regression.input_variables: np.ndarray
            Set of input features with which the model has been trained
        linear_regression.output_variables: np.ndarray
            Set of target values with which the model has been trained
        linear_regression.w_initial: float, int
            Initial w.
        linear_regression.b_initial: float, int
            Initial b.
        linear_regression.alpha: float, int
            Step for the gradient descent.
        linear_regression.w_list: list
            Set of the w values, w path.
        linear_regression.b_list: list
            Set of the b values, b path.
        linear_regression.cost_list: list
            Set of the cost error, J path.
        linear_regression.w: float 
            W for convergence
        linear_regression.b: float 
            B for convergence
        linear_regression.j: float 
            Cost error function for `linear_regression.w` and `linear_regression.b`

        Parameters
        ----------
        input_variables: np.ndarray
            Set of input features with which the model will be trained
        output_variables: np.ndarray
            Set of target features with which the model will be trained
        w_initial: float, int, optional
            initial w, slope of the model with which the algorithm will start.
            Default is 0.
        b_initial: float, int, optional
            initial b, abcisa of the model. Default is 0.
        alpha: float, int, optional
            scalar value to adjust the step for the gradient descent. Default
            is 0.0001.
        """

        self.input_variables = input_variables
        self.output_variables = output_variables
        self.alpha = alpha
        self.w_init = w_init
        self.b_init = b_init
        self.w = None
        self.b = None
        self.j = None
        self.plots = plots_linear_regression(self)
        self.prints = prints_linear_regression(self)
        self.b_list = []
        self.w_list = []
        self.cost_list = []

    def gradient_descent(self, numerically: bool = False) -> None:
        """This function creates a gradient descent algorithm based on the
        input parameter, alpha, initial w and b and the training input variables/features
        x and output/target variables y

        Parameters
        ----------
        numerically: bool, optional 
            If numerically, the partial derivative will be calculated using definition 
            of derivative. Otherwise, with the derivative of the cost error function with
            respect to the corresponding parameter, it uses a 'batch' approach, meaning each
            step of the gradient uses all training examples. Default is False. 
        """

        def compute_total_cost(w: float | int, b: float | int) -> float:
            """
            Computes the cost function for linear regression.

            Parameters
            ----------
            w: float, int
                Model parameter to calculate the cost error.
            b: float, int
                Model parameter to calculate the cost error.

            Returns
            -------
            total_error: float, int
                The cost of using w,b as the parameters for linear regression
                    to fit the data points in x and y
            """
            error_array = (w * self.input_variables + b) - self.output_variables
            total_error = float(np.mean(error_array ** 2) / 2.0)
            return total_error
        

        def compute_dj(
            w: float | int,
            b: float | int,
            prev_total_cost: float | int,
            numerically: bool = False,
        ) -> tuple:
            """
            Calculates the derivatives of the cost error function. 

            Parameters
            ----------
            w: float, int
                Previous w.
            b: float, int
                Value of the parameter b to calculate the derivative, it is also the previous b.
            prev_total_cost: float, int, optional 
                Total cost error for w_prev and b_prev. Necessary when numerically is True. 
            numerically: bool
                If numerically, the partial derivative will be calculated using definition 
                of derivative. Otherwise, with the derivative of the cost error function with
                respect to w. Default is False.

            Returns
            -------
            dj_dw, dj_db: tuple
                Derivative of the cost error function with respect to w when b is constant, and 
                with respect to b when w is constant for the w and b given as parameter. 
            """

            if numerically is True:
                dj_dw = (
                    compute_total_cost(w + 0.00000001, b)
                    - prev_total_cost
                ) / 0.00000001
                dj_db = (
                    compute_total_cost(w, b + 0.00000001)
                    - prev_total_cost
                ) / 0.00000001
            else:
                error_array = (w * self.input_variables + b) - self.output_variables
                dj_db = float(np.mean(error_array)) 
                dj_dw = float(np.mean(error_array * self.input_variables))

            return dj_dw, dj_db



        prev_total_cost = compute_total_cost(self.w_init, self.b_init)
        total_cost = np.inf
        diff_total_cost = abs(total_cost - prev_total_cost)
        w_prev = self.w_init
        b_prev = self.b_init
        self.b_list.append(b_prev)
        self.w_list.append(w_prev)
        self.cost_list.append(prev_total_cost)

        while diff_total_cost >= 1e-6:
            dj_dw, dj_db = compute_dj(w_prev, b_prev, prev_total_cost, numerically)

            w = w_prev - self.alpha * dj_dw
            b = b_prev - self.alpha * dj_db
            total_cost = compute_total_cost(w, b)

            self.b_list.append(b)
            self.w_list.append(w)
            self.cost_list.append(total_cost)

            diff_total_cost = abs(total_cost - prev_total_cost)

            w_prev = w
            b_prev = b
            prev_total_cost = total_cost

        self.w = self.w_list[-1]
        self.b = self.b_list[-1]
        self.j = self.cost_list[-1]
