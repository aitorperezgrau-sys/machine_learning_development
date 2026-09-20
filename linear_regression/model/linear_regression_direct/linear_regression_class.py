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
        self.ouput_variables = output_variables
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
        x and ouput/target variables y

        Parameters
        ----------
        numerically: bool, optional 
            If numerically, the partial derivative will be calculated using definition 
            of derivative. Otherwise, with the derivative of the cost error function with
            respect to w. Default is False.
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
                total_cost: float, int
                    The cost of using w,b as the parameters for linear regression
                        to fit the data points in x and y
            """
            m = self.input_variables.shape[0]

            cost_sum = 0
            for i in range(m):
                f_wb = w * self.input_variables[i] + b
                cost = (f_wb - self.ouput_variables[i]) ** 2
                cost_sum = cost_sum + cost
            total_cost = (1 / (2 * m)) * cost_sum

            return total_cost

        def compute_cost_error_partial_w(
            w_prev: float | int,
            constant_b: float | int,
            prev_total_cost: float | int,
            numerically: bool = False,
        ) -> float:
            """
            Calculates the derivative of the cost error function with respect to w
            when b is constant with a value of `constant_b`.

            Parameters
            ----------
            w_prev: float, int
                Previous w.
            constant_b: float, int
                Value of the parameter b to calculate the derivative, it is also the previous b.
            prev_total_cost: float, int
                Total cost error for w_prev and b_prev
            numerically: bool
                If numerically, the partial derivative will be calculated using definition 
                of derivative. Otherwise, with the derivative of the cost error function with
                respect to w. Default is False.

            Returns
            -------
            cost_error_partial_w: float, int
                Derivative of the cost error function with respect to w when b is constant.
            """

            if numerically is True:
                cost_error_partial_w = (
                    compute_total_cost(w_prev + 0.00000001, constant_b)
                    - prev_total_cost
                ) / 0.00000001
            else:
                cost_error_partial_w = calculate_partial_w_with_function(
                    w_prev, constant_b
                )
            return cost_error_partial_w

        def compute_cost_error_partial_b(
            constant_w: float | int,
            b_prev: float | int,
            prev_total_cost: float | int,
            numerically: bool = False,
        ) -> float:
            """
            Calculates the derivative of the cost error function with respect to b
            when w is constant with a value of `constant_w`.

            Parameters
            ----------
            constant_w: float, int
                Value of the parameter w to calculate the derivative, it is also the previous w.
            b_prev: float, int
                Previous b.
            prev_total_cost: float, int
                Total cost error for w_prev and b_prev
            numerically: bool
                If numerically, the partial derivative will be calculated using definition 
                of derivative. Otherwise, with the derivative of the cost error function with
                respect to w. Default is False.
                
            Returns
            -------
            cost_error_partial_v: float, int
                Derivative of the cost error function with respect to w when b is constant.
            """
            if numerically is True:
                cost_error_partial_b = (
                    compute_total_cost(constant_w, b_prev + 0.00000001)
                    - prev_total_cost
                ) / 0.00000001
            else:
                cost_error_partial_b = calculate_partial_b_with_function(
                    constant_w, b_prev
                )
            return cost_error_partial_b

        def calculate_partial_b_with_function(
            constant_w: float | int, b_prev: float | int
        ) -> float:
            """
            Calculates the partial derivative of the cost error function with respect to b
            based on the function expression.

            Parameters
            ----------
            constant_w: float, int
                Value of the parameter w to calculate the derivative, it is also the previous w.
            b_prev: float, int
                Previous b.

            Returns
            --------
            cost_error_partial_b: float, int
                Partial derivative of the cost error funciton when w is `constant_w`.
            """
            m = self.input_variables.shape[0]

            cost_sum_partial = 0
            for i in range(m):
                f_wb = constant_w * self.input_variables[i] + b_prev
                cost = (f_wb - self.ouput_variables[i]) 
                cost_sum_partial = cost_sum_partial + cost
            cost_error_partial_b = (1 / m) * cost_sum_partial

            return cost_error_partial_b

        def calculate_partial_w_with_function(
            w_prev: float | int, constant_b: float | int
        ) -> float:
            """
            Calculates the partial derivative of the cost error function with respect to w
            based on the function expression.

            Parameters
            ----------
            w_prev: float, int
                Previous w.
            constant_b: float, int
                Value of the parameter b to calculate the derivative, it is also the previous b.

            Returns
            --------
            cost_error_partial_w: float, int
                Partial derivative of the cost error funciton when b is `constant_b`.
            """
            m = self.input_variables.shape[0]

            cost_sum_partial = 0
            for i in range(m):
                f_wb = w_prev * self.input_variables[i] + constant_b
                cost = (f_wb - self.ouput_variables[i]) * self.input_variables[i]
                cost_sum_partial = cost_sum_partial + cost
            cost_error_partial_w = (1 / m) * cost_sum_partial

            return cost_error_partial_w

        prev_total_cost = compute_total_cost(self.w_init, self.b_init)
        total_cost = np.inf
        diff_total_cost = abs(total_cost - prev_total_cost)
        w_prev = self.w_init
        b_prev = self.b_init
        self.b_list.append(b_prev)
        self.w_list.append(w_prev)
        self.cost_list.append(prev_total_cost)

        while diff_total_cost >= 1e-6:

            w = w_prev - self.alpha * compute_cost_error_partial_w(
                w_prev, b_prev, prev_total_cost, numerically
            )
            b = b_prev - self.alpha * compute_cost_error_partial_b(
                w_prev, b_prev, prev_total_cost, numerically
            )
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
