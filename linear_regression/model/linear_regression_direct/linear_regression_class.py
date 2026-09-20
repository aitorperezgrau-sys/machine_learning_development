import numpy as np

from model.plots.linear_regression_plots import plots_linear_regression


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
        self.plots = plots_linear_regression(self)

    def gradient_descent(self) -> None:
        """This function creates a gradient descent algorithm based on the
        input parameter, alpha, initial w and b and the training input variables/features
        x and ouput/target variables y
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

        def compute_cost_error_partial_w(w_prev, constant_b, prev_total_cost) -> float:
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

            Returns
            -------
            cost_error_partial_w: float, int
                Derivative of the cost error function with respect to w when b is constant.
            """

            cost_error_partial_w = (
                compute_total_cost(w_prev + 0.00000001, constant_b) - prev_total_cost
            ) / 0.00000001
            return cost_error_partial_w

        def compute_cost_error_partial_b(constant_w, b_prev, prev_total_cost) -> float:
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

            Returns
            -------
            cost_error_partial_v: float, int
                Derivative of the cost error function with respect to w when b is constant.
            """
            cost_error_partial_b = (
                compute_total_cost(constant_w, b_prev + 0.00000001) - prev_total_cost
            ) / 0.00000001
            return cost_error_partial_b

        prev_total_cost = compute_total_cost(self.w_init, self.b_init)
        total_cost = np.inf
        diff_total_cost = abs(total_cost - prev_total_cost)
        w_prev = self.w_init
        b_prev = self.b_init
        while diff_total_cost >= 1e-6:
            w = w_prev - self.alpha * compute_cost_error_partial_w(
                w_prev, b_prev, prev_total_cost
            )
            b = b_prev - self.alpha * compute_cost_error_partial_b(
                w_prev, b_prev, prev_total_cost
            )
            total_cost = compute_total_cost(w, b)
            diff_total_cost = abs(total_cost - prev_total_cost)
            w_prev = w
            b_prev = b
            prev_total_cost = total_cost

        self.w = w_prev
        self.b = b_prev
