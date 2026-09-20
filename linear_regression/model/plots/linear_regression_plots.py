import matplotlib.pyplot as plt
import numpy as np


class plots_linear_regression:
    def __init__(self, linear_regression):
        self.linear_regression = linear_regression

    def plot_training_set(self) -> None:
        """
        Plots the training set of the model
        """
        figure = plt.figure(figsize=(12, 6))
        ax = figure.add_subplot(111)
        ax.scatter(
            self.linear_regression.input_variables,
            self.linear_regression.ouput_variables,
            color="navy",
            marker="1",
        )

        # label
        ax.set_title("Traning set")
        ax.set_xlabel("Input Variables")
        ax.set_ylabel("Output Variables")

    def plot_model(self) -> None:
        """
        Plots the training set of the model
        """
        figure = plt.figure(figsize=(12, 6))
        ax = figure.add_subplot(111)
        x_array = np.linspace(
            min(self.linear_regression.input_variables),
            max(self.linear_regression.input_variables),
            100,
        )
        estimated_y_array = (
            self.linear_regression.w * x_array + self.linear_regression.b
        )
        ax.plot(x_array, estimated_y_array, color="darkorange")

        # label
        ax.set_title("Model")
        ax.set_xlabel("X")
        ax.set_ylabel("estimated Y")

    def plot_model_and_traning_set(self) -> None:
        """
        Plots the training set and the model together
        """

        figure = plt.figure(figsize=(12, 6))
        ax = figure.add_subplot(111)
        ax.scatter(
            self.linear_regression.input_variables,
            self.linear_regression.ouput_variables,
            color="navy",
            marker="1",
            label="Training set",
        )

        x_array = np.linspace(
            min(self.linear_regression.input_variables),
            max(self.linear_regression.input_variables),
            100,
        )
        estimated_y_array = (
            self.linear_regression.w * x_array + self.linear_regression.b
        )
        ax.plot(x_array, estimated_y_array, color="darkorange", label="Model")

        # label
        ax.set_title("Model vs training set")
        ax.set_xlabel("X")
        ax.set_ylabel("estimated Y")
        ax.legend()
    
    def plot_parameters_path(self) -> None:
        """
        Plots in 3D for each w and b calculated by the model, the corresponding J
        """

        figure = plt.figure(figsize=(20,30))
        ax = figure.add_subplot(111, projection='3d')
        ax.plot(self.linear_regression.w_list, self.linear_regression.b_list, self.linear_regression.cost_list, color = 'mediumpurple')
        ax.view_init(elev = 0, azim = 30)


        # label
        ax.set_title('Parameters path')
        ax.set_xlabel("b", fontsize=20, labelpad=15)
        ax.set_ylabel("w", fontsize=20, labelpad=15)
        ax.set_zlabel("J(w,b)", fontsize=20, labelpad=10)
        ax.tick_params(axis="both", which="major", labelsize=12)
