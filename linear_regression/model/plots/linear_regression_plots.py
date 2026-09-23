import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


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
            self.linear_regression.output_variables,
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

    def plot_model_and_training_set(self) -> None:
        """
        Plots the training set and the model together
        """

        figure = plt.figure(figsize=(12, 6))
        ax = figure.add_subplot(111)
        x_data = self.linear_regression.input_variables
        y_data = self.linear_regression.output_variables
        w = self.linear_regression.w
        b = self.linear_regression.b
        ax.scatter(
            x_data,
            y_data,
            color="navy",
            marker="1",
            label="Training set",
        )
        x_array = np.linspace(
            min(self.linear_regression.input_variables),
            max(self.linear_regression.input_variables),
            100,
        )
        estimated_y_array = (w * x_array + b)
        ax.plot(x_array, estimated_y_array, color="darkorange", label="Model")

        # label
        ax.set_title("Model vs training set")
        ax.set_xlabel("X")
        ax.set_ylabel("estimated Y")
        ax.legend()

    def plot_parameters_path(self, elev: float | int | None = None,
        azim: float | int | None = None) -> None:
        """
        Plots in 3D for each w and b calculated by the model, the corresponding J
        Parameters
        ---------
        elev : float, int, optional
            The elevation angle in degrees rotates the camera above the plane
            pierced by the vertical axis, with a positive angle corresponding
            to a location above that plane. If None, the default view is used.
            Default is None.
        azim : float, int, optional
            The azimuthal angle in degrees rotates the camera about the vertical
            axis. If None, the default view is used. Default is None.
        """
        figure = plt.figure(figsize=(16, 9))
        ax = figure.add_subplot(111, projection="3d")
        ax.plot(
            self.linear_regression.w_list,
            self.linear_regression.b_list,
            self.linear_regression.cost_list,
            color="mediumpurple",
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(None, zoom=1.0)

        ax.set_title("Parameters path", fontsize=15)
        ax.set_xlabel("w", fontsize=11, labelpad=8)
        ax.set_ylabel("b", fontsize=11, labelpad=8)
        ax.set_zlabel("") 
        ax.text2D(
            -0.05, 0.5, "J(w, b)",
            fontsize=11, rotation=90, va="center", ha="center",
            transform=ax.transAxes,
        )
        figure.subplots_adjust(top=0.92)

    def plot_error_squares(self) -> None:
        figure = plt.figure(figsize=(10, 10))
        ax = figure.add_subplot(111)

        x_data = self.linear_regression.input_variables
        y_data = self.linear_regression.output_variables
        w = self.linear_regression.w
        b = self.linear_regression.b

        x_array = np.linspace(np.min(x_data), np.max(x_data), 100)
        estimated_y_array = w * x_array + b
        ax.plot(
            x_array,
            estimated_y_array,
            color="darkgreen",
            linewidth=2,
            label="Regression Line",
        )
        ax.scatter(
            x_data,
            y_data,
            color="darkolivegreen",
            marker="1",
            label="Training set",
        )

        # Squares
        for index, (x_input, y_input) in enumerate(zip(x_data, y_data)):
            estimated_y = w * x_input + b
            diff = y_input - estimated_y  # Signed difference
            square_side = abs(diff)

            y_corner = min(estimated_y, y_input)
            x_corner = x_input  # Square extends to the right by default
            if index == 1: 
                label = 'Error Squares'
            else: 
                label = None

            rect = Rectangle(
                (x_corner, y_corner),
                width=square_side,
                height=square_side,
                facecolor="mediumaquamarine",
                edgecolor="seagreen",
                alpha=0.25,
                label = label
            )
            ax.add_patch(rect)
        
        ax.set_xlabel("X")
        ax.set_ylabel("estimated Y")
        ax.set_title("Least Squares")
        ax.legend(loc='upper left')
        plt.show()
                







        