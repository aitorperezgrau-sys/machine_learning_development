class prints_linear_regression:
    def __init__(self, linear_regression):
        self.linear_regression = linear_regression

    def parameter_info(self) -> None:
        """
        Prints the important information regarding the parameters
        """
        print(f'w: {self.linear_regression.w}')
        print(f'b: {self.linear_regression.b}')
        print(f'number of parameters before convergence: {len(self.linear_regression.w_list)}')

    def error_info(self) -> None: 
        """
        Prints the important information regarding the error. 
        """
        print(f'Final mean squared error: {self.linear_regression.j}')

    def all(self) -> None:
        """
        Prints all the relevant information of the model. 
        """
        self.parameter_info()
        self.error_info()