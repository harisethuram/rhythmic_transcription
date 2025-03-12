from scipy.optimize import minimize, dual_annealing


class LinearQuantizer: # TODO: need to finish this, basically replace the piece_wise_linear_fit logic with this
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        
    def fit(x, loss_fn, gamma=0, initial_alpha=3.0, upper_bound=20):
        """ Finds the optimal alpha that minimizes the deviation from integer values. 
        x: np array of shape (n)
        loss_fn: function that takes in alpha and x and returns a scalar
        initial_alpha: initial guess for alpha
        upper_bound: upper bound for alpha
        gamma: alpha regularization term
        """

        initial_alpha = np.array([3.0])
        bounds = [(1e-6, 99)]
        # gamma = 0.01

        result = dual_annealing(
            loss_fn,
            bounds=bounds,
            args=(x,gamma)
        )
        final_loss = loss_fn(result.x, x, gamma=0)
        # print(onset_lengths, final_loss, result.x)

        return result.x[0], final_loss