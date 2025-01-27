# determines portion of a note that is sound vs rest using a beta distribution

class BetaChannel(nn.Module):
    def __init__(self, modes: List[int]):
        super(BetaChannel, self).__init__()
        self.modes = modes
        # compute alpha and betas for each mode
        