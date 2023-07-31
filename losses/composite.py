class CompositeLoss:
    def __init__(self, losses, return_components=False):
        self.__losses = losses
        self.__return_components = return_components
    
    def __len__(self):
        return len(self.__losses)

    def __call__(self, original, distorted):
        total_loss = 0.0
        components = dict()
        for (loss_fn, delta) in self.__losses:
            loss = loss_fn(original, distorted)
            total_loss += loss * delta
            components[type(loss_fn).__name__] = loss

        if self.__return_components:
            return total_loss, components
        else:
            return total_loss


