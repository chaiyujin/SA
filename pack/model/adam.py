import numpy as np


class SingleAdam():
    def __init__(self, learning_rate=1e-4, beta1=0.9, beta2=0.999,
                 epsilon=1e-8):
        self.t = 0
        self.m = 0
        self.v = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_t = 1
        self.beta2_t = 1
        self.alpha = learning_rate
        self.eps = epsilon

    def apply_gradient(self, var, grad):
        self.t += 1
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1_t)
        v_hat = self.v / (1 - self.beta2_t)
        new_var = var - self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)
        return new_var


class Adam():
    def __init__(self, learning_rate=1e-4, beta1=0.9, beta2=0.999,
                 epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = learning_rate
        self.eps = epsilon
        self.adams = {}

    def apply_gradients(self, var, grad, indexes):
        assert(var.shape[0] == grad.shape[0])
        assert(var.shape[0] == indexes.shape[0])
        new_var = np.zeros_like(var)
        for i in range(indexes.shape[0]):
            idx = indexes[i]
            if not (idx in self.adams):
                self.adams[idx] = SingleAdam(self.alpha, self.beta1,
                                             self.beta2, self.eps)
            new_var[i] = self.adams[idx].apply_gradient(
                var[i], grad[i]
            )
        return new_var


if __name__ == '__main__':
    def f(x):
        return x * x + 2 * x + 1

    def grad(x):
        return 2 * x + 2

    adam = Adam(1e-4)

    x = 10
    epoch = 0
    while True:
        y = f(x)
        x = adam.apply_gradient(x, grad(x))
        if epoch % 1000 == 0:
            print(x, ':', y)
        epoch += 1
