import autograd.numpy as np
from autograd import grad, hessian

class Optimizer:
    def __init__(self):
        self.n = 2
        self.x = np.array([1.5, 0.2])
        self.epsilon = 0.1
        self.max_iter = 50
        self.history = [self.x.copy()]

    def evaluate(self, x):
        return 7 * x[0] ** 2 + 2 * x[0] * x[1] + 5 * x[1] ** 2 + x[0] - 10 * x[1]

    def print_history(self):
        print("Таблица векторов")
        for vec in self.history:
            print(round(vec[0], 3), round(vec[1], 3), round(self.evaluate(vec), 3))

    def optimize(self):
        g_func = grad(self.evaluate)
        H_func = hessian(self.evaluate)

        k = 0

        while k < self.max_iter:
            grad_vec = g_func(self.x)
            norm_grad = np.linalg.norm(grad_vec)

            print(f"Итерация {k}"
                  f"\nКоординаты базисного вектора: {round(self.x[0], 3)} {round(self.x[1], 3)}"
                  f"\nСкалярное значение базисного вектора: {round(self.evaluate(self.x), 3)}"
                  f"\nКоординаты вектора градиента: {round(grad_vec[0], 3)} {round(grad_vec[1], 3)}"
                  f"\nЗначение нормы вектора: {round(norm_grad, 3)}")

            if norm_grad <= self.epsilon:
                print("Минимум найден!\n")
                break

            H = H_func(self.x)
            h_k = np.dot(grad_vec, grad_vec) / np.dot(H @ grad_vec, grad_vec)

            self.x -= h_k * grad_vec

            print(f"Новый шаг: {h_k}"
                  f"\nКоординаты нового вектора: {round(self.x[0], 3)} {round(self.x[1], 3)}"
                  f"\nСкалярное значение нового вектора: {round(self.evaluate(self.x), 3)}\n")

            self.history.append(self.x.copy())
            k += 1

        self.print_history()

        print("\nИтоговая точка:", self.x)
        print("f(x) =", self.evaluate(self.x))


opt = Optimizer()
opt.optimize()