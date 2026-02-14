from math import sqrt
import numpy as np

class Optimizer:

    def __init__(self):
        self.n = 2
        self.m = 0.5
        self.gamma = 0.45
        self.beta = 2.2
        self.epsilon = 0.1
        self.x = [[1.5, 0.2], [], []]
        self.F = None

    def evaluate(self, x):
        return 7 * x[0] ** 2 + 2 * x[0] * x[1] + 5 * x[1] ** 2 + x[0] - 10 * x[1]

    def optimize(self):
        # расчёт двух вершин начального симплекса
        delta1 = (sqrt(self.n + 1) - 1) / (self.n * sqrt(2)) * self.m
        delta2 = (sqrt(self.n + 1) + self.n - 1) / (self.n * sqrt(2)) * self.m
        print(f"Приращения\ndelta1 = {round(delta1, 3)}\ndelta2 = {round(delta2, 3)}")
        self.x[1] = [self.x[0][0] + delta1, self.x[0][1] + delta2]
        self.x[2] = [self.x[0][0] + delta2, self.x[0][1] + delta1]
        print(f"Исходный базис\n{self.x[0][0]} {self.x[0][1]}\n"
              f"{round(self.x[1][0], 3)} {round(self.x[1][1], 3)}\n"
              f"{round(self.x[2][0], 3)} {round(self.x[2][1], 3)}\n")
        self.F = [self.evaluate(self.x[i]) for i in range(self.n + 1)] # оценка вершин

        iteration = 0
        while True:
            print(f"Итерация: {iteration}")
            k1, k2, k = np.argsort(self.F) # сортировка вершин
            # определение центра тяжести (без максимальной вершины)
            xc = [
                (self.x[k1][0] + self.x[k2][0]) / self.n,
                (self.x[k1][1] + self.x[k2][1]) / self.n
            ]
            print(f"Центр тяжести: {round(xc[0], 3)} {round(xc[1], 3)}")
            # отражение максимальной вершины
            x = [
                2 * xc[0] - self.x[k][0],
                2 * xc[1] - self.x[k][1]
            ]
            print(f"Отражённая вершина: {round(x[0], 3)} {round(x[1], 3)}")
            F = self.evaluate(x)

            squeeze_flag, reduction_flag = False, False

            if F < self.F[k]:
                self.x[k] = x
                self.F[k] = F
                if self.F[k] < self.F[k1]:
                    x = [
                        xc[0] + self.beta * (self.x[k][0] - xc[0]),
                        xc[1] + self.beta * (self.x[k][1] - xc[1])
                    ]
                    F = self.evaluate(x)
                    if F < self.F[k]:
                        self.x[k] = x
                        self.F[k] = F
                    else:
                        squeeze_flag = True
                else:
                    squeeze_flag = True
            else:
                squeeze_flag = True

            # сжатие
            if squeeze_flag:
                if self.F[k1] < F < self.F[k]:
                    x = [
                        xc[0] + self.gamma * (self.x[k][0] - xc[0]),
                        xc[1] + self.gamma * (self.x[k][1] - xc[1])
                    ]
                    F = self.evaluate(x)
                    if F < self.F[k]:
                        self.x[k] = x
                        self.F[k] = F
                    else:
                        reduction_flag = True
                else:
                    reduction_flag = True

            # редукция
            if reduction_flag:
                r = np.argmin(self.F)
                xr = self.x[r]
                for i in range(self.n + 1):
                    if i != r:
                        self.x[i] = [
                            xr[0] + 0.5 * (self.x[i][0] - xr[0]),
                            xr[1] + 0.5 * (self.x[i][1] - xr[1])
                        ]
            # центр тяжести всего симплекса
            xc = [
                (self.x[0][0] + self.x[1][0] + self.x[2][0]) / (self.n + 1),
                (self.x[0][1] + self.x[1][1] + self.x[2][1]) / (self.n + 1)
            ]
            Fc = self.evaluate(xc)

            print(f"Центр тяжести симплекса: {round(xc[0], 3)} {round(xc[1], 3)}"
                  f"\nЗначение функции в центре: {round(Fc, 3)}")

            iteration += 1

            print(f"Полученный базис\n{round(self.x[0][0], 3)} {round(self.x[0][1], 3)}\n"
                  f"{round(self.x[1][0], 3)} {round(self.x[1][1], 3)}\n"
                  f"{round(self.x[2][0], 3)} {round(self.x[2][1], 3)}\n")

            # проверка условия остановки
            sigma = sqrt(sum([(self.F[i] - Fc) ** 2 for i in range(self.n + 1)]) / (self.n + 1))
            if sigma < self.epsilon:
                x_min = np.argmin(self.F)
                print("Критерий остановки выполнен")
                print(f"Минимум в точке: {self.x[x_min][0]:.6f} {self.x[x_min][1]:.6f}")
                print(f"Значение функции: {self.F[x_min]:.6f}")
                print(f"Итераций выполнено: {iteration}")
                break


opt = Optimizer()
opt.optimize()
