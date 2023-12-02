import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from ipywidgets import interact, RadioButtons, IntSlider, FloatSlider, Dropdown, BoundedFloatText

def sigmoid(x):
    """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)"""
    return sigmoid(x) * (1 - sigmoid(x))



class Neuron:

    def __init__(self, weights, activation_function=sigmoid, activation_function_derivative=sigmoid_prime):
        """
        weights - вертикальный вектор весов нейрона формы (m, 1), weights[0][0] - смещение
        activation_function - активационная функция нейрона, сигмоидальная функция по умолчанию
        activation_function_derivative - производная активационной функции нейрона
        """

        assert weights.shape[1] == 1, "Incorrect weight shape"

        self.w = weights
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def forward_pass(self, single_input):
        """
        активационная функция логистического нейрона
        single_input - вектор входов формы (m, 1),
        первый элемент вектора single_input - единица (если вы хотите учитывать смещение)
        """

        result = 0
        for i in range(self.w.size):
            result += float(self.w[i] * single_input[i])
        return self.activation_function(result)

    def summatory(self, input_matrix):
        """
        Вычисляет результат сумматорной функции для каждого примера из input_matrix.
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных.
        Возвращает вектор значений сумматорной функции размера (n, 1).
        """
        # Этот метод необходимо реализовать
        res_sum = input_matrix.dot(self.w)
        return res_sum

        pass

    def activation(self, summatory_activation):
        """
        Вычисляет для каждого примера результат активационной функции,
        получив на вход вектор значений сумматорной функций
        summatory_activation - вектор размера (n, 1),
        где summatory_activation[i] - значение суммматорной функции для i-го примера.
        Возвращает вектор размера (n, 1), содержащий в i-й строке
        значение активационной функции для i-го примера.
        """
        # Этот метод необходимо реализовать
        res_activation = sigmoid(np.array(summatory_activation))
        return res_activation

        pass

    def vectorized_forward_pass(self, input_matrix):
        """
        Векторизованная активационная функция логистического нейрона.
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных.
        Возвращает вертикальный вектор размера (n, 1) с выходными активациями нейрона
        (элементы вектора - float)
        """
        return self.activation(self.summatory(input_matrix))

    def SGD(self, X, y, batch_size, learning_rate=0.1, eps=1e-6, max_steps=200):
        """
        Внешний цикл алгоритма градиентного спуска.
        X - матрица входных активаций (n, m)
        y - вектор правильных ответов (n, 1)

        learning_rate - константа скорости обучения
        batch_size - размер батча, на основании которого
        рассчитывается градиент и совершается один шаг алгоритма

        eps - критерий остановки номер один: если разница между значением целевой функции
        до и после обновления весов меньше eps - алгоритм останавливается.
        Вторым вариантом была бы проверка размера градиента, а не изменение функции,
        что будет работать лучше - неочевидно. В заданиях используйте первый подход.

        max_steps - критерий остановки номер два: если количество обновлений весов
        достигло max_steps, то алгоритм останавливается

        Метод возвращает 1, если отработал первый критерий остановки (спуск сошёлся)
        и 0, если второй (спуск не достиг минимума за отведённое время).
        """
        # Этот метод необходимо реализовать
        for i in range(max_steps):
            batch_rows_random = np.random.choice(int(X.shape[0]), batch_size, replace=False)
            delta_fun = self.update_mini_batch(X[batch_rows_random, :], y[batch_rows_random, :], learning_rate, eps)
            if delta_fun == 1:
                return 1

        return 0
        pass

    def update_mini_batch(self, X, y, learning_rate, eps):
        """
        X - матрица размера (batch_size, m)
        y - вектор правильных ответов размера (batch_size, 1)
        learning_rate - константа скорости обучения
        eps - критерий остановки номер один: если разница между значением целевой функции
        до и после обновления весов меньше eps - алгоритм останавливается.

        Рассчитывает градиент (не забывайте использовать подготовленные заранее внешние функции)
        и обновляет веса нейрона. Если ошибка изменилась меньше, чем на eps - возвращаем 1,
        иначе возвращаем 0.
        """
        # Этот метод необходимо реализовать

        target_fun_w = J_quadratic(self, X, y)
        grad = compute_grad_analytically(self, X, y)
        delta_w = learning_rate*grad
        self.w = self.w - delta_w
        new_target_fun_w = J_quadratic(self, X, y)
        if abs(target_fun_w - new_target_fun_w) <= eps:
            return 1
        else:
            return 0

        pass

def J_quadratic(neuron, X, y):
    """
    Оценивает значение квадратичной целевой функции.
    Всё как в лекции, никаких хитростей.

    neuron - нейрон, у которого есть метод vectorized_forward_pass, предсказывающий значения на выборке X
    X - матрица входных активаций (n, m)
    y - вектор правильных ответов (n, 1)

    Возвращает значение J (число)
    """

    assert y.shape[1] == 1, 'Incorrect y shape' # проверка на размерность

    return 0.5 * np.mean((neuron.vectorized_forward_pass(X) - y) ** 2)


def J_quadratic_derivative(y, y_hat):
    """
    Вычисляет вектор частных производных целевой функции по каждому из предсказаний.
    y_hat - вертикальный вектор предсказаний,
    y - вертикальный вектор правильных ответов,

    В данном случае функция смехотворно простая, но если мы захотим поэкспериментировать
    с целевыми функциями - полезно вынести эти вычисления в отдельный этап.

    Возвращает вектор значений производной целевой функции для каждого примера отдельно.
    """

    assert y_hat.shape == y.shape and y_hat.shape[1] == 1, 'Incorrect shapes'

    return (y_hat - y) / len(y)


def compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative):
    """
    Аналитическая производная целевой функции
    neuron - объект класса Neuron
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений
    y - правильные ответы для примеров из матрицы X
    J_prime - функция, считающая производные целевой функции по ответам

    Возвращает вектор размера (m, 1)
    """
    # Вычисляем активации
    # z - вектор результатов сумматорной функции нейрона на разных примерах # (n,1)

    z = neuron.summatory(X) # (n,1)
    y_hat = neuron.activation(z) # (n,1) предсказанные значения

    # Вычисляем нужные нам частные производные
    dy_dyhat = J_prime(y, y_hat) # (n,1) производная целевой функции по предсказанным
    dyhat_dz = neuron.activation_function_derivative(z) # (n,1) производная активационной функции нейрона(сигмоиды)

    # осознайте эту строчку:
    dz_dw = X # (n,m) производная от значений сумматорной функции по весам

    # а главное, эту:
    grad = ((dy_dyhat * dyhat_dz).T).dot(dz_dw)

    # можно было написать в два этапа. Осознайте, почему получается одно и то же
    # grad_matrix = dy_dyhat * dyhat_dz * dz_dw
    # grad = np.sum(, axis=0)
    # Сделаем из горизонтального вектора вертикальный
    grad = grad.T
    return grad

def compute_grad_numerically(neuron, X, y, J=J_quadratic, eps=10e-2):
    """
    Численная производная целевой функции
    neuron - объект класса Neuron
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений
    y - правильные ответы для тестовой выборки X
    J - целевая функция, градиент которой мы хотим получить
    eps - размер $\delta w$ (малого изменения весов)
    """

    initial_cost = J(neuron, X, y) # определяем целевую функцию
    w_0 = neuron.w
    num_grad = np.zeros(w_0.shape)

    for i in range(len(w_0)):
        old_wi = neuron.w[i].copy()
        # Меняем вес
        neuron.w[i] += eps

        # Считаем новое значение целевой функции c дельта w[i] и вычисляем приближенное значение градиента
        num_grad[i] = (J(neuron, X, y) - initial_cost) / eps

        # Возвращаем вес обратно. Лучше так, чем -= eps, чтобы не накапливать ошибки округления
        neuron.w[i] = old_wi

    # проверим, что не испортили нейрону веса своими манипуляциями
    assert np.allclose(neuron.w, w_0), "МЫ ИСПОРТИЛИ НЕЙРОНУ ВЕСА" # сравниваем размеры двух векторов
    return num_grad

def print_grad_diff(eps):
    num_grad = compute_grad_numerically(neuron, X, y, J=J_quadratic, eps=float(eps))
    an_grad = compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative)
    print(np.linalg.norm(num_grad - an_grad))

def J_by_weights(weights, X, y, bias):
    """
    Посчитать значение целевой функции для нейрона с заданными весами.
    Только для визуализации
    """
    new_w = np.hstack((bias, weights)).reshape((3,1)) # смещение w1 и w2
    return J_quadratic(Neuron(new_w), X, y)

if __name__ == '__main__':
    # weights = np.array([1, 2, 3, 4, 5]).reshape(5, 1)
    # obj = Neuron(weights)
    # input_matrix = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]). reshape(3, 5)
    # rows = np.random.choice(3, 2)
    # print(input_matrix[rows, :])
    # res_sum = obj.summatory(input_matrix)
    # res_activation = obj.activation(res_sum)
    #
    # print(obj.update_mini_batch(input_matrix, res_activation, 2, 1e-3))


    data = np.loadtxt("data.csv", delimiter=",")

    X = data[:, :-1]
    y = data[:, -1]

    X = np.hstack((np.ones((len(y), 1)), X)) # добавили 1 в первый стобец для смещения b
    y = y.reshape((len(y), 1))  # Обратите внимание на эту очень противную и важную строчку
    print(X[:3])
    print(y[:3])

    w = np.random.random((X.shape[1], 1))
    neuron = Neuron(w, activation_function=sigmoid, activation_function_derivative=sigmoid_prime)

    # Посчитаем пример
    num_grad = compute_grad_numerically(neuron, X, y, J=J_quadratic)
    an_grad = compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative)

    print("Численный градиент: \n", num_grad)
    print("Аналитический градиент: \n", an_grad)
    print_grad_diff(1)
    print_grad_diff(0.01)




    max_b = 40
    min_b = -40
    max_w1 = 40
    min_w1 = -40
    max_w2 = 40
    min_w2 = -40

    g_bias = 0  # график номер 2 будет при первой генерации по умолчанию иметь то значение b, которое выставлено в первом
    X_corrupted = X.copy()
    y_corrupted = y.copy()


    # @interact(fixed_bias=FloatSlider(min=min_b, max=max_b, continuous_update=False),
    #           mixing=FloatSlider(min=0, max=1, continuous_update=False, value=0),
    #           shifting=FloatSlider(min=0, max=1, continuous_update=False, value=0)
    #             )


    def visualize_cost_function(fixed_bias, mixing, shifting):
        """
        Визуализируем поверхность целевой функции на (опционально) подпорченных данных и сами данные.
        Портим данные мы следующим образом: сдвигаем категории навстречу друг другу, на величину, равную shifting
        Кроме того, меняем классы некоторых случайно выбранных примеров на противоположнее.
        Доля таких примеров задаётся переменной mixing

        Нам нужно зафиксировать bias на определённом значении, чтобы мы могли что-нибудь визуализировать.
        Можно посмотреть, как bias влияет на форму целевой функции
        """
        xlim = (min_w1, max_w1)
        ylim = (min_w2, max_w2)
        xx = np.linspace(*xlim, num=101)
        yy = np.linspace(*ylim, num=101)
        xx, yy = np.meshgrid(xx, yy)
        points = np.stack([xx, yy], axis=2)

        # не будем портить исходные данные, будем портить их копию
        corrupted = data.copy()

        # инвертируем ответы для случайно выбранного поднабора данных
        mixed_subset = np.random.choice(range(len(corrupted)), int(mixing * len(corrupted)), replace=False)
        corrupted[mixed_subset, -1] = np.logical_not(corrupted[mixed_subset, -1])

        # сдвинем все груши (внизу справа) на shifting наверх и влево
        pears = corrupted[:, 2] == 1
        apples = np.logical_not(pears)
        corrupted[pears, 0] -= shifting
        corrupted[pears, 1] += shifting

        # вытащим наружу испорченные данные
        global X_corrupted, y_corrupted
        X_corrupted = np.hstack((np.ones((len(corrupted), 1)), corrupted[:, :-1]))
        y_corrupted = corrupted[:, -1].reshape((len(corrupted), 1))

        # посчитаем значения целевой функции на наших новых данных
        calculate_weights = partial(J_by_weights, X=X_corrupted, y=y_corrupted, bias=fixed_bias)
        J_values = np.apply_along_axis(calculate_weights, -1, points)

        fig = plt.figure(figsize=(16, 5))
        # сначала 3D-график целевой функции
        ax_1 = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax_1.plot_surface(xx, yy, J_values, alpha=0.3)
        ax_1.set_xlabel("$w_1$")
        ax_1.set_ylabel("$w_2$")
        ax_1.set_zlabel("$J(w_1, w_2)$")
        ax_1.set_title("$J(w_1, w_2)$ for fixed bias = ${}$".format(fixed_bias))
        # потом плоский поточечный график повреждённых данных
        ax_2 = fig.add_subplot(1, 2, 2)
        plt.scatter(corrupted[apples][:, 0], corrupted[apples][:, 1], color="red", alpha=0.7)
        plt.scatter(corrupted[pears][:, 0], corrupted[pears][:, 1], color="green", alpha=0.7)
        ax_2.set_xlabel("yellowness")
        ax_2.set_ylabel("symmetry")

        plt.show()


    # @interact(b=BoundedFloatText(value=str(g_bias), min=min_b, max=max_b, description="Enter $b$:"),
    #           w1=BoundedFloatText(value="0", min=min_w1, max=max_w1, description="Enter $w_1$:"),
    #           w2=BoundedFloatText(value="0", min=min_w2, max=max_w2, description="Enter $w_2$:"),
    #           learning_rate=Dropdown(options=["0.01", "0.05", "0.1", "0.5", "1", "5", "10"],
    #                                  value="0.01", description="Learning rate: ")
    #           )

    def learning_curve_for_starting_point(b, w1, w2, learning_rate=0.1):
        w = np.array([b, w1, w2]).reshape(X_corrupted.shape[1], 1) # (m, 1)
        learning_rate = float(learning_rate)
        neuron = Neuron(w, activation_function=sigmoid, activation_function_derivative=sigmoid_prime)

        story = [J_quadratic(neuron, X_corrupted, y_corrupted)]
        for _ in range(2000):
            neuron.SGD(X_corrupted, y_corrupted, 2, learning_rate=learning_rate, max_steps=4)
            story.append(J_quadratic(neuron, X_corrupted, y_corrupted))
        plt.plot(story)

        plt.title("Learning curve.\n Final $b={0:.3f}$, $w_1={1:.3f}, w_2={2:.3f}$".format(*neuron.w.ravel()))
        plt.ylabel("$J(w_1, w_2)$")
        plt.xlabel("Weight and bias update number")
        plt.show()
        return neuron



    learning_rate = 0.1
    b, w1, w2 = 0, 0, 1
    w = np.array([b, w1, w2]).reshape(X_corrupted.shape[1], 1)
    neuron = Neuron(w, activation_function=sigmoid, activation_function_derivative=sigmoid_prime)

    print(neuron.w)
    # neuron.update_mini_batch(X, y, 0.1, eps = 0.00001 )
    #neuron.SGD(X, y, 2, learning_rate=learning_rate, max_steps=4)
    learning_neuron = learning_curve_for_starting_point(b, w1, w2, 0.1)
    print(learning_neuron.w)
    w11 = 1.408992992910701458e-01
    w22 = 4.303189802502183081e-02
    test_data = np.array([1, w11, w22]).reshape(3, 1)
    res = learning_neuron.summatory(test_data.T)
    print(sigmoid(res))
    pred_y = []
    for example, y_cur in zip(X, y):
        pred = sigmoid(learning_neuron.summatory(example.T))
        if pred > 0.5:
         pred_y.append(int(1))
        else:
            pred_y.append(int(0))

    pred_y = np.array(pred_y)
################################ график ошибок
    import matplotlib.colors
    prices = np.c_[y[:22], pred_y[:22]]

    asort = np.argsort(prices, axis=1)
    sort = np.sort(prices, axis=1)

    cmap = matplotlib.colors.ListedColormap(["black", "red"])
    ind = np.arange(prices.shape[0])
    width = 0.5
    for i in range(prices.shape[1] - 1, -1, -1):
        plt.bar(ind, sort[:, i], width, color=cmap(asort[:, i]))

    plt.show()
