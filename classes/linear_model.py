import numpy as np


__all__ = ["LinearRegression", "LogisticRegression"]


class LinearRegression:
    """Класс линейной регрессии."""

    def __init__(self, learning_rate=0.005, epochs=1, report_interval=1):
        self.learning_rate = learning_rate  # шаг обучения
        self.epochs = epochs  # кол-во шагов градиентного спуска
        self.report_interval = report_interval  # через какой интервал считать характеристики

        self.num_obj = None  # кол-во объектов в обучающем наборе
        self.num_feat = None  # кол-во признаков
        self.weights = None  # веса
        self.bias = None  # свободный член
        # Списки ошибок.
        self.losses_train = None
        self.losses_test = None

    @staticmethod
    def __mean_squared_error(y_true, y_pred):
        """Среднеквадратичная ошибка."""
        return np.sum((y_true - y_pred) ** 2, axis=0) / len(y_true)

    def fit(self, X, y, eval_set=None):
        """Метод обучения модели с использованием цикла."""

        # Количество объектов и признаков.
        self.num_obj, self.num_feat = X.shape
        # Инициализация весов (вектор 1 x num_feat) и свободного члена по закону нормального распределения.
        self.weights = np.random.randn(self.num_feat, 1) * 0.001
        self.bias = np.random.randn() * 0.001
        if eval_set is not None:
            self.losses_train = []
            self.losses_test = []

        # Эпохи - кол-во проходов по всему набору данных или кол-во шагов по антиградиенту.
        for epoch in range(self.epochs):
            # Инициализация градиента.
            grad_w = np.zeros((self.num_feat, 1))
            grad_b = 0

            # Цикл по всем объектам.
            for i in range(self.num_obj):
                # Линейная комбинация признаков.
                # Матричное произведение Xi (1 x num_feat) на w (num_feat x 1) плюс bias.
                f = (X[i].reshape(1, self.num_feat).dot(self.weights) + self.bias)[0][0]  # f - скаляр
                # Расчёт градиента.
                grad_w += (f - y[i]) * X[i].reshape(self.num_feat, 1)
                grad_b += (f - y[i])

            # Усреднение градиента.
            grad_w *= 2 / self.num_obj
            grad_b *= 2 / self.num_obj

            # Корректировка весов и свободного члена. Шаг в сторону антиградиента.
            self.weights = self.weights - self.learning_rate * grad_w
            self.bias = self.bias - self.learning_rate * grad_b

            # Сохранение ошибки для графика.
            if epoch % self.report_interval == 0 and eval_set is not None:
                self.losses_train.append(self.__mean_squared_error(y, self.predict(X)))
                self.losses_test.append(self.__mean_squared_error(eval_set[1], self.predict(eval_set[0])))

    def fit_vec(self, X, y, eval_set=None):
        """Метод обучения с использованием векторизации."""

        # Количество объектов и признаков.
        self.num_obj, self.num_feat = X.shape
        # Инициализация весов (вектор 1 x num_feat) и свободного члена по закону нормального распределения.
        self.weights = np.random.randn(self.num_feat, 1) * 0.001
        self.bias = np.random.randn() * 0.001
        if eval_set is not None:
            self.losses_train = []
            self.losses_test = []

        # Эпохи - кол-во проходов по всему набору данных или кол-во шагов по антиградиенту.
        for epoch in range(self.epochs):
            # Линейная комбинация признаков.
            F = X.dot(self.weights) + self.bias

            # Расчёт градиента.
            grad_w = (2 / self.num_obj) * np.sum((F.reshape(self.num_obj, 1) - y.reshape(self.num_obj, 1)) * X, axis=0)
            grad_b = (2 / self.num_obj) * np.sum((F.reshape(self.num_obj, 1) - y.reshape(self.num_obj, 1)), axis=0)

            # Корректировка весов и свободного члена. Шаг в сторону антиградиента.
            self.weights = self.weights - self.learning_rate * grad_w.reshape(self.num_feat, 1)
            self.bias = self.bias - self.learning_rate * grad_b

            # Сохранение ошибки для графика.
            if epoch % self.report_interval == 0 and eval_set is not None:
                self.losses_train.append(self.__mean_squared_error(y, self.predict(X)))
                self.losses_test.append(self.__mean_squared_error(eval_set[1], self.predict(eval_set[0])))

    def predict(self, X):
        """Функция предсказаний."""
        predictions = np.array([x.reshape(1, self.num_feat).dot(self.weights) + self.bias for x in X])
        return predictions.reshape(*X.shape)


class LogisticRegression:
    """Класс логистической регрессии."""

    def __init__(self, learning_rate=0.005, reg_type=None, reg_coeff=0, epochs=100, report_interval=1):
        available_reg_types = (None, "l1", "l2", "elasticnet")
        if reg_type not in available_reg_types:
            raise ValueError(f"Wrong regularization type! Available types: {available_reg_types}")
            
        self.learning_rate = learning_rate  # шаг обучения
        self.reg_type = reg_type  # тип регуляризации: "l1", "l2", "elasticnet"
        self.reg_coeff = reg_coeff  # коэффициент регуляризации
        self.epochs = epochs  # кол-во шагов градиентного спуска
        self.report_interval = report_interval  # через какой интервал считать характеристики

        self.num_obj = None  # кол-во объектов в обучающем наборе
        self.num_feat = None  # кол-во признаков
        self.weights = None  # веса
        self.bias = None  # свободный член
        # Списки ошибок.
        self.losses_train = None
        self.losses_test = None

    @staticmethod
    def __log_loss(y_true, y_pred):
        """Логистическая функция потерь."""
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=0) / len(y_true)

    @staticmethod
    def __sigmoid(x):
        """Сигмоида."""
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, eval_set=None):
        """Метод обучения модели с использованием цикла."""

        # Количество объектов и признаков.
        self.num_obj, self.num_feat = X.shape
        # Инициализация весов (вектор 1 x num_feat) и свободного члена по закону нормального распределения.
        self.weights = np.random.randn(self.num_feat, 1) * 0.001
        self.bias = np.random.randn() * 0.001
        if eval_set is not None:
            self.losses_train = []
            self.losses_test = []

        # Эпохи - кол-во проходов по всему набору данных или кол-во шагов по антиградиенту.
        for epoch in range(self.epochs):
            # Инициализация градиента.
            grad_w = np.zeros((self.num_feat, 1))
            grad_b = 0
            
            # Цикл по всем объектам.
            for i in range(self.num_obj):
                # Линейная комбинация признаков.
                # Матричное произведение Xi (1 x num_feat) на w (num_feat x 1) плюс bias.
                f = (X[i].reshape(1, self.num_feat).dot(self.weights) + self.bias)[0][0]  # f - скаляр
                # Результат подаётся на вход сигмоиде.
                a = self.__sigmoid(f)
                # Расчёт градиента.
                grad_w += (a - y[i]) * X[i].reshape(self.num_feat, 1)
                grad_b += (a - y[i])

            # Усреднение градиента.
            grad_w /= self.num_obj
            grad_b /= self.num_obj

            # Корректировка весов и свободного члена. Шаг в сторону антиградиента.
            self.weights = self.weights - self.learning_rate * grad_w
            self.bias = self.bias - self.learning_rate * grad_b

            # Сохранение ошибки для графика.
            if epoch % self.report_interval == 0 and eval_set is not None:
                self.losses_train.append(self.__log_loss(y, self.predict(X)))
                self.losses_test.append(self.__log_loss(eval_set[1], self.predict(eval_set[0])))

    def fit_vec(self, X, y, eval_set=None):
        """Метод обучения с использованием векторизации."""

        # Количество объектов и признаков.
        self.num_obj, self.num_feat = X.shape
        # Инициализация весов (вектор 1 x num_feat) и свободного члена по закону нормального распределения.
        self.weights = np.random.randn(self.num_feat, 1) * 0.001
        self.bias = np.random.randn() * 0.001
        if eval_set is not None:
            self.losses_train = []
            self.losses_test = []

        # Эпохи - кол-во проходов по всему набору данных или кол-во шагов по антиградиенту.
        for epoch in range(self.epochs):
            # Линейная комбинация признаков.
            F = X.dot(self.weights) + self.bias
            # Результат подаётся на вход сигмоиде.
            A = self.__sigmoid(F)

            # Расчёт градиента.
            grad_w = np.sum((A.reshape(self.num_obj, 1) - y.reshape(self.num_obj, 1)) * X, axis=0) / self.num_obj
            grad_b = np.sum((A.reshape(self.num_obj, 1) - y.reshape(self.num_obj, 1)), axis=0) / self.num_obj
            grad_reg = 0

            # Расчёт регуляризационного слагаемого.
            if self.reg_type is not None:
                if self.reg_type == "l2":
                    grad_reg = 2 * self.reg_coeff * self.weights
                elif self.reg_type == "l1":
                    grad_reg = self.reg_coeff * np.sign(self.weights)
                elif self.reg_type == "elasticnet":
                    grad_reg = self.reg_coeff * (2 * self.weights + np.sign(self.weights))

            # Корректировка весов и свободного члена. Шаг в сторону антиградиента.
            self.weights = self.weights - self.learning_rate * (grad_w.reshape(self.num_feat, 1) + grad_reg)
            self.bias = self.bias - self.learning_rate * grad_b

            # Сохранение ошибки для графика.
            if epoch % self.report_interval == 0 and eval_set is not None:
                self.losses_train.append(self.__log_loss(y, self.predict(X)))
                self.losses_test.append(self.__log_loss(eval_set[1], self.predict(eval_set[0])))

    def predict(self, X):
        """Функция предсказаний."""
        return np.array([self.__sigmoid((x.reshape(1, self.num_feat).dot(self.weights) + self.bias)[0][0]) for x in X])
        