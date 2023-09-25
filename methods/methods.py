import numpy as np


__all__ = ["determination_coeff", "normalize", "gen_cluster"]


def determination_coeff(y_true, y_pred):
    """Коэффициент детерминации."""
    return (1 - (np.sum((y_true - y_pred) ** 2, axis=0) / np.sum((y_true - np.mean(y_true)) ** 2, axis=0)))[0]


def normalize(data):
    """Функция нормализации."""
    return (data - data.mean(axis=0).reshape(1, -1)) / data.std(axis=0).reshape(1, -1)


def gen_cluster(dots, radius=1, center=(0, 0)):
    """Генерирует равномерное распределение точек внутри окружности с заданным радиусом и центром."""
    x = radius * (2 * np.random.random(dots) - 1)
    y = ((radius ** 2 - x ** 2) ** 0.5) * (2 * np.random.random(dots) - 1)
    return np.stack((x + center[0], y + center[1]))
    