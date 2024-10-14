import numpy as np
import matplotlib.pyplot as plt

# Определяем целевую функцию
def F(x1, x2):
    return x1**2 - x1 * x2 + 2 * x2**2 + 5 * x1 + 10 * x2

# Градиент функции
def gradient(x1, x2):
    dF_dx1 = 2 * x1 - x2 + 5
    dF_dx2 = -x1 + 4 * x2 + 10
    return np.array([dF_dx1, dF_dx2])

# Градиентный спуск
def gradient_descent(x0, steps, step_size):
    trajectory = [x0]
    x = np.array(x0)
    
    for _ in range(steps):
        grad = gradient(x[0], x[1])
        x = x - step_size * grad
        trajectory.append(tuple(x))
    
    print("Gradient Descent Trajectory:", trajectory)
    return np.array(trajectory)

# Метод наискорейшего спуска
def steepest_descent(x0, steps):
    trajectory = [x0]
    x = np.array(x0)
    
    for _ in range(steps):
        grad = gradient(x[0], x[1])
        step_size = 1 / np.linalg.norm(grad)  # Длина шага равна обратной норме градиента
        x = x - step_size * grad
        trajectory.append(tuple(x))
    
    print("Steepest Descent Trajectory:", trajectory)
    return np.array(trajectory)

# Метод Гаусса-Зейделя
def gauss_seidel(x0, steps):
    x1, x2 = x0
    trajectory = [x0]
    
    for _ in range(steps):
        x1 = (x2 - 5) / 2
        x2 = (x1 - 10) / 4
        trajectory.append((x1, x2))
    
    print("Gauss-Seidel Trajectory:", trajectory)
    return np.array(trajectory)

# Метод Ньютона
def newton_method(x0):
    x = np.array(x0)
    H = np.array([[2, -1], [-1, 4]])  # Гессиан
    grad = gradient(x[0], x[1])
    x = x - np.linalg.inv(H).dot(grad)  # Обновление точки
    
    trajectory = [x0, tuple(x)]
    print("Newton's Method Trajectory:", trajectory)
    return np.array([x])

# Функция для построения графика
def plot_results(a, b, trajectories, labels):
    # Создаем сетку для контуров
    x1 = np.linspace(a, b, 400)
    x2 = np.linspace(a, b, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = F(X1, X2)

    # Строим контурный график
    plt.figure(figsize=(12, 8))
    contour = plt.contour(X1, X2, Z, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='F(x1, x2)')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    # Отображаем траектории для каждого метода
    for trajectory, label in zip(trajectories, labels):
        plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', label=label)

    # Настройка графика
    plt.title('Методы оптимизации на плоскости $x_1$, $x_2$')
    plt.legend()
    plt.grid(True)
    plt.show()

# Задаем начальную точку
x0 = (0, 0)

# Запускаем методы
gd_trajectory = gradient_descent(x0, 4, step_size=0.1)
sd_trajectory = steepest_descent(x0, 3)
gs_trajectory = gauss_seidel(x0, 2)
newton_trajectory = newton_method(x0)

# Собираем данные для отображения
trajectories = [gd_trajectory, sd_trajectory, gs_trajectory, newton_trajectory]
labels = ['Gradient Descent (4 steps)', 'Steepest Descent (3 steps)', 'Gauss-Seidel (2 steps)', 'Newton\'s Method (1 step)']

# Строим график с контуром
plot_results(-6, 6, trajectories, labels)
