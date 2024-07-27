from typing import Tuple
import numpy as np
# Inicializacion de parametros
w: float = 10.
b: float = 10.

# Definicion de la funcion
def f(x: np.array) -> np.array:
    global w,b
    return w*x + b

#Definicion de la funcion de costo
def J(x: np.array, y: np.array) -> float:
    return ((f(x) - y)**2).sum()

#Definicion de las derivadas parciales
def dJ_dw(x: np.array, y: np.array) -> float:
    global w,b
    return 2*((f(x) - y)*x).sum()

#Definicion de las derivadas parciales
def dJ_db(x: np.array, y: np.array) -> float:
    global w,b
    return 2*(f(x) - y).sum()

def grad_J(x: np.array, y: np.array) -> Tuple[float, float]:
    return dJ_dw(x, y), dJ_db(x,y)

def decenso_de_gradiente(x: np.array, y: np.array, alpha: float) -> None:
    global w,b
    w -= alpha * dJ_dw(x, y)
    b -= alpha * dJ_db(x, y)

def training(x: np.array, y: np.array, epochs: int, alpha: float) -> None:
    for _ in range(epochs):
        decenso_de_gradiente(x, y, alpha)
        print('El costo es: ', J(x, y))

if __name__ == '__main__':
    # modelo esperado: f(x) = 3x
    x: np.array = np.array([0, 1, 2, 3])
    y: np.array = np.array([0, 3, 6, 9])

    epochs: int = 1000
    alpha: float = 1e-2

    training(x, y, epochs, alpha)

    print(f'El modelo f es: {w}x + {b}')
