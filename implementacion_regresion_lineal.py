from typing import Tuple
import numpy as np
from numpy.typing import NDArray

# Inicializacion de parametros
w: float = 10.
b: float = 10.

# Definicion de la funcion
def f(x: NDArray) -> NDArray:
    global w,b
    return w*x + b

#Definicion de la funcion de costo
def J(x: NDArray, y: NDArray, m: int) -> float:
    return (1/m) * ((f(x) - y)**2).sum()

#Definicion de las derivadas parciales
def dJ_dw(x: NDArray, y: NDArray, m: int) -> float:
    global w,b
    return (2 / m)*((f(x) - y)*x).sum()

#Definicion de las derivadas parciales
def dJ_db(x: NDArray, y: NDArray, m: int) -> float:
    global w,b
    return (2 / m) *(f(x) - y).sum()

def grad_J(x: NDArray, y: NDArray, m: int) -> Tuple[float, float]:
    return dJ_dw(x, y, m), dJ_db(x,y, m)

def decenso_de_gradiente(x: NDArray, y: NDArray, alpha: float, m: int) -> None:
    global w,b
    w -= alpha * dJ_dw(x, y, m)
    b -= alpha * dJ_db(x, y, m)

def training(x: NDArray, y: NDArray, epochs: int, alpha: float) -> None:
    for _ in range(epochs):
        m: int = x.shape[-1]
        decenso_de_gradiente(x, y, alpha, m)
        print('El costo es: ', J(x, y, m))

if __name__ == '__main__':
    # modelo esperado: f(x) = 3x
    x: NDArray = np.array([0, 1, 2, 3])
    y: NDArray = np.array([0, 3, 6, 9])

    epochs: int = 1000
    alpha: float = 1e-2

    training(x, y, epochs, alpha)

    print(f'El modelo f es: {w}x + {b}')
