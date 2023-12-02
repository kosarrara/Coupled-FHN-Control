"""
    La finalidad de este archivo es implementar las herramientas de control óptimo\\
    En el proyecto definido sobre neuronas modeladas mediante FitzHugh-Nagumo (FHN)\\
"""

import numpy as np
import matplotlib.pyplot as plt

def BangBangCustom(minimo: float, maximo: float):
    """
        El control del problema será de tipo bang bang\\
        Por esto, crearemos una función que sea una Heavisde\\
        Esta función devolverá una función con la estructura de tipo Bang Bang\\
        Devuelve una función que cuando t es negativo, devuelve el minimo\\
        En cambio cuando t es positivo, devuelve el máximo\n
        ####################################################\\
        Inputs:\\
        minimo: float, valor minimo, del control bang bang\\
        maximo: float, valor maximo del control bang bang\\
        ####################################################\\
        Outputs\\
        BangBang: Funcion, toma un parámetro real, que cuando es negativo\\
                  devuelve el valor maximo, y cuando es negativo, el minimo\\
    """
    assert minimo <= maximo
    def BangBang(t: float) -> float:
        """
            Control tipo Bang Bang, que cuando t<0 da el valor mínimo permitido\\
            Y cuando t>0 da el valor máximo permitido. Si es 0 se asigna el valor maximo\\
            Aunque en el resutlado final no importa mucho este caso porque ponderará\\
            A ese valor 0\\
            ####################################################\\
            Inputs:\\
            t: float, parámetro que determinará el control\\
            ####################################################\\
            Outputs:\\
            valor = float, resultado del control, será máximo o mínimo\\
                    según los valores permitidos del control\\
        """
        heaviside = (t>0)
        valor = 0.5 * (maximo + minimo) + (maximo - minimo) * (heaviside - 0.5)
        return valor
    return BangBang

def CrearHamiltoniano(A, B, epsilon, a):
    """
        Esta función crea un Hamiltoniano en base a una matriz de interacciones A de tamaño N\\
        Y una matriz de acoplamiento cruzado B, que será de tamaño 2x2. Asi como un factor de tiempo\\
        epsilon, y uno de inhibición a\\
        Este Hamiltoniano tomará como parámetros x, p y C, donde X y P son vectores\\
        de tamaño 2N respectivamente. Las primeras N coordenadas serán relativas\\
        A la variable excitativa, y las siguientes N serán relativas a las inhibitorias\\
        ####################################################\\
        Inputs:\\
        A: Matriz cuadrada de tamaño NxN\\
        B: Matriz cuadrada de rotacion de tamaño 2x2
        ####################################################\\
        Outputs:\\
        Hamiltoniano: Funcion que toma un vector de estado, un vector adjunto y un control 1D\\
    """
    tamano = np.shape(A)[0]
    def Hamiltoniano(x,p,C):
        """
            Hamiltoniano del sistema
        """
        densidad_lagrangiana = 0
        for j in range(tamano):
            for k in range(tamano):
                dif = x[j] - x[k]
                densidad_lagrangiana += 0.5 * dif ** 2
        termino_u = 0
        for k in range(tamano):
            suma_parcial = (1/epsilon) * (x[k] - 1/3 * x[k] ** 3 - x[tamano+k])
            parcial = 0
            for j in range(tamano):
                val_extra = A[k,j] * (B[0,0] * (x[j] - x[k]) + B[0,1] * (x[tamano+j] - x[tamano+k]))
                parcial += val_extra
            parcial *= C/epsilon
            suma_parcial += parcial
            termino_u += suma_parcial * p[k]
        termino_v = 0
        for k in range(tamano):
            suma_parcial = x[k] + a
            for j in range(tamano):
                val = C * A[k,j] * (B[1,0]*(x[j]-x[k]) + B[1,1] * (x[tamano+j] - x[tamano+k]))
                suma_parcial += val
            suma_parcial *= p[tamano+k]
            termino_v += suma_parcial
        H = densidad_lagrangiana + termino_u + termino_v
        return H
    return Hamiltoniano
            