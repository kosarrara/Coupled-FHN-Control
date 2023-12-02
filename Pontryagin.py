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
