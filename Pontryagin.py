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


def crearDinamica_x(A,B,epsilon,a):
    """
        Crea dinámica de u en funcion de una matriz de interaccion A\\
        y una matriz de acoplamiento cruzado B, ademas de un factor de escala epsilon\\
        ####################################################\n
        Inputs:\\
        A: Matriz de tamano nxn, es un numpy array\\
        B: Matriz de tamano 2x2, es un numpy array\\
        epsilon: float, es un factor de escala temporal\\
        ####################################################\\
        Outputs:\\
        dinamica: Funcion que toma como entrada\\
            x: Vector de estado de tamano 2n, con las primeras n siendo u,\\
               las segundas n siendo v\\
            C: Control, es un float\\
            y tiene como salida f(x,C), que es un vector de tamano 2*N\\
    """
    N = np.shape(A)[0]
    def dinamica(x,C):
        """
            Dinamica del vector de estado, esto es, de las variables u y v\\
            Del problema original, codificados en el vector x del dominio\\
            Considerando además un control C en esta misma función\\
            ####################################################\\
            Inputs:\\
            x: ndarray, vector de tamano 2N. Con coordenadas representando:\\
                x[:N]: Vector u\\
                x[N:2N]: Vector v\\
            C: float, es el control que determina interaccion con resto de neuronas\\
            ####################################################\\
            Outputs:\\
            salida: Vector de tamano 2N, con la derivada temporal de x del input
        """
        salida = np.zeros(2*N)
        for k in range(N):
            uk = x[k]
            vk = x[N+k]
            sin_sumatoria_u = (1/epsilon)*(uk - (1/3) * uk ** 3 - vk)
            sin_sumatoria_v = uk + a
            sumatoria_u = 0
            sumatoria_v = 0
            for j in range(N):
                uj = x[j]
                vj = x[N+j]
                sumatoria_u += A[k,j] * (B[0,0] * (uj-uk) + B[0,1] * (vj-vk))
                sumatoria_v += A[k,j] * (B[1,0] * (uj-uk) + B[1,1] * (vj-vk))
            sumatoria_u *= C/epsilon
            sumatoria_v *= C
            sin_sumatoria_u += sumatoria_u
            sin_sumatoria_v += sumatoria_v
            salida[k] = sin_sumatoria_u
            salida[N+k] = sin_sumatoria_v
        return salida
    return dinamica

def crearDinamica_p(A, B, epsilon, a):
    """
        Crea función de dinamica de p, determinado por matriz de interaccion A\\
        y matriz de interaccion cruzada B, ademas de un parametro de escala temporal\\
        epsilon y la propia regulacion de cada neurona, codificada en el parametro a\\
        Esta dinamica viene determinada por el principio de Maximo de Pontryagin.\\
        ####################################################\\
        Inputs:\\
        A: ndarray, matriz de tamano NxN, es la matriz de interaccion\\
        B: ndarray, matriz de 2x2, es la matriz de acoplamiento cruzado\\
        epsilon: float, es escala temporal de evolucion de u\\
        a: float, es un parametro que determina la regulacion de la neurona aislada\\
        ####################################################\\
        Outputs:\\
        dinamica: Funcion que toma como entrada\\
            x: Vector de estado de tamano 2n, con las primeras n siendo u,\\
               las segundas n siendo v\\
            p: Vectores adjuntos de vector x\\
            C: Control, es un float\\
            y tiene como salida f(x,p,C), que es un vector de tamano 2*N y es la derivada de p\\
    """
    N = np.shape(A)[0]
    def dinamica(x,p,C):
        """
            Dinamica del vector adjunto, esto es, de las variables pu y pv\\
            Del problema original, codificados en el vector p del dominio\\
            Considerando además un control C en esta misma función\\
            ####################################################\\
            Inputs:\\
            x: ndarray, vector de tamano 2N. Con coordenadas representando:\\
                x[:N]: Vector u\\
                x[N:2N]: Vector v\\
            p: ndarray, vector adjunto de x, con correspondencia analoga a x\\
            C: float, es el control que determina interaccion con resto de neuronas\\
            ####################################################\\
            Outputs:\\
            salida: Vector de tamano 2N, con la derivada temporal de p del input
        """
        der_p = np.zeros(2*N)
        for i in range(N):
            ui = x[i]
            pui = p[i]
            pvi = p[N+i]
            sum_nada = (1/epsilon) * pui * (1 - ui ** 2)
            sum_control = 0
            sum_u1 = 0
            sum_u2 = 0
            sum_v1 = 0
            sum_v2 = 0
            for j in range(N):
                uj = x[j]
                puj = p[j]
                pvj = p[N+j]
                sum_control += ui - uj
                sum_u1 += puj * A[j,i]
                sum_u2 += A[i,j]
                sum_v1 += pvj * A[j,i]
                sum_v2 += A[i,j]
            final = 2 * sum_control + sum_nada + C*B[0,0]/epsilon * (sum_u1 - pui * sum_u2)\
                    + pvi + C*B[1,0] * (sum_v1 - pvi * sum_v2)
            der_p[i] = (-1) * final
            final_v = - pui / epsilon + (C * B[0,1] / epsilon) * (sum_u1 - pui * sum_u2)\
                      + C * B[1,1] * (sum_v1 - pvi * sum_v2)
            der_p[N+i] = (-1) * final_v
        return der_p
    return dinamica

def crearDominioControl(A,B,epsilon,a):
    """
        Crea funcion que determinara control, aprovechando que sabemos que el control es de\\
        tipo bang bang. Para esto se le da a la funcion la matriz de interaccion A y\\
        un acoplamiento cruzado B, ademas de una escala temporal epsilon y un escalar que\\
        determina la interaccion interna de cada neurona para regular su dinamica\\
        ####################################################\\
        Inputs:\\
        A: ndarray, matriz de tamano NxN, es la matriz de interaccion\\
        B: ndarray, matriz de 2x2, es la matriz de acoplamiento cruzado\\
        epsilon: float, es escala temporal de evolucion de u\\
        a: float, es un parametro que determina la regulacion de la neurona aislada\\
        ####################################################\\
        Outputs:\\
        entrada: Funcion que da la entrada del control tipo bang bang.
    """
    N = np.shape(A)[0]
    def entrada(t,x,p):
        """
            Entrada de funcion que determinara valor del control
        """
        val = 0
        for k in range(N):
            for j in range(N):
                dif_u = x[j] - x[k]
                dif_v = x[N+j] - x[N+k]
                acomp_u = (B[0,0] * p[k] / epsilon) + p[N+k] * B[1,0]
                acomp_v = (B[0,1] * p[k] / epsilon) + p[N+k] * B[1,1]
                val += A[k,j] * (dif_u * acomp_u +dif_v * acomp_v)
        return (-1)*val
    return entrada

def crearDinamicaTotal(A, B, epsilon, a, mini, maxi):
    """
        Junta las dinamicas de x y p para llevar al final al solver\\
        Esto crea la función que se debe aplicar al final allí.
    """
    N = np.shape(A)[0]
    def derivada(t,y):
        """
            Derivada total del sistema
        """
        x = y[:2*N]
        p = y[2*N:4*N]
        resultado = np.zeros(4*N)
        der_x = crearDinamica_x(A, B, epsilon, a)
        der_p = crearDinamica_p(A, B, epsilon, a)
        control = crearDominioControl(A, B, epsilon, a)
        input_control = control(t,x,p)
        res_control = BangBangCustom(mini, maxi)(input_control)
        der_x_eval = der_x(x,res_control)
        der_p_eval = der_p(x,p,res_control)
        resultado[:2*N] = der_x_eval
        resultado[2*N:4*N] =der_p_eval
        return resultado
    return derivada
