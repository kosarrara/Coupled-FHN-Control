from Pontryagin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def ring_coupling(size, neighbors=1):
    coupling_matrix = np.zeros((size, size))
    
    if size > 2 * neighbors:
        correction = -2 * neighbors
    else:
        correction = -size + 1
    
    for i in range(size):
        if i + neighbors <= size and i - neighbors >= 0:
            coupling_matrix[i, (i + np.arange(1, neighbors + 1))%size] += 1
            coupling_matrix[i, (i - np.arange(1, neighbors + 1))%size] += 1
            coupling_matrix[i, i] = correction
        else:
            indices = np.unique([j % size for j in range(i - neighbors, i + neighbors + 1) if j != i])
            coupling_matrix[i, indices] += 1
            coupling_matrix[i, i] = correction
    
    return coupling_matrix

def wattsstrogatzmatrix(size, neighbors, rewiring_prob):
    coupling_matrix = ring_coupling(size, neighbors=neighbors)
    
    for i in range(size):
        for j in range(i, size):
            if coupling_matrix[i, j] == 1:
                if np.random.rand() < rewiring_prob:
                    coupling_matrix[i, j] = 0
                    rand_index = np.random.randint(0, size)
                    
                    while rand_index == i or coupling_matrix[i, rand_index] == 1:
                        rand_index = np.random.randint(0, size)
                    
                    coupling_matrix[i, rand_index] = 1
                    coupling_matrix[rand_index, i] = 1
    
    return coupling_matrix


N = 2
vec = 1
prob_conec = 0.5

A = wattsstrogatzmatrix(N,vec, prob_conec)
# A = ring_coupling(N, vec)
phi = np.pi/2
B = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
eps = 0.1
a = 0.05
tf = 15

cs = 0.1

mini = -cs
maxi = cs

derivada_total = crearDinamicaTotal(A, B, eps, a, mini, maxi)

x_0 = np.random.rand(2*N)
print("x_0 es",x_0)

def cond_ini(x,p):
    """
        Funcion que funcionara para definir el shooting luego. Dada una condicion inicial\\
        de P, devuelve la solucion de la edo
    """
    final = np.zeros(4*N)
    final[:2*N] = x
    final[2*N:4*N] = p
    solver = solve_ivp(derivada_total, (0,tf), final, method="RK45", dense_output=True)
    return solver

def recuperar_control(solu):
    """
        Devuelve la función de control sobre la solucion
    """
    funcion = solu.sol
    control = crearDominioControl(A, B, eps, a)
    val_control = BangBangCustom(mini, maxi)
    def controles_aplicados(t):
        """
            El control relativo a solu
        """
        y = funcion(t)
        x = y[:2*N]
        p = y[2*N:4*N]
        valor = val_control(control(t,x,p))
        return valor
    return controles_aplicados

def minimizar_shooting(p):
    """
        Esta función devuelve la norma de p(T)\\
        Se va a minimizar para que de 0. Se asume que x0=x_0\\
    """
    solucion = cond_ini(x_0, p)
    p_fin = solucion.sol(tf)[2*N:4*N]
    return np.linalg.norm(p_fin)

p_random = np.random.rand(2*N)

fin = minimize(minimizar_shooting, p_random)
p_optimo_estado = fin.success
p_optimo = fin.x
print(p_optimo_estado)
if p_optimo_estado:
    print("El p_0 optimo es ",p_optimo)
    test = cond_ini(x_0, p_optimo)
else:
    print("minimize no se la pudo!")
    print("Usaremos p_0 igual a ", p_random)
    test = cond_ini(x_0, p_random)

rec = recuperar_control(test)

tiempos = np.linspace(0,tf, num=1000)
control_apic = rec(tiempos)


fig, ax = plt.subplots(1,3, figsize=(15,5))

ax[0].plot(tiempos, control_apic)
ax[0].set_title("Control")

respuestas = test.sol(tiempos)

neur_1 = respuestas[0,:]
neur_2 = respuestas[1,:]
# neur_3 = respuestas[3,:]

v_1 = respuestas[N+0,:]
v_2 = respuestas[N+1,:]
# v_3 = respuestas[N+2,:]

ax[1].plot(tiempos, neur_1, label="Neurona 1")
ax[1].plot(tiempos, neur_2, label="Neurona 2")
# ax[1].plot(tiempos, neur_3, label="Neurona 3")
ax[1].plot(tiempos, v_1, label="Inhibitorio 1")
ax[1].plot(tiempos, v_2, label="Inhibitorio 2")
# ax[1].plot(tiempos, v_3, label="Inhibitorio 3")
ax[1].legend()
ax[1].set_title("Vector de Estado")


p1 = respuestas[2*N+0,:]
p2 = respuestas[2*N+1,:]
# p3 = respuestas[2*N+2,:]

ax[2].plot(tiempos, p1, label="Vector Adjunto 1")
ax[2].plot(tiempos, p2, label="Vector Adjunto 2")
# ax[2].plot(tiempos, p3, label="Vector Adjunto 3")
ax[2].legend()
ax[2].set_title("Vector Adjunto")

plt.show()
