from Pontryagin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import minimize, fsolve
from time import time

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
prob_conec = 0.3

A = wattsstrogatzmatrix(N,vec, prob_conec)
# A = ring_coupling(N, vec)
phi = np.pi/2-0.001
B = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
eps = 2
a = 0.5
tf = 20

cs = 1

mini = 0.8 * cs
maxi = cs

derivada_total = crearDinamicaTotal(A, B, eps, a, mini, maxi)

x_0 = np.random.rand(2*N)
print("x_0 es",x_0)

# Hay que adaptar la función derivada total para que en su dominio pueda recibir
# todos los tiempos de forma simultánea

def derivada_todos_tiempos(ts, ys):
    """
        Es la función derivada total, adaptada para recibir\\
        todos los tiempos de forma simultánea. Se considerará\\
        que ts es un vector de tamano M, y que ys es una matriz de tamano 4N x M.\\
        #################################\\
        Inputs:\\
        ts: array de tamano m, es creciente parte en 0 y termina en tf\\
        ys: matriz de dimensiones 4NxM\\
            ys[:,i]: Valores del vector de estado en el tiempo de ts[i]\\
            ys[j,:]: Toda la evolucion temporal de la coordenada j del vector de estado\\
        #################################\\
        Outputs:\\
        sols: matriz de tamano 4NxM\\
            sols[:,i]: Derivada del vector de estado en el tiempo ts[i]\\
            sols[j,:]: Derivada a lo largo del tiempo de la coordenada j del vector de estado\\
    """
    M = np.shape(ts)[0]
    dim = np.shape(ys)[0]
    sols = np.zeros((dim, M))
    for i in range(M):
        t = ts[i]
        y = ys[:,i]
        deriv = derivada_total(t,y)
        sols[:,i] = deriv
    return sols


# Ahora vamos a hacer la función que resuelve el bvp

def bc(x0, xT):
    """
        Funcion que trae las condiciones de contorno\\
        En concreto, x0 son los valores iniciales\\
        Y los pT son los valores finales.\\
        #################################\\
        Inputs:\\
        x0: array de numpy, es de dimension 4*N, trae las condiciones iniciales\\
        xT: array de numpy, es de dimension 4*N, trae las condiciones finales\\
        #################################\\
        Outputs:\\
        borde: array de numpy, trae lo que deberían dar los bordes (igualado a 0)
    """
    assert np.shape(x0)[0] == np.shape(xT)[0] and np.shape(x0)[0] == 4*N
    ini = x0[:2*N] - x_0
    fin = xT[2*N:4*N]
    return np.hstack((ini, fin))

# tinca_final = np.zeros(4*N)
# tinca_inicial = np.zeros(4*N)
# print(bc(tinca_inicial, tinca_final))

angulos_nodos = np.linspace(0,2*np.pi, num=N, endpoint=False)
x_nodos = np.cos(angulos_nodos)
y_nodos = np.sin(angulos_nodos)

pasos_t = 500
tiempos = np.linspace(0, tf, num=pasos_t)

tincada_sol_inicial = np.zeros((4*N, pasos_t))

sol = solve_bvp(derivada_todos_tiempos, bc, x=tiempos, y=tincada_sol_inicial)

x_func = sol.y[:2*N]
p_func = sol.y[2*N:4*N]

neurs = sol.y[:N]
inhib = sol.y[N:2*N]

adjunto_neur = sol.y[2*N:3*N]
adjunto_inhib = sol.y[3*N:4*N]

new_t = sol.x
############################################
# Acá crearemos la funcion de control para graficarla en la solucion

input_control = crearDominioControl(A,B, eps, a)


control = BangBangCustom(mini, maxi)(input_control(new_t, x_func, p_func))

taman = 6
multiplicador = 3
fig, ax = plt.subplots(1,3, figsize=(multiplicador * taman, taman))

ax[0].plot(tiempos, neurs[0], label="Neurona 1")
ax[0].plot(tiempos, neurs[1], label="Neurona 2")

ax[0].plot(tiempos, inhib[0], label="Inhibitorio 1")
ax[0].plot(tiempos, inhib[1], label="Inhibitorio 2")


ax[1].plot(new_t, adjunto_neur[0], label="Adjunto Neurona 1")
ax[1].plot(new_t, adjunto_neur[1], label="Adjunto Neurona 2")

ax[1].plot(new_t, adjunto_inhib[0], label="Adjunto Inhibitorio 1")
ax[1].plot(new_t, adjunto_inhib[1], label="Adjunto Inhibitorio 2")


ax[2].plot(new_t, control, label="Control")


ax[0].set_xlabel("tiempo", fontsize=15)
ax[1].set_xlabel("tiempo", fontsize=15)
ax[2].set_xlabel("tiempo", fontsize=15)

ax[0].set_ylabel("Variables de Estado", fontsize=20)
ax[1].set_ylabel("Variables Adjuntas", fontsize=20)
ax[2].set_ylabel("Control", fontsize=20)


for i in range(3):
    ax[i].legend()
    ax[i].tick_params(labelsize=15)

# for i in range(2):
#     for j in range(2):
#         ax[i,j].legend()

plt.subplots_adjust(hspace=0.3)
plt.tight_layout()

fig.savefig("Grafico_estado_1 con "+str(N)+" neuronas"+".pdf")


fig2, ax2 = plt.subplots(1,1,figsize=(taman, taman))


for i in range(N):
    ax2.plot(x_nodos[i], y_nodos[i], "*", label="Neurona "+str(i+1))
    for j in range(i-1,N):
        lista_x = [x_nodos[i], x_nodos[j]]
        lista_y = [y_nodos[i], y_nodos[j]]
        ax2.plot(lista_x, lista_y, color="Navy", alpha=A[i,j]/N)

ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(-1.1, 1.1)
ax2.set_aspect("equal")
ax2.legend()
ax2.set_xticks([])
ax2.set_yticks([])

fig2.savefig("Conectividad con "+str(N)+" neuronas"+".pdf")