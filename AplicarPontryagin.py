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
prob_conec = 0.5

# A = wattsstrogatzmatrix(N,vec, prob_conec)
A = ring_coupling(N, vec)
phi = np.pi/2-0.001
B = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
eps = 0.5
a = 0.5
tf = 100

cs = 0

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

pasos_t = 300
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
fig, ax = plt.subplots(2,2, figsize=(2*taman+0.1*taman, 2 * taman + 0.1*taman))

ax[0,0].plot(tiempos, neurs[0], label="Neurona 1")
ax[0,0].plot(tiempos, neurs[1], label="Neurona 2")

ax[0,0].plot(tiempos, inhib[0], label="Inhibitorio 1")
ax[0,0].plot(tiempos, inhib[1], label="Inhibitorio 2")


ax[0,1].plot(new_t, adjunto_neur[0], label="Adjunto Neurona 1")
ax[0,1].plot(new_t, adjunto_neur[1], label="Adjunto Neurona 2")

ax[0,1].plot(new_t, adjunto_inhib[0], label="Adjunto Inhibitorio 1")
ax[0,1].plot(new_t, adjunto_inhib[1], label="Adjunto Inhibitorio 2")


ax[1,0].plot(new_t, control, label="Control")

for i in range(N):
    ax[1,1].plot(x_nodos[i], y_nodos[i], "*", label="Neurona "+str(i+1))
    for j in range(i-1,N):
        lista_x = [x_nodos[i], x_nodos[j]]
        lista_y = [y_nodos[i], y_nodos[j]]
        ax[1,1].plot(lista_x, lista_y, color="Navy", alpha=A[i,j]/N)

ax[1,1].set_xlim(-1.1, 1.1)
ax[1,1].set_ylim(-1.1, 1.1)

for i in range(2):
    for j in range(2):
        ax[i,j].legend()


plt.show()



# def cond_ini(x,p):
#     """
#         Funcion que funcionara para definir el shooting luego. Dada una condicion inicial\\
#         de P, devuelve la solucion de la edo
#     """
#     final = np.zeros(4*N)
#     final[:2*N] = x
#     final[2*N:4*N] = p
#     solver = solve_ivp(derivada_total, (0,tf), final, method="RK45", dense_output=True)
#     return solver

# def recuperar_control(solu):
#     """
#         Devuelve la función de control sobre la solucion
#     """
#     funcion = solu.sol
#     control = crearDominioControl(A, B, eps, a)
#     val_control = BangBangCustom(mini, maxi)
#     def controles_aplicados(t):
#         """
#             El control relativo a solu
#         """
#         y = funcion(t)
#         x = y[:2*N]
#         p = y[2*N:4*N]
#         valor = val_control(control(t,x,p))
#         return valor
#     return controles_aplicados

# def minimizar_shooting(p):
#     """
#         Esta función devuelve la norma de p(T)\\
#         Se va a minimizar para que de 0. Se asume que x0=x_0\\
#     """
#     solucion = cond_ini(x_0, p)
#     p_fin = solucion.sol(tf)[2*N:4*N]
#     return p_fin

# p_random = np.random.rand(2*N)
# print("El primer intento sera con ",p_random)

# # ti = time()
# # fin = minimize(minimizar_shooting, p_random)
# # tfinal = time()
# # print("Minimize se demoro ",tfinal-ti,"(s)")
# # p_optimo_estado = fin.success
# #p_optimo = fin.x
# ini = np.zeros(2*N)
# p_optimo_estado = fsolve(minimizar_shooting, x0=ini)

# print(p_optimo_estado, "\n")

# # print("El p_0 optimo es ",p_optimo_estado.x,"\n")


# test = cond_ini(x_0, p_optimo_estado)

# #     print("Usaremos p_0 igual a ", p_random)
# #     test = cond_ini(x_0, p_random)

# rec = recuperar_control(test)

# tiempos = np.linspace(0,tf, num=1000)
# control_apic = rec(tiempos)


# fig, ax = plt.subplots(1,3, figsize=(15,5))

# ax[0].plot(tiempos, control_apic,".")
# ax[0].set_title("Control")

# respuestas = test.sol(tiempos)

# neur_1 = respuestas[0,:]
# neur_2 = respuestas[1,:]
# # neur_3 = respuestas[3,:]

# v_1 = respuestas[N+0,:]
# v_2 = respuestas[N+1,:]
# # v_3 = respuestas[N+2,:]

# ax[1].plot(tiempos, neur_1, label="Neurona 1")
# ax[1].plot(tiempos, neur_2, label="Neurona 2")
# # ax[1].plot(tiempos, neur_3, label="Neurona 3")
# ax[1].plot(tiempos, v_1, label="Inhibitorio 1")
# ax[1].plot(tiempos, v_2, label="Inhibitorio 2")
# # ax[1].plot(tiempos, v_3, label="Inhibitorio 3")
# ax[1].legend()
# ax[1].set_title("Vector de Estado")


# p1 = respuestas[2*N+0,:]
# p2 = respuestas[2*N+1,:]
# # p3 = respuestas[2*N+2,:]

# ax[2].plot(tiempos, p1, label="Vector Adjunto 1")
# ax[2].plot(tiempos, p2, label="Vector Adjunto 2")
# # ax[2].plot(tiempos, p3, label="Vector Adjunto 3")
# ax[2].legend()
# ax[2].set_title("Vector Adjunto")

# plt.show()
