import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt


# ----------- CONSTANTS -----------
# Grid parameters
H = -0.0001
ε = 0.001
# Parameters
Re_m = 10
Re_n = 5
Da = 0.07
γ = 20
β = 2
B = 15
θ_c = 0
# ----------------------------------


class vector:
    y = 0.0
    ζ = 0.0
    θ = 0.0
    φ = 0.0
    z = 1.0

    def roll_back(self, η_1, η_2):
        self.y = η_1
        self.ζ = 0.0
        self.θ = η_2
        self.φ = 0.0
        self.z = 1.0


class var_vector:
    py_1 = 1.0
    py_2 = 0.0
    pθ_1 = 0.0
    pθ_2 = 1.0
    sy_1 = 0.0
    sy_2 = 0.0
    sθ_1 = 0.0
    sθ_2 = 0.0
    z = 1.0

    def roll_back(self):
        self.py_1 = 1.0
        self.py_2 = 0.0
        self.pθ_1 = 0.0
        self.pθ_2 = 1.0
        self.sy_1 = 0.0
        self.sy_2 = 0.0
        self.sθ_1 = 0.0
        self.sθ_2 = 0.0
        self.z = 1.0


class parameter:
    η_1 = 0.10323
    η_2 = 0.64076



class shoot:
    W = np.zeros((2, 2))

    var_vec = var_vector()
    var_vec1 = var_vector()

    res = parameter()
    res.η_1, res.η_2 = 0, 0
    res1 = parameter()

    vec = vector()
    vec.y = res1.η_1
    vec.θ = res1.η_2

    vec1 = vector()
    vec.y = res1.η_1
    vec.θ = res1.η_2


    def roll_back(self):
        self.res.η_1, self.res.η_2 = self.res1.η_1, self.res1.η_2
        self.vec.roll_back(self.res.η_1, self.res.η_2)
        self.vec1.roll_back(self.res.η_1, self.res.η_2)
        self.var_vec.roll_back()
        self.var_vec1.roll_back()


    def runge_vec_prepare(self, step, k):
        self.vec1.y = self.vec.y + step*k.y
        self.vec1.ζ = self.vec.ζ + step*k.ζ
        self.vec1.θ = self.vec.θ + step*k.θ
        self.vec1.φ = self.vec.φ + step*k.φ
        self.vec1.z = self.vec.z + step


    def runge_vec(self):
        k1 = self.count_k_vec()
        self.runge_vec_prepare(H/2, k1)
        k2 = self.count_k_vec()
        self.runge_vec_prepare(H/2, k2)
        k3 = self.count_k_vec()
        self.runge_vec_prepare(0, k3)
        k4 = self.count_k_vec()

        self.vec.y = self.vec.y + H/6*(k1.y + 2*k2.y + 2*k3.y + k4.y)
        self.vec.ζ = self.vec.ζ + H/6*(k1.ζ + 2*k2.ζ + 2*k3.ζ + k4.ζ)
        self.vec.θ = self.vec.θ + H/6*(k1.θ + 2*k2.θ + 2*k3.θ + k4.θ)
        self.vec.φ = self.vec.φ + H/6*(k1.φ + 2*k2.φ + 2*k3.φ + k4.φ)
        self.vec.z = self.vec.z + H
    
    
    def count_k_vec(self):
        vec = self.vec1
        k = vector()

        k.y = vec.ζ
        k.ζ = (vec.ζ - Da*(1 - vec.y)*np.exp((γ*vec.θ)/(γ + vec.θ)))*Re_m
        k.θ = vec.φ
        k.φ = (vec.φ - B*Da*(1 - vec.y)*np.exp((γ*vec.θ)/(γ + vec.θ)) + β*(vec.θ - θ_c))*Re_n
        k.z = vec.z
        return k


    def runge_var_vec_prepare(self, step, k):
        var_vec = self.var_vec

        self.var_vec1.py_1 = var_vec.py_1 + step*k.py_1
        self.var_vec1.py_2 = var_vec.py_2 + step*k.py_2
        self.var_vec1.pθ_1 = var_vec.pθ_1 + step*k.pθ_1
        self.var_vec1.pθ_2 = var_vec.pθ_2 + step*k.pθ_2
        self.var_vec1.sy_1 = var_vec.sy_1 + step*k.sy_1
        self.var_vec1.sy_2 = var_vec.sy_2 + step*k.sy_2
        self.var_vec1.sθ_1 = var_vec.sθ_1 + step*k.sθ_1
        self.var_vec1.sθ_2 = var_vec.sθ_2 + step*k.sθ_2
        self.var_vec1.z = var_vec.z + step


    def runge_var_vec(self):
        k1 = self.count_k_var_vec()
        self.runge_var_vec_prepare(H/2, k1)
        k2 = self.count_k_var_vec()
        self.runge_var_vec_prepare(H/2, k2)
        k3 = self.count_k_var_vec()
        self.runge_var_vec_prepare(0, k3)
        k4 = self.count_k_var_vec()

        self.var_vec.pθ_1 = self.var_vec.pθ_1 + H/6*(k1.pθ_1 + 2*k2.pθ_1 + 2*k3.pθ_1 + k4.pθ_1)
        self.var_vec.pθ_2 = self.var_vec.pθ_2 + H/6*(k1.pθ_2 + 2*k2.pθ_2 + 2*k3.pθ_2 + k4.pθ_2)
        self.var_vec.sθ_1 = self.var_vec.sθ_1 + H/6*(k1.sθ_1 + 2*k2.sθ_1 + 2*k3.sθ_1 + k4.sθ_1)
        self.var_vec.sθ_2 = self.var_vec.sθ_2 + H/6*(k1.sθ_2 + 2*k2.sθ_2 + 2*k3.sθ_2 + k4.sθ_2)
        self.var_vec.py_1 = self.var_vec.py_1 + H/6*(k1.py_1 + 2*k2.py_1 + 2*k3.py_1 + k4.py_1)
        self.var_vec.py_2 = self.var_vec.py_2 + H/6*(k1.py_2 + 2*k2.py_2 + 2*k3.py_2 + k4.py_2)
        self.var_vec.sy_1 = self.var_vec.sy_1 + H/6*(k1.sy_1 + 2*k2.sy_1 + 2*k3.sy_1 + k4.sy_1)
        self.var_vec.sy_2 = self.var_vec.sy_2 + H/6*(k1.sy_2 + 2*k2.sy_2 + 2*k3.sy_2 + k4.sy_2)
        self.var_vec.z = self.var_vec.z + H


    def count_k_var_vec(self):
        var_vec = self.var_vec1
        vec = self.vec
        k = var_vector()
        
        k.py_1 = var_vec.py_1
        k.py_2 = var_vec.py_2
        k.pθ_1 = var_vec.pθ_1
        k.pθ_2 = var_vec.pθ_2
        k.sy_1 = (var_vec.sy_1 + var_vec.py_1*(Da*np.exp((γ*vec.θ)/(γ + vec.θ))) - var_vec.pθ_1 * Da * (1-vec.y)*np.exp((γ*vec.θ)/(γ + vec.θ)) * γ*γ/(γ + vec.θ)/(γ + vec.θ))*Re_m 
        k.sy_2 = (var_vec.sy_2 + var_vec.py_2*(Da*np.exp((γ*vec.θ)/(γ + vec.θ))) - var_vec.pθ_2 * Da * (1-vec.y)*np.exp((γ*vec.θ)/(γ + vec.θ)) * γ*γ/(γ + vec.θ)/(γ + vec.θ))*Re_m
        k.sθ_1 = (var_vec.sθ_1 + var_vec.py_1*(B*Da*np.exp((γ*vec.θ)/(γ + vec.θ))) - var_vec.pθ_1 * (B*Da*(1-vec.y)*np.exp((γ*vec.θ)/(γ+vec.θ))*γ*γ/(γ+vec.θ)/(γ+vec.θ) - β))*Re_n
        k.sθ_2 = (var_vec.sθ_2 + var_vec.py_2*(B*Da*np.exp((γ*vec.θ)/(γ + vec.θ))) - var_vec.pθ_2 * (B*Da*(1-vec.y)*np.exp((γ*vec.θ)/(γ+vec.θ))*γ*γ/(γ+vec.θ)/(γ+vec.θ) - β))*Re_n
        k.z = var_vec.z

        return k


    def count_newton(self):
        var_vec = self.var_vec
        vec = self.vec

        self.W[0][0] = Re_m*var_vec.py_1 - var_vec.sy_1
        self.W[0][1] = Re_m*var_vec.py_2 - var_vec.sy_2
        self.W[1][0] = Re_n*var_vec.pθ_1 - var_vec.sθ_1
        self.W[1][1] = Re_n*var_vec.pθ_2 - var_vec.sθ_2

        self.res1.η_1 = self.res.η_1 - 1/(self.W[0][0]*self.W[1][1] - self.W[1][0]*self.W[0][1])*( self.W[1][1]*(Re_m*vec.y - vec.ζ) - self.W[0][1]*(Re_n*vec.θ - vec.φ))
        self.res1.η_2 = self.res.η_2 - 1/(self.W[0][0]*self.W[1][1] - self.W[1][0]*self.W[0][1])*(-self.W[1][0]*(Re_m*vec.y - vec.ζ) + self.W[0][0]*(Re_n*vec.θ - vec.φ))



if __name__ == "__main__":
    shoot = shoot()

    while((abs(shoot.res1.η_1 - shoot.res.η_1) > ε) or (abs(shoot.res1.η_2 - shoot.res.η_2) > ε)):
        shoot.roll_back()
        y, θ, z = [], [], []
        while(shoot.vec.z > 0):
            shoot.runge_var_vec()
            shoot.runge_vec()
            y.append(shoot.vec.y)
            θ.append(shoot.vec.θ)
            z.append(shoot.vec.z)
        
        shoot.count_newton()
        print("η1={}, η2={}, y(0)={}, y'(0)={}, θ(0)={}, θ'(0)={}".format(shoot.res1.η_1, shoot.res1.η_2, shoot.vec.y, shoot.vec.ζ, shoot.vec.θ, shoot.vec.φ))
    
    plt.plot(z, y)
    plt.title("y(z), Da = {}".format(Da))
    plt.show()
    plt.plot(z, θ)
    plt.title("θ(z), Da = {}".format(Da))
    plt.show()
