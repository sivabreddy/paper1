# import numpy as np
# import random,math
#
#
# def algm(w):
#     w = np.asarray(w)
#     Xmin,Xmax = 1,5
#     N,M = len(w),len(w)
#     alpha = 1
#     r = random.random()
#     l1,l2,l3 = 5*math.exp(-2),100,1*math.exp(-2)
#     lb = np.min(w[0].flatten())
#     ub = np.max(w[0].flatten())
#     Tmax = 10
#     def generate(N,M):
#         data = []
#         for i in range(N):
#             tem = []
#             for j in range(M):
#                 tem.append(random.random())
#             data.append(tem)
#         return data
#
#     def bound(arr):
#         data = []
#         for i in range(len(arr)):
#             tem=[]
#             for j in range(len(arr[i])):
#                 if(arr[i][j]<0 or arr[i][j]>100):
#                     tem.append(random.uniform(lb,ub))
#                 else:
#                     tem.append(random.uniform(lb,ub))
#             data.append(tem)
#         return data
#
#     def fitness(soln):
#         Fit = []
#         for i in range(len(soln)):
#             F = 0
#             for j in range(len(soln[i])):
#                 hr = random.random()
#                 F += soln[i][j] + hr
#             Fit.append(F)
#         return Fit
#
#     X = generate(N,M)
#     Hj,Pij,Cj= l1*random.random(),l2*random.random(),l3*random.random()
#     F = random.uniform(-1,1)
#     Fit = fitness(X)
#     Fbest = np.max(Fit)
#     best = np.argmax(Fit)
#     Xbest = np.max(X[best])
#     t = 0
#     g = 1  # Absorption coefficient
#     E = random.sample(range(1, len(w)+1), len(X))
#     while(t<Tmax):
#         beta,epsilon,T_teta,K = 0.5,0.05,298.15,0.5
#         T = math.exp(-t / Tmax)  # Temperature
#         Hj = Hj * math.exp(-Cj * (1 / T) - (1 / T_teta))  # Henry's coefficient (eq.8)
#         S = K * Hj * Pij  # solubility (eq.9)
#         rr = np.sqrt(np.sum((X[0][0] - X[1][1]) ** 2))
#         beta0 = math.exp(-g * rr)
#         new_X = []
#         for i in range(len(X)):
#             tem = []
#             for j in range(len(X[i])):
#                 Gamma = beta * math.exp(-(Fbest + epsilon) / (Fit[i] + epsilon))
#                 ############ proposed updated equation ##############
#                 n = ((beta0*math.exp(-Gamma*r**2)*X[i][j]*alpha*E[i])*((F*r*Gamma)+((F*r*alpha)-1))+(F*r*(Gamma*Xbest+alpha*S*Xbest))*(1-beta0*math.exp(-Gamma*r**2)))/((F*r*Gamma)+(F*r*alpha)-beta0*math.exp(-Gamma*r**2))
#                 tem.append(n)
#             new_X.append(tem)
#
#         X = bound(new_X)
#
#         c1,c2 = 0.1,0.2
#         Nw = M*(random.uniform(c1,c2)+c1) # eq.11
#         G = Xmin + r* (Xmax-Xmin)
#         worst = round(G)
#
#         Fit = fitness(X)
#         Fbest = np.max(Fit)
#         best = np.argmax(Fit)
#         Xbest = np.max(X[best])
#         t += 1
#     return np.asarray(X[best])
#
#
#
"""
Hybrid Feature Guided Swarm Optimization (HFGSO)
-----------------------------------------------
Nature-inspired optimization algorithm combining:
- Henry's gas solubility optimization
- Particle swarm optimization
- Feature guidance mechanisms

Used to optimize neural network weights in the proposed model
"""

import numpy as np
import random, math

def algm(w):
    """
    HFGSO algorithm for optimizing neural network weights
    
    Args:
        w: List of network weight matrices to optimize
        
    Returns:
        List of optimized weight matrices with original shapes preserved
    """
    # Store original shapes to reconstruct later
    original_shapes = [weight.shape for weight in w]

    # Flatten each weight array and concatenate into a single vector
    flattened = []
    for weight in w:
        flattened.extend(weight.flatten())

    flattened = np.array(flattened)

    # Get min and max values for bounds
    lb = np.min(flattened)
    ub = np.max(flattened)

    # Set parameters for the algorithm
    Xmin, Xmax = 1, 5
    N, M = 10, len(flattened)  # Use fixed population size and actual dimension
    alpha = 1
    r = random.random()
    l1, l2, l3 = 5*math.exp(-2), 100, 1*math.exp(-2)
    Tmax = 10

    def generate(N, M):
        """
        Generates initial population of solutions
        within weight value bounds [lb, ub]
        """
        data = []
        for i in range(N):
            tem = []
            for j in range(M):
                tem.append(random.uniform(lb, ub))
            data.append(tem)
        return data

    def bound(arr):
        """
        Ensures solution values stay within bounds
        Replaces out-of-bounds values with random values in [lb, ub]
        """
        data = []
        for i in range(len(arr)):
            tem = []
            for j in range(len(arr[i])):
                if arr[i][j] < lb or arr[i][j] > ub:
                    tem.append(random.uniform(lb, ub))
                else:
                    tem.append(arr[i][j])
            data.append(tem)
        return data

    def fitness(soln):
        """
        Evaluates solution fitness
        Higher values indicate better solutions
        """
        Fit = []
        for i in range(len(soln)):
            F = 0
            for j in range(len(soln[i])):
                hr = random.random()
                F += soln[i][j] + hr
            Fit.append(F)
        return Fit

    X = generate(N, M)
    Hj, Pij, Cj = l1*random.random(), l2*random.random(), l3*random.random()
    F = random.uniform(-1, 1)
    Fit = fitness(X)
    Fbest = np.max(Fit)
    best = np.argmax(Fit)
    Xbest = np.max(X[best])
    t = 0
    g = 1  # Absorption coefficient
    E = random.sample(range(1, N+1), N)  # Changed to use N instead of len(w)

    while(t < Tmax):
        beta, epsilon, T_teta, K = 0.5, 0.05, 298.15, 0.5
        T = math.exp(-t / Tmax)  # Temperature
        Hj = Hj * math.exp(-Cj * (1 / T) - (1 / T_teta))  # Henry's coefficient (eq.8)
        S = K * Hj * Pij  # solubility (eq.9)

        # Calculate distance between first two points (if possible)
        if len(X) >= 2 and len(X[0]) >= 1 and len(X[1]) >= 1:
            rr = np.sqrt((X[0][0] - X[1][0]) ** 2)
        else:
            rr = 0.1  # Default value if not enough points

        beta0 = math.exp(-g * rr)
        new_X = []

        for i in range(len(X)):
            tem = []
            for j in range(len(X[i])):
                Gamma = beta * math.exp(-(Fbest + epsilon) / (Fit[i] + epsilon))
                ############ proposed updated equation ##############
                n = ((beta0*math.exp(-Gamma*r**2)*X[i][j]*alpha*E[i])*((F*r*Gamma)+((F*r*alpha)-1))+(F*r*(Gamma*Xbest+alpha*S*Xbest))*(1-beta0*math.exp(-Gamma*r**2)))/((F*r*Gamma)+(F*r*alpha)-beta0*math.exp(-Gamma*r**2))
                tem.append(n)
            new_X.append(tem)

        X = bound(new_X)

        c1, c2 = 0.1, 0.2
        Nw = M*(random.uniform(c1, c2)+c1)  # eq.11
        G = Xmin + r*(Xmax-Xmin)
        worst = round(G)

        Fit = fitness(X)
        Fbest = np.max(Fit)
        best = np.argmax(Fit)
        Xbest = np.max(X[best])
        t += 1

    # Get the best solution
    best_solution = np.array(X[best])

    # Reconstruct the original weight structure
    result = []
    index = 0
    for shape in original_shapes:
        size = np.prod(shape)
        weight_flat = best_solution[index:index+size]
        weight_reshaped = weight_flat[:size].reshape(shape)
        result.append(weight_reshaped)
        index += size

    return result