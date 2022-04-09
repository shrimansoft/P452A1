from library import lu_decomposition, jacobi, conjgrad, gauss_seidel
from library import matmul, mat_print
import matplotlib.pyplot as plt

# Define relevant matrices and vectors
A = [
    [2, -3, 0, 0, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 0, -1],
    [0, 0, 0, 2, -3, 0],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4],
]
b = [-5 / 3, 2 / 3, 3, -4 / 3, -1 / 3, 5 / 3]

# Solution using LU and Jacobi
x_lu = lu_decomposition(A, b)
xj = jacobi(A, b, 1e-4)[0]
xgs = gauss_seidel(A, b, 1e-4)[0]

print("Solutions- \n")
print("LU Decomposition: x = {}".format(x_lu))
print("Jacobi: x = {}".format(xj))
print("Gauss-Seidel: x = {}".format(xgs))

# Calculating inverse of A using Jacobi, Gauss-Seidel and Conjugate Gradient
b1 = [1, 0, 0, 0, 0, 0]
b2 = [0, 1, 0, 0, 0, 0]
b3 = [0, 0, 1, 0, 0, 0]
b4 = [0, 0, 0, 1, 0, 0]
b5 = [0, 0, 0, 0, 1, 0]
b6 = [0, 0, 0, 0, 0, 1]

# Jacobi
jx1, iterj, resigj = jacobi(A, b1, 1e-4)
jx2 = jacobi(A, b2, 1e-4)[0]
jx3 = jacobi(A, b3, 1e-4)[0]
jx4 = jacobi(A, b4, 1e-4)[0]
jx5 = jacobi(A, b5, 1e-4)[0]
jx6 = jacobi(A, b6, 1e-4)[0]

tempMat = [jx1, jx2, jx3, jx4, jx5, jx6]
matInvj = list(zip(*tempMat))  # Inverse of A using Jacobi
print("\nInverse of A using Jacobi method:")
mat_print(matInvj)

# Gauss-Seidel
gsx1, itergs, resigs = gauss_seidel(A, b1, 1e-4)
gsx2 = gauss_seidel(A, b2, 1e-4)[0]
gsx3 = gauss_seidel(A, b3, 1e-4)[0]
gsx4 = gauss_seidel(A, b4, 1e-4)[0]
gsx5 = gauss_seidel(A, b5, 1e-4)[0]
gsx6 = gauss_seidel(A, b6, 1e-4)[0]

tempMat = [gsx1, gsx2, gsx3, gsx4, gsx5, gsx6]
matInvgs = list(zip(*tempMat))  # Inverse of A using Gauss-Seidel
print("\nInverse of A using Gauss-Seidel method:")
mat_print(matInvgs)

# Comparing convergence rates of Jacobi and Gauss-Seidel
plt.plot(iterj, resigj, label="Jacobi")
plt.plot(itergs, resigs, label="Gauss-Seidel")
plt.xlabel("Iterations")
plt.ylabel("Residue")
plt.title("Convergence rate of various methods")
plt.legend()
plt.show()

# ------------------------------* OUTPUT *-----------------------------------------

# Solutions-

# LU Decomposition: x = [-0.3333333333333335, 0.33333333333333326, 0.9999999999999999, -0.6666666666666665, 5.401084984662924e-17, 0.6666666666666667]
# Jacobi: x = [-0.3331943690356095, 0.3333953635563268, 1.000022329842889, -0.6665529565984641, 5.075735993963215e-05, 0.6666849384673739]
# Gauss-Seidel: x = [-0.3334588615972566, 0.33327730062340244, 0.9999798291615298, -0.6667693824547573, -4.584978549287966e-05, 0.6666501615106759]

# Inverse of A using Jacobi method:
# (0.9351929274269596, 0.8702365333861066, 0.2598666464795029, 0.20791546310548104, 0.4156816047431495, 0.16895246192380803)
# (0.2901004226657191, 0.5801341919241123, 0.17321658907233445, 0.13858315664924734, 0.2770996598911687, 0.11260825478849718)
# (0.08660065336867459, 0.1731773126254245, 0.3203666291311983, 0.0562968618805311, 0.1125697296491375, 0.1082445985240106)
# (0.20789693987395835, 0.4156716947368404, 0.16893458708283318, 0.9351657909702235, 0.8702093969293704, 0.25983951002276673)
# (0.13857488833716416, 0.27709523630963456, 0.11260027590190659, 0.29008830962308696, 0.58012207888148, 0.1732044760297023)
# (0.05629388542645969, 0.11256813723376477, 0.10824172625827622, 0.08659629287590452, 0.17317295213265438, 0.3203622686384282)

# Inverse of A using Gauss-Seidel method:
# (0.9349588126322961, 0.8700347176927217, 0.2596261485714596, 0.20766251629577676, 0.41546813044124453, 0.16869171435560132)
# (0.28999591961636617, 0.5800441063938442, 0.17310923675922424, 0.1384702474535563, 0.27700437024648294, 0.11249186352716885)
# (0.08656303401866457, 0.173144883341127, 0.3203279840938959, 0.0562562164582921, 0.11253542697066368, 0.10820269961635934)
# (0.2077053711803345, 0.4155065553697246, 0.16873779529075325, 0.9349588126322961, 0.8700347176927212, 0.2596261485714596)
# (0.1384893768137482, 0.27702152218355813, 0.11251243291658419, 0.2899959196163661, 0.5800441063938441, 0.17310923675922424)
# (0.056263102708103196, 0.11254160138117128, 0.10821010425262002, 0.08656303401866455, 0.17314488334112693, 0.3203279840938959)
