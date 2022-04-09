from library import gauss_jordan, lu_decomposition


# Define relevant matrices and vectors
A = [
    [1, -1, 4, 0, 2, 9],
    [0, 5, -2, 7, 8, 4],
    [1, 0, 5, 7, 3, -2],
    [6, -1, 2, 3, 0, 8],
    [-4, 2, 0, 5, -5, 3],
    [0, 7, -1, 5, 4, -2],
]
b = [19, 2, 13, -7, -9, 2]

# Solution
x_gj = gauss_jordan(A, b)
x_lu = lu_decomposition(A, b)

print("Solutions - \n")
print("Gauss-Jordan x = {}\n".format(x_gj))
print("LU-decomposition x = {}".format(x_lu))


# -----------------------* OUTPUT *-----------------------------------
# Solutions -

# Gauss-Jordan x = [-1.7618170439978567, 0.8962280338740136, 4.051931404116157, -1.6171308025395428, 2.041913538501914, 0.15183248715593495]

# LU-decomposition x = [-1.7618170439978567, 0.8962280338740136, 4.051931404116157, -1.6171308025395428, 2.041913538501914, 0.15183248715593495]
