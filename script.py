import sympy as sp


# Helper: Define an elementwise (Hadamard) product for Sympy matrices.
def hadamard_product(A, B):
    m, n = A.shape
    return sp.Matrix(m, n, lambda i, j: A[i, j] * B[i, j])


# ------------------ Activation Functions ------------------ #

def sigmoid(x):
    # x is assumed to be a sp.Matrix; apply the function elementwise.
    return x.applyfunc(lambda z: 1 / (1 + sp.exp(-z)))


def sigmoid_derivative(a):
    # Assumes 'a' is the output of the sigmoid function.
    return a.applyfunc(lambda z: z * (1 - z))


def tanh(x):
    return x.applyfunc(sp.tanh)


def tanh_derivative(a):
    return a.applyfunc(lambda z: 1 - z ** 2)


def softmax(x):
    # x is assumed to be a column vector (n x 1)
    # Convert to a list of elements for processing.
    x_list = list(x)
    max_val = max(x_list)
    exp_vals = [sp.exp(xi - max_val) for xi in x_list]
    sum_exp = sum(exp_vals)
    result = [val / sum_exp for val in exp_vals]
    return sp.Matrix(result)


def softmax_derivative(a):
    # Simplified derivative applied elementwise (not the full Jacobian).
    return a.applyfunc(lambda z: z * (1 - z))


def custom_activation(x):
    return x  # Identity (customize as needed)


def custom_activation_derivative(a):
    return sp.ones(a.shape[0], a.shape[1])  # Derivative of identity is 1


# Dictionary of activation functions.
# Ensure you type the names exactly: 'sigmoïde', 'tanh', 'softmax', 'personnalisée'
activation_functions = {
    'sigmoide': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'softmax': (softmax, softmax_derivative),
    'personnalisée': (custom_activation, custom_activation_derivative)
}


# ------------------ User Input and Initialization ------------------ #

def get_user_input():
    num_layers = int(input("Entrez le nombre total de couches (y compris entrée, cachées et sortie) : "))
    layer_sizes = []
    activations = []  # activation function names for hidden/output layers

    for i in range(num_layers):
        size = int(input(f"Entrez le nombre de neurones dans la couche {i + 1} : "))
        layer_sizes.append(size)
        if i > 0:  # pas de fonction d'activation pour la couche d'entrée
            act = input(
                f"Entrez la fonction d'activation pour la couche {i + 1} (sigmoïde/tanh/softmax/personnalisée) : ").strip()
            activations.append(act)

    weights = []
    biases = []
    # For each connection between layers, get weights and biases.
    for i in range(num_layers - 1):
        print(f"\nEntrez les poids pour les connexions de la Couche {i + 1} à la Couche {i + 2} :")
        w = []
        for j in range(layer_sizes[i + 1]):
            row = list(map(sp.Float, input(
                f"Poids pour le neurone {j + 1} dans la Couche {i + 2} (séparés par un espace) : ").split()))
            w.append(row)
        weights.append(sp.Matrix(w))  # Each weight matrix: (neurones_couche_suivante x neurones_couche_actuelle)

        print(f"Entrez les biais pour la Couche {i + 2} :")
        b = list(map(sp.Float, input(f"Biais pour la Couche {i + 2} (séparés par un espace) : ").split()))
        # sp.Matrix(b) returns a column vector if b is a 1D list.
        biases.append(sp.Matrix(b))

    print("\nEntrez les valeurs d'entrée :")
    X = list(map(sp.Float, input("Valeurs d'entrée (séparés par un espace) : ").split()))
    X = sp.Matrix(X)

    print("Entrez les valeurs de sortie attendues :")
    Y = list(map(sp.Float, input("Valeurs de sortie attendues (séparés par un espace) : ").split()))
    Y = sp.Matrix(Y)

    return X, Y, layer_sizes, activations, weights, biases


# ------------------ Forward Propagation ------------------ #

def forward_propagation(X, weights, biases, activations):
    a = X
    activations_cache = [a]  # store activations per layer
    i_cache = []  # store the weighted inputs per layer
    print("\nPhase 1 : Activation\n")
    for i in range(len(weights)):
        w = weights[i]
        b = biases[i]
        act = activations[i]  # activation function for the (i+1)-th layer
        i_val = w * a + b  # weighted input (matrix multiplication)
        a = activation_functions[act][0](i_val)
        i_cache.append(i_val)
        activations_cache.append(a)
        print(f"Couche {i + 1} (i) :\n{i_val}")
        print(f"Couche {i + 1} (a) :\n{a}")
    return activations_cache, i_cache


# ------------------ Backward Propagation ------------------ #

def backward_propagation(Y, weights, activations_cache, i_cache, activations):
    # Initialize lists to hold deltas and gradients.
    deltas = [None] * len(weights)
    gradients_W = [None] * len(weights)
    gradients_B = [None] * len(weights)

    print("\nPhase 2 : Signal d'erreur\n")
    L = len(weights) - 1  # Index of the output layer connection

    # Compute delta for output layer.
    delta_L = activations_cache[-1] - Y  # Corrected sign here
    if activations[L] != 'softmax':
        # Elementwise multiplication with the derivative of the activation function.
        delta_L = hadamard_product(delta_L, activation_functions[activations[L]][1](activations_cache[-1]))
    deltas[L] = delta_L
    print(f"Delta Couche {L + 1} :\n{delta_L}")

    # Backpropagate through hidden layers.
    for l in range(L - 1, -1, -1):
        # Compute error signal from the next layer.
        temp = weights[l + 1].T * deltas[l + 1]
        deriv = activation_functions[activations[l]][1](activations_cache[l + 1])
        deltas[l] = hadamard_product(temp, deriv)
        print(f"Delta Couche {l + 1} :\n{deltas[l]}")

    print("\nPhase 3 : Facteur de correction\n")
    # Compute gradients for weights and biases.
    for l in range(len(weights)):
        gradients_W[l] = deltas[l] * activations_cache[l].T
        gradients_B[l] = deltas[l]  # For biases, the delta itself is used.
        print(f"ΔW Couche {l + 1} (affiché en format (neurones_prev x neurones_next)) :\n{-gradients_W[l].T}")
    return gradients_W, gradients_B


# ------------------ Weight Update ------------------ #

def update_weights(weights, biases, gradients_W, gradients_B, learning_rate=sp.Float(0.1)):
    print("\nPhase 4 : Mise à jour\n")
    for l in range(len(weights)):
        weights[l] = weights[l] - learning_rate * gradients_W[l]
        biases[l] = biases[l] - learning_rate * gradients_B[l]
        print(f"W{l + 1} (affiché en format (neurones_prev x neurones_next)) :\n{weights[l].T}")
        print(f"B{l + 1} :\n{biases[l]}")
    return weights, biases


# ------------------ Main ------------------ #

if __name__ == "__main__":
    X, Y, layer_sizes, activations, weights, biases = get_user_input()
    print("\nValeurs initiales :")
    print("X :\n", X)
    print("Y :\n", Y)
    print("Layer sizes :", layer_sizes)
    print("Activations :", activations)
    print("Weights :", weights)
    print("Biases :", biases)

    activations_cache, i_cache = forward_propagation(X, weights, biases, activations)
    gradients_W, gradients_B = backward_propagation(Y, weights, activations_cache, i_cache, activations)
    updated_weights, updated_biases = update_weights(weights, biases, gradients_W, gradients_B,
                                                     learning_rate=sp.Float(0.1))