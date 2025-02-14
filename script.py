import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(a):
    return a * (1 - a)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(a):
    return 1 - np.square(a)


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Pour la stabilité numérique
    return exp_x / exp_x.sum(axis=0, keepdims=True)


def softmax_derivative(a):
    return a * (1 - a)


def custom_activation(x):
    return x  # Modifier ici pour une fonction personnalisée


def custom_activation_derivative(a):
    return np.ones_like(a)  # Modifier ici pour la dérivée de la fonction personnalisée


activation_functions = {
    'sigmoide': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'softmax': (softmax, softmax_derivative),
    'personnalisée': (custom_activation, custom_activation_derivative)
}


def get_user_input():
    num_layers = int(input("Entrez le nombre total de couches (y compris entrée, cachées et sortie) : "))
    layer_sizes = []
    activations = []

    for i in range(num_layers):
        size = int(input(f"Entrez le nombre de neurones dans la couche {i + 1} : "))
        layer_sizes.append(size)
        if i > 0:  # Pas de fonction d'activation pour la couche d'entrée
            act = input(
                f"Entrez la fonction d'activation pour la couche {i + 1} (sigmoïde/tanh/softmax/personnalisée) : ")
            activations.append(act)

    weights = []
    biases = []
    for i in range(num_layers - 1):
        print(f"Entrez les poids pour les connexions de la Couche {i + 1} à la Couche {i + 2} :")
        w = []
        for j in range(layer_sizes[i + 1]):
            row = list(map(float, input(
                f"Poids pour le neurone {j + 1} dans la Couche {i + 2} (séparés par un espace) : ").split()))
            w.append(row)
        weights.append(np.array(w))

        print(f"Entrez les biais pour la Couche {i + 2} :")
        b = list(map(float, input(f"Biais pour la Couche {i + 2} (séparés par un espace) : ").split()))
        biases.append(np.array(b).reshape(-1, 1))

    print("Entrez les valeurs d'entrée :")
    X = list(map(float, input(f"Valeurs d'entrée (séparés par un espace) : ").split()))
    X = np.array(X).reshape(-1, 1)

    print("Entrez les valeurs de sortie attendues :")
    Y = list(map(float, input(f"Valeurs de sortie attendues (séparés par un espace) : ").split()))
    Y = np.array(Y).reshape(-1, 1)

    return X, Y, layer_sizes, activations, weights, biases


def forward_propagation(X, weights, biases, activations):
    a = X
    activations_cache = [a]
    i_cache = []
    print("\nPhase 1 : Activation\n")
    for i, (w, b, act) in enumerate(zip(weights, biases, activations)):
        i_val = np.dot(w, a) + b
        a = activation_functions[act][0](i_val)
        i_cache.append(i_val)
        activations_cache.append(a)
        print(f"Couche {i + 1} (i) :\n{i_val}")
        print(f"Couche {i + 1} (a) :\n{a}")
    return activations_cache, i_cache


# def backward_propagation(Y, weights, activations_cache, i_cache, activations):
#     deltas = [None] * len(weights)
#     gradients_W = [None] * len(weights)
#     gradients_B = [None] * len(weights)
#     print("\nPhase 2 : Signal d'erreur\n")
#     L = len(weights) - 1
#     # Calcul du delta pour la couche de sortie
#     delta_L = activations_cache[-1] - Y
#     if activations[L] != 'softmax':
#         delta_L *= activation_functions[activations[L]][1](activations_cache[-1])
#     deltas[L] = delta_L
#     print(f"Delta Couche {L + 1} :\n{delta_L}")
#
#     # Propagation de l'erreur vers les couches précédentes
#     for l in range(L - 1, -1, -1):
#         deltas[l] = (weights[l + 1].T @ deltas[l + 1]) * activation_functions[activations[l]][1](
#             activations_cache[l + 1])
#         print(f"Delta Couche {l + 1} :\n{deltas[l]}")
#
#     print("\nPhase 3 : Facteur de correction\n")
#     for l in range(len(weights)):
#         # Note: gradients_W[l] a la même forme que weights[l] (i.e. (neurons_next, neurons_prev))
#         # Pour l'affichage, nous transposons afin d'avoir (neurons_prev, neurons_next)
#         gradients_W[l] = deltas[l] @ activations_cache[l].T
#         gradients_B[l] = np.sum(deltas[l], axis=1, keepdims=True)
#         print(f"ΔW Couche {l + 1} (affiché en format (neurones_prev x neurones_next)) :\n{gradients_W[l].T}")
#     return gradients_W, gradients_B
def backward_propagation(Y, weights, activations_cache, i_cache, activations):
    deltas = [None] * len(weights)
    gradients_W = [None] * len(weights)
    gradients_B = [None] * len(weights)
    print("\nPhase 2 : Signal d'erreur\n")
    L = len(weights) - 1
    # Change here: compute delta as (Y - a) instead of (a - Y)
    delta_L = Y - activations_cache[-1]
    if activations[L] != 'softmax':
        delta_L *= activation_functions[activations[L]][1](activations_cache[-1])
    deltas[L] = delta_L
    print(f"Delta Couche {L + 1} :\n{delta_L}")

    for l in range(L - 1, -1, -1):
        deltas[l] = (weights[l + 1].T @ deltas[l + 1]) * activation_functions[activations[l]][1](activations_cache[l + 1])
        print(f"Delta Couche {l + 1} :\n{deltas[l]}")

    print("\nPhase 3 : Facteur de correction\n")
    for l in range(len(weights)):
        gradients_W[l] = deltas[l] @ activations_cache[l].T
        gradients_B[l] = np.sum(deltas[l], axis=1, keepdims=True)
        print(f"ΔW Couche {l + 1} (affiché en format (neurones_prev x neurones_next)) :\n{gradients_W[l].T}")
    return gradients_W, gradients_B


def update_weights(weights, biases, gradients_W, gradients_B, learning_rate=0.1):
    print("\nPhase 4 : Mise à jour\n")
    for l in range(len(weights)):
        weights[l] -= learning_rate * gradients_W[l]
        biases[l] -= learning_rate * gradients_B[l]
        # Pour l'affichage, transposons weights[l] pour avoir une matrice de taille (neurones_prev x neurones_next)
        print(f"W{l + 1} (affiché en format (neurones_prev x neurones_next)) :\n{weights[l].T}")
        print(f"B{l + 1} :\n{biases[l]}")
    return weights, biases


if __name__ == "__main__":
    X, Y, layer_sizes, activations, weights, biases = get_user_input()
    activations_cache, i_cache = forward_propagation(X, weights, biases, activations)
    gradients_W, gradients_B = backward_propagation(Y, weights, activations_cache, i_cache, activations)
    updated_weights, updated_biases = update_weights(weights, biases, gradients_W, gradients_B, learning_rate=0.1)
