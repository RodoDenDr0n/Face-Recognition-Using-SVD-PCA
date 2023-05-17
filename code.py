import numpy as np
import pandas as pd

import zipfile
from PIL import Image
import matplotlib.pyplot as plt

import ipywidgets as widget
from ipywidgets import interact, fixed

from power_iteration_svd import power_iter_svd, reduce_svd

def read_data(path):
    image_shape = (192, 168)
    X = np.empty((image_shape[0] * image_shape[1], 0))
    people = []
    with zipfile.ZipFile(path, 'r') as zip_file:
        target_dir = 'CroppedYale/'
        items = [name for name in zip_file.namelist() if name.startswith(target_dir)]

        directories = [name[len(target_dir):].split('/')[0] for name in items if name.endswith('/')]
        directories.sort()

        for i in range(len(directories)):
            print(f"DIRECTORY {i+1}")
            directory = directories[i]
            files = [name for name in items if name.startswith(target_dir + directory + '/') and not name.endswith('/')]
            first = True
            for file in files:
                with zip_file.open(file) as image_file:
                    image = Image.open(image_file)
                    image = np.array(image)
                    if first:
                        people.append(image)
                    first = False
                    image = image.reshape(-1, 1)
                    X = np.hstack((X, image))
    return X, people


def show_eigenfaces(U, average_face, m=192, n=168):
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))

    axs[0].imshow(average_face.reshape(m, n), cmap="gray")
    axs[0].set_title("Average Face")
    axs[0].axis("off")

    axs[1].imshow(U[:, 0].reshape(m, n), cmap="gray")
    axs[1].set_title("Eigenface 1")
    axs[1].axis("off")

    axs[2].imshow(U[:, 1].reshape(m, n), cmap="gray")
    axs[2].set_title("Eigenface 2")
    axs[2].axis("off")

    plt.show()


def interactive_k_rank(X, U, S, m=192, n=168):
    test_face = np.array(Image.open("foreign_face.jpg"))
    reshaped_test_face = np.reshape(test_face, (m * n, 1))

    test_face_demeaned = reshaped_test_face - average_face

    @interact(k=widget.IntSlider(min=1, max=min(m * n, len(S)), value=100), image=fixed(test_face))
    def recover_face(k, image):
        recovered_face = average_face + U[:, :k] @ (U[:, :k].T @ test_face_demeaned)

        fig, axs = plt.subplots(2, 2, figsize=(10, 5))

        axs[0][0].imshow(test_face, cmap="gray")
        axs[0][0].set_title(f"Original Image")
        axs[0][0].axis("off")

        axs[0][1].imshow(recovered_face.reshape(m, n), cmap="gray")
        axs[0][1].set_title(f"{k}-Rank Approximation")
        axs[0][1].axis("off")

        # Plot singular values in logarithmic scale
        axs[1][0].semilogy(S, 'k')
        axs[1][0].axvline(x=k, color='r', linestyle='--', linewidth=0.5)
        axs[1][0].set_title('Singular Values (log scale)')
        axs[1][0].set_xlim([0, X.shape[1]])

        # Plot cumulative sum of singular values deivided by the total sum
        axs[1][1].plot(np.cumsum(S) / np.sum(S), 'k')
        axs[1][1].axvline(x=k, color='r', linestyle='--', linewidth=0.5)
        axs[1][1].set_title('Cumulative Sum of Singular Values')
        axs[1][1].set_xlim([0, X.shape[1]])
        axs[1][1].set_ylim([0, 1])

        plt.show()


def PCA_scatter_plot(X, U):
    person1 = 2
    person2 = 7
    k = 64

    person1_faces = X[:, (person1 - 1) * k:person1 * k]
    person2_faces = X[:, (person2 - 1) * k:person2 * k]

    ones = np.ones((1, 64))
    X_mean = average_face @ ones

    person1_demeaned = person1_faces - X_mean
    person2_demeaned = person2_faces - X_mean

    # Choose PCA_modes
    PCA_modes = [5, 6]

    # Projecting person 1 on PCA coordinates
    PCA_coordinates_person1 = U[:, PCA_modes - np.ones_like(PCA_modes)].T @ person1_demeaned

    # Projecting person 2 on PCA coordinates
    PCA_coordinates_person2 = U[:, PCA_modes - np.ones_like(PCA_modes)].T @ person2_demeaned

    plt.plot(PCA_coordinates_person1[0, :], PCA_coordinates_person1[1, :], "d", color="k", label="Person 2")
    plt.plot(PCA_coordinates_person2[0, :], PCA_coordinates_person2[1, :], "^", color="r", label="Person 7")

    plt.legend()
    plt.show()


def train(X):
    # Traing set
    upd_X = []
    n = 1  # person number
    k = 64
    while (n - 1) * k < X.shape[1]:
        i = (n - 1) * k
        person = X[:, i:int(i + k / 2)]
        upd_X.append(person)
        n += 1
    upd_X = np.hstack(upd_X)

    # Computing SVD
    m, n = (192, 168)
    average_face = np.mean(X, axis=1).reshape(m * n, 1)
    ones = np.ones((1, upd_X.shape[1]))
    X_mean = average_face @ ones

    A = upd_X - X_mean
    U, S, V_T = np.linalg.svd(A, full_matrices=False)

    U = U[:32256, :1216]
    S = np.diag(1216)
    V_T = V_T[:1216, :1216]
    return upd_X, U, S, V_T


def get_PCA_coordinates(upd_X, X, U, num, PCA_modes):
    k = 32
    people_PCA = []
    ones = np.ones((1, k))
    X_mean_person = average_face @ ones

    # Calculate PCA coordinates for people
    for i in range(1, int(upd_X.shape[1] / k + 1)):
        person_faces = upd_X[:, (i - 1) * k:i * k]
        demeaned_faces = person_faces - X_mean_person
        PCA_coordinates = U[:, PCA_modes - np.ones_like(PCA_modes)].T @ demeaned_faces
        people_PCA.append(PCA_coordinates)

    # Chose the face to recognize and calculate its PCA coordinates
    test_person = X[:, (num - 1) * 64 + k]  # choose person not present in the trained set
    demeaned_test_person = test_person - X_mean_person[:, 0]
    PCA_coordinates_test_person = U[:, PCA_modes - np.ones_like(PCA_modes)].T @ demeaned_test_person

    return PCA_coordinates_test_person, people_PCA


def find_closest(num, PCA_coordinates_test_person, people_PCA):
    # Calculate distances
    distances = []
    for PCA_coordinates_person in people_PCA:
        for i in range(PCA_coordinates_person.shape[1]):
            distance = np.linalg.norm(PCA_coordinates_test_person - PCA_coordinates_person[:, i])
            distances.append(distance)

    # Find the closest match
    closest_match = np.argmin(distances)
    closest_person = int(np.floor(closest_match / 32) + 1)

    return num, closest_person


def test_precision(upd_X, X, U):
    PCA_modes = [9, 10, 11, 19, 20, 21, 49, 50, 51, 99, 100, 101, 199, 200, 201, 299, 300]
    total_faces = int(upd_X.shape[1]/32)

    correctly_recognised = 0
    for face in range(total_faces):
        num = face + 1
        PCA_coordinates_test_person, people_PCA = get_PCA_coordinates(upd_X, X, U, num, PCA_modes)
        real, recognised = find_closest(num, PCA_coordinates_test_person, people_PCA)
        if real == recognised:
            correctly_recognised += 1

    correct_percentage = np.round(correctly_recognised / total_faces * 100)
    index = ["total faces", "correctly recognised", "percentage of all"]
    summary = pd.DataFrame([int(total_faces), int(correctly_recognised), f"{int(correct_percentage)}%"], index=index)

    return summary


if __name__ == '__main__':
    # Read data
    path = "yalefaces_cropped.zip"
    X, _ = read_data(path)

    # Compute average face and subtract average
    m, n = (192, 168)
    average_face = np.mean(X, axis=1).reshape(m*n, 1)
    ones = np.ones((1, X.shape[1]))
    X_mean = average_face @ ones

    A = X - X_mean
    U, S, V_T = power_iter_svd(A)
    U = U[:32256, :2432]
    S = np.diad(S[:2432])
    V_T = V_T[:2432, :2432]

    # Show eigen faces
    show_eigenfaces(U, average_face)

    # Interactive k-rank approximation
    interactive_k_rank(X, U, S)

    # PCA scatter plot
    PCA_scatter_plot(X, U)

    # Demonstration of recognition
    upd_X, U, S, V_T = train(X)
    num = 2
    PCA_modes = [9, 10]
    PCA_coordinates_test_person, people_PCA = get_PCA_coordinates(upd_X, X, U, num, PCA_modes)
    find_closest(PCA_coordinates_test_person, people_PCA)

    # Test precision of the classifier
    test_precision(upd_X, X, U)
