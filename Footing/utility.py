"""
@author: Pin ZHANG, 2026, Singapore
@Reference: A Comprehensive Investigation of Physics-Informed Learning in Forward and Inverse Analysis of Elastic and Elastoplastic Footing.
            Computers and Geotechnics, 181, 107110
"""

import numpy as np
import torch



def get_gauss_integration_points_2D(width, height, elements_x, elements_y):

    dx = width / elements_x
    dy = height / elements_y

    x = np.linspace(0, width, elements_x + 1)
    y = np.linspace(0, height, elements_y + 1)
    mesh_grid = np.meshgrid(x, y)

    gauss_points = [
        (-1 / np.sqrt(3), -1 / np.sqrt(3)),
        (1 / np.sqrt(3), -1 / np.sqrt(3)),
        (1 / np.sqrt(3), 1 / np.sqrt(3)),
        (-1 / np.sqrt(3), 1 / np.sqrt(3))
    ]

    global_integration_points = []
    for i in range(elements_x):
        for j in range(elements_y):
            element_integration_points = []
            for gp in gauss_points:
                xi, eta = gp
                x_global = mesh_grid[0][i, j] + (xi + 1) * dx / 2
                y_global = mesh_grid[1][i, j] + (eta + 1) * dy / 2
                element_integration_points.append((x_global, y_global))
            global_integration_points.append(element_integration_points)
    return global_integration_points





def gauss_training_data_2D(width, height, elements_x, elements_y):

    gauss_integration_points = get_gauss_integration_points_2D(width=width, height=height, elements_x=elements_x, elements_y=elements_y)

    data = []
    for element in gauss_integration_points:
        for i in range(4):
            data.append(element[i])

    x = torch.tensor([point[0] for point in data])[:, None]
    y = torch.tensor([point[1] for point in data])[:, None]
    X = torch.cat([x, y], dim=1)

    return X

# print(gauss_training_data_2D(width=10.0, height=10.0, elements_x=2, elements_y=2))


def get_gauss_integration_points_1D(integration_range, elements):

    gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]

    element_size = (integration_range[1] - integration_range[0]) / elements

    global_integration_points = []
    for i in range(elements):
        element_integration_points = []
        for gp in gauss_points:
            x_global = integration_range[0] + (i + (gp + 1) / 2) * element_size
            element_integration_points.append(x_global)
        global_integration_points.extend(element_integration_points)
    return global_integration_points





def gauss_training_data_1D(integration_range, elements, y_corrd):

    gauss_integration_points = get_gauss_integration_points_1D(integration_range=integration_range, elements=elements)
    x = torch.tensor(gauss_integration_points)[:, None]
    y = y_corrd * torch.ones_like(x)
    y = y.double()

    X = torch.cat([x, y], dim=1)
    return X

def gauss_training_data_1D_yaxis(integration_range, elements, x_corrd):

    gauss_integration_points = get_gauss_integration_points_1D(integration_range=integration_range, elements=elements)
    y = torch.tensor(gauss_integration_points)[:, None]
    x = x_corrd * torch.ones_like(y)
    x = x.double()

    X = torch.cat([x, y], dim=1)
    return X








