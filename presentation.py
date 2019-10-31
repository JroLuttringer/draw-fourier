from PIL import Image
from pylab import *

import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
from numpy import interp

from math import tau
from scipy.integrate import quad

def travel_salesman(image_path):
    og_img = Image.open(image_path)
    bw_img = og_img.convert('1', dither=Image.NONE)

    # convert to array of int
    bw_img_array = np.array(bw_img, dtype=np.int)

    # get black_indices where the pixel is black
    black_indices = np.argwhere(bw_img_array == 0)

    chosen_black_indices = black_indices[np.random.choice(black_indices.shape[0], replace=False, size = 3000)]

    plt.figure(figsize=(6, 8), dpi=100)
    plt.scatter([x[1] for x in chosen_black_indices],
                [x[0] for x in chosen_black_indices],
                color='black', s=1)
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.savefig("tmp_first.png", bbox_inches='tight')


    # distance between each pixels
    distances = pdist(chosen_black_indices)
    #X * X matrix containing the distance between each and every pixels
    distance_matrix = squareform(distances)

    optimized_path = solve_tsp(distance_matrix)

    optimized_path_points = [chosen_black_indices[x] for x in optimized_path]
    plt.figure(figsize=(8, 10), dpi=100,frameon=False)
    plt.plot([x[1] for x in optimized_path_points],
             [x[0] for x in optimized_path_points],
             color='black', lw=1)
    plt.xlim(0, 600)
    plt.ylim(0, 800)
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig("tmp.png", bbox_inches='tight')

def create_close_loop(image_name, level=[200]):
    # Prepare Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))


    # read image to array, then get image border with contour
    im = array(Image.open(image_name).convert('L'))
    #contour_plot = np.argwhere(im == 0)
    contour_plot = ax[0].contour(im, levels=level, colors='black', origin='image')

    # Get Contour Path and create lookup-table
    contour_path = contour_plot.collections[0].get_paths()[0]
    x_table, y_table = contour_path.vertices[:, 0], contour_path.vertices[:, 1]
    time_table = np.linspace(0, tau, len(x_table))

    # Simple method to center the image
    x_table = x_table - min(x_table)
    y_table = y_table - min(y_table)
    x_table = x_table - max(x_table) / 2
    y_table = y_table - max(y_table) / 2

    # Visualization
    ax[1].plot(x_table, y_table, 'k-')

    return time_table, x_table, y_table

def f(t, time_table, x_table, y_table):
    return interp(t, time_table, x_table) + 1j*interp(t, time_table, y_table)

def DFT(t, coef_list, order=100):
    kernel = np.array([np.exp(-n*1j*t) for n in range(-order, order+1)])
    series = np.sum( (coef_list[:,0]+1j*coef_list[:,1]) * kernel[:])
    return np.real(series), np.imag(series)

def coef_list(time_table, x_table, y_table, order=1):
    coef_list = []
    for n in range(-order, order+1):
        real_coef = quad(lambda t: np.real(f(t, time_table, x_table, y_table) * np.exp(-n*1j*t)), 0, tau, limit=100, full_output=1)[0]/tau
        imag_coef = quad(lambda t: np.imag(f(t, time_table, x_table, y_table) * np.exp(-n*1j*t)), 0, tau, limit=100, full_output=1)[0]/tau
        coef_list.append([real_coef, imag_coef])
    return np.array(coef_list)

def visualize(x_DFT, y_DFT, coef, order, space, fig_lim):
    fig, ax = plt.subplots()
    plt.axis('off')
    plt.gca().invert_yaxis()
    lim = max(fig_lim)
    ax.set_xlim([-lim-300, lim])
    ax.set_ylim([-lim, lim])
    ax.set_aspect('equal')

    # Initialize
    line = plt.plot([], [], 'k-', linewidth=2)[0]
    line2 = plt.plot([], [], 'k-', linewidth=0.5)[0]
    radius = [plt.plot([], [], 'r-', linewidth=0.5, marker='o', markersize=1)[0] for _ in range(2 * order + 1)]
    circles = [plt.plot([], [], 'r-', linewidth=0.5, color="blue")[0] for _ in range(2 * order + 1)]

    def update_c(c, t):
        new_c = []
        for i, j in enumerate(range(-order, order + 1)):
            dtheta = -j * t
            ct, st = np.cos(dtheta), np.sin(dtheta)
            v = [ct * c[i][0] - st * c[i][1], st * c[i][0] + ct * c[i][1]]
            new_c.append(v)
        return np.array(new_c)

    def sort_velocity(order):
        idx = []
        for i in range(1,order+1):
            idx.extend([order+i, order-i])
        return idx


    def animate(i):
        # animate lines
        line.set_data(x_DFT[:i], y_DFT[:i])

        # animate circles
        r = [np.linalg.norm(coef[j]) for j in range(len(coef))]
        pos = coef[order]
        c = update_c(coef, i / len(space) * tau)
        idx = sort_velocity(order)
        x = None
        y = None
        for j, rad, circle in zip(idx,radius,circles):
            new_pos = pos + c[j]
            rad.set_data([pos[0]-300, new_pos[0]-300], [pos[1], new_pos[1]])
            theta = np.linspace(0, tau, 50)
            x, y = r[j] * np.cos(theta) + pos[0]-300, r[j] * np.sin(theta) + pos[1]
            circle.set_data(x, y)
            pos = new_pos

        line2.set_data([pos[0]-300, x_DFT[i]],[pos[1],y_DFT[i]])

    # Animation
    ani = animation.FuncAnimation(fig, animate, frames=len(space), interval=5)
    return ani
