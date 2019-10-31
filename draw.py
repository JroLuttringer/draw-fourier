import urllib.request
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from tsp_solver.greedy_numpy import solve_tsp
from presentation import *
import sys
import argparse




parser = argparse.ArgumentParser()
parser.add_argument('--in_file', required=True)
parser.add_argument('--out_file', required=True)
parser.add_argument('--lsp', action='store_true', default=False)
args =parser.parse_args()

image_path = args.in_file
out_file = args.out_file


if args.lsp:
    og_img = Image.open(image_path)
    bw_img = og_img.convert('1', dither=Image.NONE)

    # convert to array of int
    bw_img_array = np.array(bw_img, dtype=np.int)

    # get black_indices where the pixel is black
    black_indices = np.argwhere(bw_img_array == 0)

    chosen_black_indices = black_indices[np.random.choice(black_indices.shape[0], replace=False, size = 500)]

    plt.figure(figsize=(6, 8), dpi=100)
    plt.scatter([x[1] for x in chosen_black_indices],
                [x[0] for x in chosen_black_indices],
                color='black', s=1)
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.savefig("tmp_first.png", bbox_inches='tight')


    ## distance between each pixels
    distances = pdist(chosen_black_indices)
    ##X * X matrix containing the distance between each and every pixels
    distance_matrix = squareform(distances)

    optimized_path = solve_tsp(distance_matrix)

    optimized_path_points = [chosen_black_indices[x] for x in optimized_path]

    plt.figure(figsize=(8, 10), dpi=100,frameon=False)
    plt.plot([x[1] for x in optimized_path_points],
             [x[0] for x in optimized_path_points],
             color='black', lw=1)
    plt.xlim(0, 600)
    plt.ylim(0, 800)
    #plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig("tmp.png", bbox_inches='tight')
    time_table, x_table, y_table = np.linspace(0, tau, len([x[1] for x in optimized_path_points])),[x[1] for x in optimized_path_points],[x[0] for x in optimized_path_points]


else:
    time_table, x_table, y_table = create_close_loop(image_path)

order = 100# We need higher order approximation to get better approximation
coef = coef_list(time_table, x_table, y_table, order)

space = np.linspace(0, tau, 300) # Did you know what tau is ? Check my previous video about it ! :D
x_DFT = [DFT(t, coef, order)[0] for t in space]
y_DFT = [DFT(t, coef, order)[1] for t in space]


fig, ax = plt.subplots(figsize=(5, 5))
#plt.xlim(xmin=-8000)
plt.axis('off')
ax.plot(x_DFT, y_DFT, 'r--')
ax.plot(x_table, y_table, 'k-')
ax.set_aspect('equal', 'datalim')
xmin, xmax = xlim()
ymin, ymax = ylim()


anim = visualize(x_DFT, y_DFT, coef, order, space, [xmin, xmax, ymin, ymax])
#Writer = animation.writers['html']
writer='imagemagick'
anim.save(args.out_file+'.gif',writer=writer, fps=60)
