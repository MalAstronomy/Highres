import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from train_new import LABELS, LOWER, UPPER
from adding_legends import legends

from lampe.plots import corner 
from lampe.plots import LinearAlphaColormap


def corner_mod(theta, legend=['NPE', 'NS'], color= ['steelblue', 'orange'] , figsize=(10,10), \
               domain = (LOWER[12:], UPPER[12:]), labels= LABELS[12:] ):

    # creating a whole new figure and define legends needed
    figure, axes = plt.subplots( figsize[0], figsize[0], squeeze=False, sharex='col', \
                                gridspec_kw={'wspace': 0., 'hspace': 0.}, )
    for i in range(len(legend)):
        lines = axes[0, -1].plot([], [], color=color[i], label=legend[i])
    handles, texts = legends(axes, alpha = (0., .9))
    plt.close(figure)
    
    fig = None

    for i, th in enumerate(theta):
        fig = corner(
                th,
                smooth=2,
                domain = domain,
                labels= labels,
                figsize= figsize,
                creds= [0.997, 0.955, 0.683], 
                alpha = [0, 0.9],
                color= color[i],
#                 show_titles = True,
                figure = fig
            )

    for index,ax in enumerate(fig.get_axes()):   
        ax.tick_params(axis='both', labelsize=12)
        if index<10:
            if index==0:
                ax.set_xlabel('', fontsize=12)
                ax.set_ylabel('', fontsize=12)
            else:
                ax.set_xlabel(LABELS[index], fontsize=12)
                ax.set_ylabel(LABELS[index], fontsize=12)
        else:continue

    # replacing the new figure legends into the corner plot
    fig.legends.clear()
    fig.legend(handles, texts, loc='center', bbox_to_anchor=(0.4,0.915), frameon=False,  prop={'size': 12})
    
    return fig

