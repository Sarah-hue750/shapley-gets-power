import matplotlib.pyplot as plt



def scale_fonts(scaling_factor):
    """
    Scale font sizes for LaTeX-style plots.

    Parameters:
    scaling_factor (float): The scaling factor for font sizes.

    Returns:
    None
    """
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.sf": "Arial",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "pdf.fonttype": 42,
        }
    )

    SMALL_SIZE = 15 * scaling_factor
    MEDIUM_SIZE = 20 * scaling_factor
    BIGGER_SIZE = 30 * scaling_factor

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title