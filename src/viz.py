import arviz as az
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot_density(x, title="", width=10, height=7, shade=0.5, bw=0.99,
                 title_fontsize=12, xlabel=""):
    plt.figure(figsize=(width, height))
    az.plot_density(x, outline=False, shade=shade,
                    bw=bw, point_estimate="median")
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel)
