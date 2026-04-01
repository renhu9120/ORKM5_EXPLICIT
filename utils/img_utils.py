import matplotlib.pyplot as plt

def plot_conv_curvs(x, data_series, title="", xlabel="Iteration", ylabel="log(dist(x,z))", figsize=(10, 6), grid=True,
                    filename=""):
    """
    将多组向量绘制在同一张图中。

    参数：
    - x: 横坐标向量（共用）
    - data_series: 列表，元素为字典，格式如：
        {"y": y1, "label": "method1", "style": "-", "color": "b"}
    - title: 图标题
    - xlabel, ylabel: 坐标轴名称
    - figsize: 图像大小
    - grid: 是否显示网格
    - save_path: 若提供文件名，则保存图像到该路径

    用法示例：
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    data_series = [
        {"y": y1, "label": "sin(x)", "style": "-", "color": "blue"},
        {"y": y2, "label": "cos(x)", "style": "--", "color": "red"}
    ]
    plot_vectors(x, data_series)
    """

    plt.figure(figsize=figsize)

    for series in data_series:
        y = series["y"]
        label = series.get("label", "")
        style = series.get("style", "-")
        color = series.get("color", None)
        markevery = series.get("markevery", None)
        plt.plot(x, y, style, label=label, color=color, linewidth=2, markevery=markevery)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if grid:
        plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    # plt.legend(loc="upper right")  # 与plt.legend(loc=1)等价
    plt.tight_layout()

    # 判断 filename 是否非空，如果是就保存图片
    if filename.strip():
        plt.savefig(f"{filename}.pdf", format='pdf', bbox_inches='tight')
        print(f"Figure saved as {filename}.pdf")
        plt.show()
    else:
        plt.show()



def plot_sr_curvs(
        alg_list,
        *,
        x_key: str,
        y_key: str,
        xlabel: str,
        ylabel: str,
        title: str = "",
        figsize=(10, 6),
        save_name: str = "comparison_plot.pdf",
        dpi: int = 300
):
    """
    Plot comparison curves for multiple algorithms using their CSV data files.

    Parameters
    ----------
    alg_list : list of dict
        Each element is:
        {
            "file": "filename.csv",
            "label": "Algorithm Name",
            "linestyle": "--",
            "color": "red"
        }

    x_key : str
        Column name for x-axis (e.g., 'nd_rate').

    y_key : str
        Column name for y-axis (e.g., 'success_rate', 'avg_iters').

    xlabel, ylabel : str
        Axis labels for the plot.

    title : str, optional
        Title of the plot.

    figsize : tuple
        Figure size.

    save_name : str
        PDF file name to save the figure.

    dpi : int
        DPI for high-quality paper figures.
    """
    plt.figure(figsize=figsize)

    for item in alg_list:
        filename = item["file"]
        label = item.get("label", "Algorithm")
        ls = item.get("linestyle", "-")
        color = item.get("color", None)

        # Read CSV
        df = pd.read_csv(filename)

        # Ensure x and y keys exist
        if x_key not in df.columns or y_key not in df.columns:
            raise KeyError(f"Column {x_key} or {y_key} not found in file {filename}")

        x = df[x_key].values
        y = df[y_key].values

        # Plot
        plt.plot(
            x, y,
            ls,
            color=color,
            linewidth=2,
            markersize=5,
            label=label
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if title:
        plt.title(title, fontsize=13)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.legend(loc="upper left")  # 与plt.legend(loc=1)等价

    # 判断 filename 是否非空，如果是就保存图片
    if save_name.strip():
        # Save figure
        plt.tight_layout()
        plt.savefig(save_name, dpi=dpi)

    plt.show()
    print(f"Figure saved as {save_name}")
