'''
Script to make submission plots.
'''

import json
import plotly.io as pio
import numpy as np
import plotly.graph_objs as go
from scipy.signal import savgol_filter
import colorlover as cl
from IPython.display import HTML

def plotter(data, labels, colors, filename):
    # we use plotly for plotting

    traces = []
    for d, l, c in zip(data, labels, colors):
        d = np.asarray(d)
        print("hello")
        trace = go.Scatter(
            x=d[:, 0],
            y=1-d[:, 1],
            mode='lines',
            name=l,
            line=dict(
                width=6,
                color=c,
            )
        )
        traces.append(trace)

    layout = dict(
        showlegend=True,
        legend=dict(orientation='v', x=0.8, y=0.15, font=dict(size=26), xanchor='left'),
        margin=go.Margin(l=30, r=30, b=30, t=30, pad=4),
        width=1200,
        height=800,
        xaxis=dict(range=[0, 3e3], domain=[0.05, 1.], zeroline=False, title="Training Iterations", titlefont=dict(size=26),
                   tickfont=dict(
                       size=20,
                       color='black'
                   )),
        yaxis=dict(range=[.2, 1.], domain=[.1, 1.], zeroline=False, title="Accuracy", titlefont=dict(size=26),
                   tickfont=dict(
                       size=20,
                       color='black'
                   )),
    )

    fig = go.Figure(data=traces, layout=layout)
    pio.write_image(fig, filename)
    return


def create_data_from_json(filename):
    #get the correct datapoints from a json file. in this case the validation accuracy.
    data = []
    with open(filename) as f:
        json_data = json.load(f)
        train_loss = json_data['val_accuracy']
        for key, value in train_loss.items():
            data.append([float(key), float(value)])
        return data


if __name__ == '__main__':
    #list of json files containing data.
    list_of_jsons = [
        'runs/W-Net-Intermediate_opt_Adam_data_constant_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164639',
        'runs/U-Net_opt_Adam_data_constant_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164642',
        'runs/W-Net-Intermediate_opt_Adam_data_TrainAugmented_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164709',
        'runs/U-Net_opt_Adam_data_TrainAugmented_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164709',
        'runs/W-Net-Intermediate_opt_Adam_data_Train_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164708',
        'runs/U-Net_opt_Adam_data_Train_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_162733']
    # labels used in the legend of the plot
    labels = [
        'W-Net, Sup.+Aug.',
        'U-Net, Sup.+Aug.',
        'W-Net, Augmented',
        'U-Net, Augmented',
        'W-Net, Original',
        'U-Net, Original',
    ]
    colors = cl.scales[str(len(list_of_jsons))]['qual']['Paired']
    d = []
    for l in list_of_jsons:
        d.append(create_data_from_json(l + '/data.json'))
    plotter(d, labels, colors, "wu_all_data.pdf")


list_of_jsons = [
    'runs/model_Jun_30_174947',
]

labels = [
    'baseline'
]
d = []
for l in list_of_jsons:
    d.append(create_data_from_json(l + '/data.json'))
plotter(d, labels, colors, "baseline.pdf")