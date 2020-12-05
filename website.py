import re
import pandas as pd
import streamlit as st

import plotter
from model_types import ModelTypes, model_types_map, NatModelTypes, nat_model_types_map


st.write("""
## [Measuring Robustness to Natural Distribution Shifts in Image Classification](https://modestyachts.github.io/imagenet-testbed/)
[Rohan Taori](https://rohantaori.com/), 
[Achal Dave](http://achaldave.com/), 
[Vaishaal Shankar](http://vaishaal.com/), 
[Nicholas Carlini](https://nicholas.carlini.com/), 
[Benjamin Recht](http://people.eecs.berkeley.edu/~brecht/), 
[Ludwig Schmidt](https://people.csail.mit.edu/ludwigs/)
### Plotting Playground
Play around with the evaluation data from [our testbed](https://github.com/modestyachts/imagenet-testbed) below (options in the left sidebar).
""")
    
df = pd.read_csv('results.csv', index_col='model')
df_metadata = pd.read_csv('results_metadata.csv', index_col='model')

df.loc['resnet50', 'imagenet-a'] = float('nan')

help = st.sidebar.button('Plotting Options Help')

st.sidebar.markdown('''
### Plotting Playround Options

''')

presets = {'ImageNetV2': ['val', 'imagenetv2-matched-frequency-format-val'],
           'ObjectNet': ['val-on-objectnet-classes', 'objectnet-1.0-beta'],
           'Vid-Robust (Dataset)': ['val-on-vid-robust-classes', 'imagenet-vid-robust_pm0'],
           'YTBB-Robust (Dataset)': ['val-on-ytbb-robust-classes', 'ytbb-robust_pm0'],
           'Vid-Robust (Consistency)': ['imagenet-vid-robust_pm0', 'imagenet-vid-robust_pm10'],
           'YTBB-Robust (Consistency)': ['ytbb-robust_pm0', 'ytbb-robust_pm10'],
           'Image Corruptions': ['val', 'avg_corruptions'],
           'Lp Adversarial Attacks': ['val', 'avg_pgd'],}
use_preset = st.sidebar.checkbox('Use a Preset Distribution Shift')
if use_preset:
    preset = st.sidebar.selectbox('Preset Distribution Shift', list(presets.keys()))

def format_fn(x):
    x = re.sub('[-_]', ' ', x)
    x = re.sub('format val', '', x)
    x = re.sub('pgd.', 'pgd ', x)
    x = re.sub('.eps', ' eps ', x)
    x = re.sub('avg', ' average ', x)
    x = re.sub('val', 'imagenet val', x)
    x = re.sub('in memory', '(in memory)', x)
    x = re.sub('on disk', '(on disk)', x)
    return x

def selector(columns, category):
    if category == 'ImageNet Validation Set':
        return [x for x in columns if 'val-on' in x or x == 'val']
    if category == 'Image Corruptions':
        return [x for x in columns if 'disk' in x or 'memory' in x or x in ['greyscale', 'stylized_imagenet', 'avg_corruptions']]
    if category == 'Lp Adversarial Attacks':
        return [x for x in columns if 'pgd' in x or x == 'avg_pgd']
    if category == 'Natural Distribution Shifts':
        return [x for x in columns if 'pm0' in x or 'pm10' in x or 'imagenetv2' in x or x in ['imagenet-a', 'imagenet-r', 'imagenet-sketch', 'objectnet-1.0-beta']]


if use_preset:
    x_axis, y_axis = presets[preset]
else:
    categories = ['ImageNet Validation Set', 'Natural Distribution Shifts', 'Image Corruptions', 'Lp Adversarial Attacks']
    x_axis_category = st.sidebar.selectbox('X-axis test set category:', categories, 0)
    x_axis = st.sidebar.selectbox('X-axis test set:', sorted(selector(df.columns, x_axis_category)), 5 if x_axis_category == 'Natural Distribution Shifts' else 0, format_func=format_fn)
    y_axis_category = st.sidebar.selectbox('Y-axis test set category:', categories, 1)
    y_axis = st.sidebar.selectbox('Y-axis test set:', sorted(selector(df.columns, y_axis_category)), 5 if y_axis_category == 'Natural Distribution Shifts' else 0, format_func=format_fn)


plot_style = st.sidebar.radio('Plot Style:', ['Pretty', 'Interactive'])

color_scheme = st.sidebar.radio('Model Color Scheme:', ['Simple', 'Highlight Lp-robust models'])
if color_scheme == 'Simple':
    model_types, model_map = NatModelTypes, nat_model_types_map
else:
    model_types, model_map = ModelTypes, model_types_map    

transform = st.sidebar.radio('Axis Scaling:', ['Logit', 'Linear'])

def make_plot(x_axis, y_axis, df, df_metadata):
    df = plotter.add_plotting_data(df, [x_axis, y_axis])

    df_visible = df[df.show_in_plot == True]
    xlim = [df_visible[x_axis].min() - 2, df_visible[x_axis].max() + 2]
    xlim = [max(xlim[0], 0.1), min(xlim[1], 99.9)]
    ylim = [df_visible[y_axis].min() - 2, df_visible[y_axis].max() + 2]
    ylim = [max(ylim[0], 0.1), min(ylim[1], 99.9)]

    if plot_style == 'Pretty':
        fig, _  = plotter.model_scatter_plot(df, x_axis, y_axis, xlim, ylim, model_types,
                                         transform=transform.lower(), tick_multiplier=5, num_bootstrap_samples=100,
                                         title=f'Distribution Shift Plot ({transform} Scaling)', x_label=x_axis, y_label=y_axis, 
                                         figsize=(9, 8), include_legend=True, return_separate_legend=False)

    elif plot_style == 'Interactive':
        fig  = plotter.model_scatter_plot_interactive(df, x_axis, y_axis, xlim, ylim, model_types,
                                         transform=transform.lower(), tick_multiplier=5, num_bootstrap_samples=100,
                                         title=f'Distribution Shift Plot ({transform} Scaling)', x_label=x_axis, y_label=y_axis, 
                                         height=650, width=750, include_legend=True, return_separate_legend=False)
    
    return fig, df

def prepare_df_for_plotting(df, df_metadata, columns):
    df = df[list(set(columns))]
    df_metadata = df_metadata[[x+'_dataset_size' for x in set(columns)]]
    df = df.merge(df_metadata, right_index=True, left_index=True)
    df = df.dropna()

    df['model_type'] = df.apply(get_model_type, axis=1)
    df['show_in_plot'] = df.apply(show_in_plot, axis=1)
    df['use_for_line_fit'] = df.apply(use_for_line_fit, axis=1)

    return df

def get_model_type(df_row):
    return model_map[df_row.name]

def show_in_plot(df_row):
    model_name, model_type = df_row.name.lower(), df_row.model_type
    return 'subsample' not in model_name and model_name != 'resnet50_with_defocus_blur_aws'

def use_for_line_fit(df_row):
    model_name, model_type, in_plot = df_row.name.lower(), df_row.model_type, df_row.show_in_plot
    return 'aws' not in model_name and 'batch64' not in model_name and model_type is model_types.STANDARD and in_plot

plotter.label_fontsize = 18
plotter.legend_fontsize = 15
plotter.tick_fontsize = 15

df = prepare_df_for_plotting(df, df_metadata, [x_axis, y_axis])

if not help:
    if x_axis=='val' and y_axis=='imagenetv2-matched-frequency-format-val' and plot_style=='Pretty' and transform=='Logit' and color_scheme=='Simple':
        st.image('imagenetv2.png', use_column_width=True)

    else:
        fig, df = make_plot(x_axis, y_axis, df, df_metadata)

        if plot_style == 'Pretty':
            st.pyplot(fig, dpi=200)
        elif plot_style == 'Interactive':
            st.plotly_chart(fig)

    st.write("Top-1 accuracies on the selected distributions (click on the headers to sort):")
    st.dataframe(df[df.show_in_plot][[x_axis, y_axis]].sort_index())


st.write(
"""
**Presets**: If selected individual axes is a bit overwhelming (there are many distributions!), click on "Use a Preset Distribution Shift"
to be able to select one of the recommended presets. These presets are main figures and central to our analysis in [our paper](https://arxiv.org/abs/2007.00644).

**Axis Selection**:
Use the X and Y axis selectors to compare model performance on various distribution shifts.
The shifts are broken down by category for easier perusal.
We recommend selecting an X axis from the "ImageNet Validation Set" category, and a Y axis from one of the other categories.
This will allow you to plot out-of-distribution performance (y axis) as a function of in-distribution performance (x-axis) for the ImageNet models in our testbed.

**Plot Style**:
Select Interactive mode to be able to zoom and view information for each model on mouse hover,
and select Pretty mode for paper-quality plots with error bars.

**Model Color Scheme**:
We provide an option to separately color the $\\ell_p$-adversarially robust models 
from the rest of the robustness intervention models. Our testbed contains ~200 models (including standard ImageNet models, 
models trained to be robust to either image corruptions or adversarial noise, and models trained on extra data apart from the ImageNet training set).

**Plot Information**: 
This website allows you to create plots comparing any two distribution shifts from our testbed and evaluate the robustness of various ImageNet models.
Central to our analysis is measuring a model's robustness via its _effective robustness_, ie. robustness distinct from accuracy.
In our plots, baseline expected accuracy is shown in red by a linear fit (the fit is computed using the blue points only), 
and robust models with positive effective robustness are those significantly above the line.
In many cases, an ideal robust model would be as close to the y=x line as possible, indicating no performance drop
between in-distribution and out-of-distribution tasks. 
Please see Section 2 of [our paper](https://arxiv.org/abs/2007.00644) for a more thorough discussion.


### Citation
```
@inproceedings{taori2020measuring,
    title={Measuring Robustness to Natural Distribution Shifts in Image Classification},
    author={Rohan Taori and Achal Dave and Vaishaal Shankar and Nicholas Carlini and Benjamin Recht and Ludwig Schmidt},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2020},
    url={https://arxiv.org/abs/2007.00644},
}
```
""")
