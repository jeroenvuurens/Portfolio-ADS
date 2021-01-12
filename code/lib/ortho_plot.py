import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ortho_lib3_Copy2 import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


def get_file(category_num = 1, 
            patient_num = 1, 
            exercise = 'AB1', 
            data_dir = 'data',
            file_type = 'txt'):
    return os.path.join(data_dir, 'Category_' + str(category_num), str(patient_num), exercise + '.' + file_type)

def get_df(category_num = 1, 
            patient_num = 1, 
            exercise = 'AB1', 
            data_dir = 'data',
            with_rotations = False
          ):
    path = get_file(category_num, patient_num, exercise, data_dir)
    if with_rotations is True:
        return exercise_to_df_with_rotation(path, False).reset_index(drop=True)
    else:
        return exercise_to_df(path, False).reset_index(drop=True)

def calculate_xyz_diffs(df):
    # add tmp columns and difference columns
    df[['compl_x', 'compl_y', 'compl_z', 'diff_x', 'diff_y', 'diff_z', 'diff_xz', 'diff_xy', 'diff_yz', 'diff']] = 0

    sensor_pairs = {
        '4': '7',
        '5': '8',
        '6': '9',
        '7': '4',
        '8': '5',
        '9': '6'
    }

    # Calculate difference columns
    for i in sensor_pairs:
        # select values for with current sensor
        s_values = df[df['sensor'] == i]
        s_values.loc[s_values.index,'compl_x'] = df[df['sensor'] == sensor_pairs[i]].x.values
        s_values.loc[s_values.index,'compl_y'] = df[df['sensor'] == sensor_pairs[i]].y.values
        s_values.loc[s_values.index,'compl_z'] = df[df['sensor'] == sensor_pairs[i]].z.values

        s_values.loc[s_values.index,'diff_x'] = (s_values.x - s_values.compl_x).abs()
        s_values.loc[s_values.index,'diff_y'] = (s_values.y - 1 + s_values.compl_y).abs()
        s_values.loc[s_values.index,'diff_z'] = (s_values.z - s_values.compl_z).abs()
        s_values.loc[s_values.index,'diff_xz'] = (s_values.diff_x + s_values.diff_z).values
        s_values.loc[s_values.index,'diff_xy'] = (s_values.diff_x + s_values.diff_y).values
        s_values.loc[s_values.index,'diff_yz'] = (s_values.diff_y + s_values.diff_z).values
        s_values.loc[s_values.index,'diff'] = (s_values.diff_x + s_values.diff_y + s_values.diff_z).values

        df[df['sensor'] == i] = s_values

    # drop tmp columns
    return df.drop(['compl_x', 'compl_y', 'compl_z'], axis=1)
    
def get_df_figure_3d(df,
            types = ['scatter', 'line'],
            colored_diffs = True):
    plots = []
    df.loc[df.index, 'sensor'] = pd.Categorical(df['sensor'], ['6','5','4','3','7','8','9'])
    df = df.sort_values(['frame', 'sensor'])
    
    if colored_diffs is True:
        df = calculate_xyz_diffs(df)
    
    if 'scatter' in types:
        plots.append(plot_scatter_3d(df, colored_diffs=colored_diffs))
    
    if 'line' in types:
        plots.append(plot_line_3d(df))

    fig = go.Figure(data=plots[0])
    if len(plots) > 1:
        fig.add_trace(plots[1]['data'][0])

    # build the frame structure for all plots
    number_frames = len(df['frame'].unique())
    frames = [dict(name = k, data = [plot['frames'][k]['data'][0] for plot in plots]) for k in range(number_frames)]

    fig = fig.update(frames=frames)

    camera = dict(
        eye = dict(x=2, y=1.5, z=1.5)
    )
    scene = dict(
        aspectmode = "manual", 
        aspectratio = dict( x = 1, y = 1, z = 1)
    )
    fig.update_layout(scene = scene, scene_camera=camera)

    return fig
    
def get_exercise_figure_3d(category_num = 1, 
            patient_num = 1, 
            exercise = 'AB1', 
            data_dir = 'calibrated_data',
            types = ['scatter', 'line'],
            colored_diffs = True
                     ):
    if not types:
        print('Please specify at least one of these plot types: scatter, line')
        return
    
    df = get_df(category_num, patient_num, exercise, data_dir)
    
    return get_df_figure_3d(df, types, colored_diffs)

def plot_cone_3d(df_with_rotations, xrange = [], yrange = [], zrange = []):
    if not xrange:
        xrange = [df['x'].min(),df['x'].max()]
    if not yrange:
        yrange = [df['y'].min(),df['y'].max()]
    if not zrange:
        zrange = [df['z'].min(),df['z'].max()]

    kwargs = dict(range_x=xrange, range_y=yrange, range_z=zrange)

#     if 'diff' in df.columns and colored_diffs is True:
#         if df['diff'].max() < .15:
#             color_max = .15
#         elif df['diff'].max() < .5:
#             color_max = .5
#         else:
#             color_max = df['diff'].max()
            
#         colors = dict(color='diff',range_color=[0,color_max],color_continuous_scale=['green', 'red'])
        
#         # merge into kwargs
#         kwargs = {**kwargs, **colors}

    return go.Figure(data = go.Cone(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            u=df['u'],
            v=df['v'],
            w=df['w'],
#             colorscale='Blues',
#             sizemode="absolute",
#             sizeref=40
    ))
    
    return px.scatter_3d(df, 
                         x="x", y="y", z='z', 
                         animation_frame="frame", 
                         animation_group="sensor", 
                         hover_name="sensor",
                         log_x=False, 
                         **kwargs
                        )

def plot_scatter_3d(df, xrange = [], yrange = [], zrange = [], colored_diffs = False):
    if not xrange:
        xrange = [df['x'].min(),df['x'].max()]
    if not yrange:
        yrange = [df['y'].min(),df['y'].max()]
    if not zrange:
        zrange = [df['z'].min(),df['z'].max()]

    kwargs = dict(range_x=xrange, range_y=yrange, range_z=zrange)

    if 'diff' in df.columns and colored_diffs is True:
        if df['diff'].max() < .15:
            color_max = .15
        elif df['diff'].max() < .5:
            color_max = .5
        else:
            color_max = df['diff'].max()
            
        colors = dict(color='diff',range_color=[0,color_max],color_continuous_scale=['green', 'red'])
        
        # merge into kwargs
        kwargs = {**kwargs, **colors}
    
    return px.scatter_3d(df, 
                         x="x", y="y", z='z', 
                         animation_frame="frame", 
                         animation_group="sensor", 
                         hover_name="sensor",
                         log_x=False, 
                         **kwargs
                        )

def plot_line_3d(df, xrange = [], yrange = [], zrange = []):
    if not xrange:
        xrange = [df['x'].min(),df['x'].max()]
    if not yrange:
        yrange = [df['y'].min(),df['y'].max()]
    if not zrange:
        zrange = [df['z'].min(),df['z'].max()]
    
    return px.line_3d(df, 
                      x="x", y="y", z='z', 
                      animation_frame="frame", 
                      animation_group="sensor", 
                      hover_name="sensor",
                      log_x=False, 
                      range_x=xrange, 
                      range_y=yrange, 
                      range_z=zrange
    )

# def df_figure_2d_slices(df, side = 'both'):
#     if side not in ['left', 'right', 'both']:
#         print ('Please enther one of the following for the "side" argument: "left", "right", "both"')
#     df_right = 

#     shape = (0,0)
#     plots = []
    
#     if side == 'left' or side == 'both':
#         shape[0] += 1
#         shape[1] += 3
#         plots.append({
#             'x': 'x',
#             'y', 'z',
            
#         })
#     if side == 'right' or side == 'both':
#         shape[0] += 1
#         shape[1] += 3

def plot_scatter_2d(df, x, y):
    return px.scatter(df, 
                    x=x, y=y,
                    animation_frame=animation_frame, 
                    animation_group=animation_group, 
                    hover_name="sensor", 
                      hover_data=[
                          size_col,
                          color_col
                          ],
                    color=color_col, 
                    color_continuous_scale=['green', 'red'],
#                     size=size_col
                  )

def plot_line_2d(df, x, y):
    return px.line(df, 
                    x=x, y=y,
                    animation_frame="frame", 
                    animation_group="sensor", 
                    hover_name="sensor"
               )

def plot_angles_shoulder_elbow_height(df):
    fig, axs = plt.subplots(3, 2, sharey=True, figsize=(20,20))
    fig.suptitle('Rotation shoulders vs. height elbow & shoulder', size=25)
    angles = ['rad1', 'rad2', 'rad3']
    elbows = ['5', '8']
    shoulders = ['4', '7']
    cols = ['Left shoulder', 'Right shoulder']
    for ax, col in zip(axs[0], cols):
        ax.set_title(col, size=15)

    for col in range(2):
        for row in range(3):
            axs[row, col].plot(df[df['sensor']==shoulders[col]]['frame'], df[df['sensor']==shoulders[col]]['z'], label='height_shoulder')
            axs[row, col].plot(df[df['sensor']==elbows[col]]['frame'], df[df['sensor']==elbows[col]]['z'], label='height_elbow')
            axs[row, col].set_xlabel('frames')
            axs[row, col].set_ylabel('height')
            axs[row, col].legend(loc=3)
            ax = axs[row, col].twinx()
            ax.plot(df[df['sensor']==shoulders[col]]['frame'], df[df['sensor']==shoulders[col]][angles[row]], label=angles[row], color='r')
            ax.set_ylabel('angle in radians')
            ax.set_ylim([-1,1])
            ax.legend(loc=4)

def plot_results(f_results):
    factor = list(f_results.keys())
    fn = list(values['fn'] for values in f_results.values())
    
    fig = plt.figure(figsize = (10, 5))
    
    plt.bar(factor, fn, width=0.8/len(factor), color='maroon')
    plt.xlabel('factor')
    plt.ylabel('No. of false negatives')
    plt.title('No. of false negatives per factor result')
    
    
    
def plot_logres_model(clf, X, y):
    # General a toy dataset:s it's just a straight line with some Gaussian noise:
    xmin, xmax = X.min(), X.max()

    # and plot the result
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.scatter(X.ravel(), y, color='black', zorder=20)
    X_test = np.linspace(-5, 10, 300)

    loss = expit(X * clf.coef_ + clf.intercept_).ravel()
    plt.plot(X, loss, color='red', linewidth=3)

    plt.ylabel('y')
    plt.xlabel('X')
    plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
               loc="lower right", fontsize='small')
    plt.tight_layout()
    plt.show()