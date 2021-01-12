import pandas as pd
from functools import reduce
from collections import defaultdict
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.axes as mpla
import seaborn as sns
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import os
import itertools
import re
import pickle
import sys
from IPython.display import Audio
import copy
import ortho_plot

from scipy.spatial.transform import Rotation as R

"""

OrthoEyes Librar

Based on classes built by Jeroen Vuurens and Tony Andrioli

This file contains classes specific to the OrthoEyes project 2020.

It is used to read the dataset delivered and convert them to Pandas DataFrame.

To save you some time, here's a list with classes defined in this file:

DFFrame: Container for getting angles between sensors

Exercise:   Data container that holds a single Exercise performed by a patient
            Also includes a compare function to sort the class by category, patient num and exercise type

Exercises:  Data container that holds Exercise objects
            Contains functions to manage the dataset

Experiment: Contains methods to fit,train,test on a set of exercises

FeatureSet: A simple list of features (Dont know why this is usefull)

Frame: A frame in the dataset, contains positional data for sensors

FrameRotation: Extension of the Frame class, also holds rotational data for sensors

Patient:    Data container to remap a category seres to a DataFrame (? Danny ?)

Results: Contains methods to calcument the results of an Experiment

"""

class DFFrame(pd.DataFrame):
    """
    Pandas dataframe with predefined columns
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Shoulder angle for AF & RF (side view)
        self['angle_left_shoulder_xz'] = angle_AF_RF(self, '4', '5')
        self['angle_right_shoulder_xz'] = angle_AF_RF(self, '7', '8')
#         self['lowest_max_shoulder_angle_xz'] = lowest_max_lr(self, 'angle_left_shoulder_xz', 'angle_right_shoulder_xz')
        
        # Shoulder angle for AB (frontal view)
        self['angle_left_shoulder_yz'] = angle_AB(self, '4', '5')
        self['angle_right_shoulder_yz'] = angle_AB(self, '7', '8')
#         self['lowest_max_shoulder_angle_yz'] = lowest_max_lr(self, 'angle_left_shoulder_yz', 'angle_right_shoulder_yz')
        
        # Symmetry x for left and right
        self['diff_x_wrist'] = diff(self, 'x_6', 'x_9')
        self['diff_x_elbow'] = diff(self, 'x_5', 'x_8')
        self['diff_x_shoulder'] = diff(self, 'x_4', 'x_7')
        
        # Symmetry y for left and right
        self['diff_y_wrist'] = diff_mirrored(self, 'y_6', 'y_9')
        self['diff_y_elbow'] = diff_mirrored(self, 'y_5', 'y_8')

        # Symmetry z for left and right
        self['diff_z_wrist'] = diff(self, 'z_6', 'z_9')
        self['diff_z_elbow'] = diff(self, 'z_5', 'z_8')
        self['diff_z_shoulder'] = diff(self, 'z_4', 'z_7')
        
        # Max height for AB, AF & RF
        self['z_elbow'] = lowest_max_lr(self, 'z_5', 'z_8')
        self['z_wrist'] = lowest_max_lr(self, 'z_6', 'z_9')
        
        # Max x for EL
        self['x_wrist'] = highest_min_lr(self, 'x_6', 'x_9')
        
        # Wrist velocity over the x-axis for EL
        self['vel_wrists_x_l'] = get_velocity(self, 'x_6').abs()
        self['vel_wrists_x_r'] = get_velocity(self, 'x_9').abs()
        
        # Wrist acceleration over the x-axis for EL
        self['acc_wrists_x_l'] = get_acceleration(self, 'x_6').abs()
        self['acc_wrists_x_r'] = get_acceleration(self, 'x_9').abs()
        
        # Elbow velocity over the z-axis for AB, AF & RF
        self['vel_elbows_z_l'] = get_velocity(self, 'z_5').abs()
        self['vel_elbows_z_r'] = get_velocity(self, 'z_8').abs()
        
        # Elbow acceleration over the z-axis for AB, AF & RF
        self['acc_elbows_z_l'] = get_acceleration(self, 'z_5').abs()
        self['acc_elbows_z_r'] = get_acceleration(self, 'z_8').abs()
        
        # Length upper arms (needed for angular velocity calculations. NOT a feature)
        self['upper_arm_left'] = length_vector(self, '4', '5') 
        self['upper_arm_right'] = length_vector(self, '7', '8') 
        
        # Elbow velocity over the x- and y-axis (needed for angular velocity calculations. NOT a feature)
        self['vel_elbows_x_l'] = get_velocity(self, 'x_5').abs()
        self['vel_elbows_x_r'] = get_velocity(self, 'x_8').abs()
        self['vel_elbows_y_l'] = get_velocity(self, 'y_5').abs()
        self['vel_elbows_y_r'] = get_velocity(self, 'y_8').abs()
        
        # Elbow velocity over the x- and z-axis (needed for angular velocity calculations. NOT a feature)
        self['vel_xz_elbow_l'] = pyth(self, 'vel_elbows_x_l', 'vel_elbows_z_l')
        self['vel_xz_elbow_r'] = pyth(self, 'vel_elbows_x_r', 'vel_elbows_z_r')
        
        # Elbow angular velocity over the x- and z-axis for AF & RF (side view)
        self['angular_vel_xz_elbow_l'] = abs(self['vel_xz_elbow_l'] / self['upper_arm_left'])
        self['angular_vel_xz_elbow_r'] = abs(self['vel_xz_elbow_r'] / self['upper_arm_right'])
        
        # Elbow angular acceleration over the x- and z-axis for AF & RF (side view)
        self['angular_acc_xz_elbow_l'] = abs(self[['angular_vel_xz_elbow_l']].diff(axis=0))
        self['angular_acc_xz_elbow_r'] = abs(self[['angular_vel_xz_elbow_r']].diff(axis=0))
        
        # Elbow velocity over the y- and z-axis (needed for angular velocity calculations. NOT a feature)
        self['vel_yz_elbow_l'] = pyth(self, 'vel_elbows_y_l', 'vel_elbows_z_l')
        self['vel_yz_elbow_r'] = pyth(self, 'vel_elbows_y_r', 'vel_elbows_z_r')
        
        # Elbow angular velocity over the y- and z-axis for AB (frontal view)
        self['angular_vel_yz_elbow_l'] = abs(self['vel_yz_elbow_l'] / self['upper_arm_left'])
        self['angular_vel_yz_elbow_r'] = abs(self['vel_yz_elbow_r'] / self['upper_arm_right'])
        
        # Elbow angular acceleration over the y- and z-axis for AB (frontal view)
        self['angular_acc_yz_elbow_l'] = abs(self[['angular_vel_yz_elbow_l']].diff(axis=0))
        self['angular_acc_yz_elbow_r'] = abs(self[['angular_vel_yz_elbow_r']].diff(axis=0))


class Exercise:
    """
    Container to read and draw an exercise
    """
    def __init__(self, df, category, extype, patient):
        self.df = df
        self.extype = extype
        self.patient = patient + 100 * int(category[-1])
        self.category = category
    
    def __lt__(self, other):
        """
        Built-in Sort
        """
        if self.category != other.category:
            return self.category < other.category
        if self.patient != other.patient:
            return self.patient < other.patient
        elif self.extype < other.extype:
            return self.extype < other.extype    

    @classmethod
    def columns(self, *columns):
        df = self.df(columns)
        return cls(df, self.category, self.extype, self.patient)
    
    def draw(self, x, y, ax=plt):
        """
        draw a scatter graph of the given x and y columns
        """
        ax.scatter(self.df[x], self.df[y], marker='o', c=self.df.frame)
        
        if isinstance(ax, mpla.SubplotBase):
            ax.set_title(f'{self.category} {self.patient} {self.extype}')
            ax.set(xlabel=x, ylabel=y)
        else:
            ax.title(f'{self.category} {self.patient} {self.extype}')
            ax.xlabel(x)
            ax.ylabel(y)

    def __str__(self):
        return f'Exercise({self.extype}, patient: {self.patient}'


class Exercises(dict):
    """
    Container for a read set of Exercises
    data: A Exercises object of a list of Exercise objects

    contains the following properties:
    exercises: list(oefening)
    patienten: set(patient ids)
    categories: set(category labels)
    extypes: set(extype labels)
    
    en is een dictionary met daarin keys voor elke
    patient id: dict(extype -> Exercise)
    extype: dict(patient id -> Exercise)
    category: dict(patient id -> Exercise)
    """
    
    EXERCISE_TYPES = ('AB', 'AF', 'EL', 'RF')
    #CATEGORIES = ('Category_1', 'Category_3')
    CATEGORIES = ('Category_1', 'Category_2', 'Category_3', 'Category_4')
    
    def __init__(self, data = None):
        if type(data) == Exercises:
            super().__init__(data)
            self.exercises = list(data.exercises)
            self.patients = list(data.patients)
            self.categories = set(data.categories)
            self.extypes = set(data.extypes)
        else:
            super().__init__()
            self.exercises = []
            self.patients = list()
            self.categories = set()
            self.extypes = set()
            if data is not None:
                for o in data:
                    self.append(o)
            self.exercises.sort()
        self._df = None

    def append(self, o):
        self[o.extype][o.patient] = o
        self[o.category][o.patient] = o
        self[o.patient][o.extype] = o
        self.exercises.append(o)
        if o.patient not in self.patients:
            self.patients.append(o.patient)
        self.categories.add(o.category)
        self.extypes.add(o.extype)
    
    def select_extype(self, *extype):
        """
        select a subset of one or more exercises
        """
        return Exercises(data=[ o for o in self.exercises if o.extype in extype ])

    def select_patient(self, patient):
        """
        select a subset of one or more patients
        """
        if isinstance(patient, int):
            patient = [patient]
        
        
        return Exercises(data=[ o for o in self.exercises if o.patient in patient ])   

    def select_category(self, *category):
        """
        select a subset of one or more categories, supply as int number
        """
        category = [ f'Category_{c}' for c in category ]
        return Exercises(data=[ o for o in self.exercises if o.category in category ])
    
    def drop_patient(self, *patient):
        """
        return a subset without these patients
        """
        oef = [ o for o in self.exercises if o.patient not in patient ]
        return Exercises(oef) 

    def drop_extype(self, *extype):
        """
        return a subset without these exercise types
        """
        oef = [ o for o in self.exercises if o.extype not in extype ]
        return Exercises(oef)
    
    def drop_category(self, *category):
        """
        return a subset without these categories, supply as int number
        """
        category = [ f'Category_{c}' for c in category ]
        oef = [ o for o in self.exercises if o.category not in category ]
        return Exercises(oef)

    def add_labels(self, file, oldcat='Category_3'):
        with open(file) as fin:
            for line in fin:
                w = line.split(',')
                patientid = int(w[0])
                category = w[1].strip()
                self.categories.add(category)
                self[category][patientid] = self[patientid]
                del self[oldcat][patientid]
                for o in self[patientid].values():
                    o.category = category
    
    @classmethod
    def all(cls):
        """
        Create an Exercise instance from all files available.
        """
        return cls.from_files( FilesCategory.from_categories(*cls.CATEGORIES) )
    
    @classmethod
    def from_files(cls, files):
        """
        Create an Exercises instance from a list of tuples (filename, category_name, exercise_type, patient_id).
        """
        return cls( [ exercise_from_file(file, category, extype, patient) for file, category, extype, patient in files ] )
    
    @classmethod
    def from_exercise_objects(cls, exs):
        return cls(exs)
    
    def dump(self, file):
        with open(file, 'wb') as fout:
            pickle.dump(self, fout)
    
    @staticmethod
    def load(file):
        with open(file, 'rb') as fin:
            return pickle.load(fin)
    
    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        if type(key) == int:
            self[key] = Patient()
        else:
            self[key] = dict()
        return self[key]

    @property
    def df(self):
        if self._df is not None:
            return self._df
        else:
            self._df = pd.concat([ self[p].to_df() for p in self.patients ], axis=1).transpose()
            return self._df

    def to_df_min(self):
        return pd.concat([ self[p].to_df_min() for p in self.patients ], axis=1).transpose()

    def to_df_max(self):
        return pd.concat([ self[p].to_df_max() for p in self.patients ], axis=1).transpose()

    @property
    def ids(self):
        try:
            return self._ids
        except:
            self._ids = pd.DataFrame(self.patients)
            return self._ids

    @property
    def y(self):
        try:
            return self._y
        except:
            self._y = pd.DataFrame([[self[p][extype].category for extype in self.extypes] for p in self.patients])
            self._y = self._y[0].to_numpy().reshape(len(self.patients))
            return self._y

    def to_np(self, df = None):
        if df is None:
            df = self.df
        return df.to_numpy(), self.y
    
    def to_np_min(self):
        return self.to_np(self.to_df_min())

    def to_np_max(self):
        return self.to_np(self.to_df_max())

    def lim(self, y):
        return min([ o.df[y].min() for o in self.exercises ]),\
               max([ o.df[y].max() for o in self.exercises ])
    
    def tsne(self, df=None, perplexity=8):
        X, labels = self.to_np(df=df)
        X_embedded = TSNE(n_components=2, perplexity=perplexity).fit_transform(X)
        plt.figure(figsize=(16,10))
        for label_id in np.unique(labels):
            X_label = X_embedded[[labels == label_id]]
            plt.scatter(X_label[:, 0],
                        X_label[:, 1],
                        marker='o',
                        linewidth='1',
                        alpha=0.8,
                        label=label_id)
            plt.legend(loc='best')
    
    def draw(self, x, y, figwidth = 12, figheight=None, ncols=5, nrows=None):
        """
        teken alle oefeningen
        x: column uit de dataframes op de x-as
        y: column uit de dataframes op de y-as
        figwidth: breedte van een plot
        figheight: hoogte van een plot, of None voor auto
        ncols: aantal plaatjes naast elkaar
        nrows: aantal plaatjes boven elkaar, of None voor auto
        """
        if figheight == None:
            figheight = 0.25 * figwidth * ((len(self.exercises) - 1) // ncols + 1)
        if nrows == None:
            nrows = math.ceil(len(self.exercises) / ncols)
            
        self.fig, self.ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(figwidth, figheight))

        # ylim per extype
        ylim = { extype:self.select_extype(extype).lim(y) for extype in self.extypes}

        for i, o in enumerate(self.exercises):
            ax = self.ax[i // ncols, i % ncols]
            o.draw(x, y, ax=ax)
            plt.setp(ax, ylim=ylim[o.extype])
        self.fig.tight_layout()
        
        
class Experiment:
    def __init__(self, exercises, y_condition=lambda y: y == 'Category_3', cols = None, class_weight=None):
        self.exercises = exercises
        self.df = exercises.df
        self.class_weight = class_weight
        if cols is None:
            self.cols = self.df.columns
        else:
            self.cols = cols
            self.df = self.df[cols]
        self.y = exercises.y
        self.y = (y_condition(self.y)) * 1.0
        self.ids = exercises.patients
        self.models = {}

    def X(self, fs):
        return self.df[fs]

    def X_np(self, *columns):
        return self.X(*columns).to_numpy()

    def fit_mlp(self, X_train, y_train):
        model = MLPRegressor()
        model.fit(X_train, y_train)
        return model
    
    def fit(self, X_train, y_train):
        model = LogisticRegression(class_weight=self.class_weight)
        model.fit(X_train, y_train)
        return model

    def fit_predict(self, X_train, y_train, X_valid):
        model = self.fit(X_train, y_train)
        return model.predict(X_valid)

    def fit_predict_p(self, X_train, y_train, X_valid):
        model = self.fit(X_train, y_train)
        return model.predict_proba(X_valid)[:, 1]

    def scale(self, X_train, X_valid):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        return X_train_scaled, X_valid_scaled

    def fit_loo(self, X, y):
        loo = LeaveOneOut()
        y_pred = np.zeros(y.shape)
        for train_index, valid_index in loo.split(X):
            X_train_scaled, X_valid_scaled = self.scale(X[train_index], X[valid_index])
            y_pred[valid_index] = self.fit_predict(X_train_scaled, y[train_index], X_valid_scaled)
        return y_pred

    def fit_loo_p(self, X, y):
        loo = LeaveOneOut()
        y_pred = np.zeros(y.shape)
        for train_index, valid_index in loo.split(X):
            X_train_scaled, X_valid_scaled = self.scale(X[train_index], X[valid_index])
            y_pred[valid_index] = self.fit_predict_p(X_train_scaled, y[train_index], X_valid_scaled)
        return y_pred

    def compare_featuresets(self, *feature, results=None):
        if results is None:
            results = Results(self)
        for f in feature:
            fs = FeatureSet(f)
            X = self.X(fs).to_numpy()
            y_pred = self.fit_loo(X, self.y)
            results.score(fs, self.y, y_pred)
        return results
    
    def keep_inliers(self, X, y, factor=1.1):
        X0 = X[y == 0]
        mean0 = np.mean(X0)
        if np.mean(X[y == 1]) < mean0:
            min0 = mean0 - factor * max(mean0 - X0)
            keep = ((y == 0)|(X < min0 ).flatten())
        else:
            max0 = mean0 + factor * max(X0 - mean0)
            keep = ((y == 0)|(X > max0).flatten())
        return X[keep], y[keep]
    
#     def keep_inliers(self, X, y, factor=1.1):
#         X0 = X[y == 0]
#         mean0 = np.mean(X0)  
#         if np.mean(X[y == 1]) < mean0:
#             X0low = np.min(X0)
#             t = (1-factor) * mean0 + factor*X0low
#             #print('min', X0low, t)
#             keep = ((y == 0)|(X < t ).flatten())
#         else:
#             X0high = np.max(X0)
#             t = (1-factor) * mean0 + factor*X0high
#             keep = ((y == 0)|(X > t ).flatten())
#             #print('max', X0high, t)
#         return X[keep], y[keep]

    def fit_inliers1(self, featureset, factor=1.1, print_failed_models = False):
        loo = LeaveOneOut()
        fs = FeatureSet(featureset)
        X = self.X(fs).to_numpy()
        y = self.y
        y_pred = np.zeros(y.shape)
        for train_index, valid_index in loo.split(X):
            X_keep, y_keep = self.keep_inliers(X[train_index], y[train_index], factor=factor)
            X_train_scaled, X_valid_scaled = self.scale(X_keep, X[valid_index])
#             X_train_scaled = X_keep
#             X_valid_scaled = X[valid_index]
            try:
                model = self.fit(X_train_scaled, y_keep)
                if featureset not in self.models:
                    self.models[featureset] = []
                self.models[featureset].append(model)
                y_pred[valid_index] = model.predict(X_valid_scaled)
#                 y_pred[valid_index] = self.fit_predict(X_train_scaled, y_keep, X_valid_scaled)
            except Exception as e: 
                if print_failed_models is True:
                    print(f'Model creation for feature {featureset} failed:\n', e,'\n')
        return y_pred

    def fit_inliers1_p(self, featureset, factor=1.1):
        loo = LeaveOneOut()
        fs = FeatureSet(featureset)
        X = self.X(fs).to_numpy()
        y = self.y
        y_pred = np.zeros(y.shape)
        for train_index, valid_index in loo.split(X):
            X_keep, y_keep = self.keep_inliers(X[train_index], y[train_index], factor=factor)
            X_train_scaled, X_valid_scaled = self.scale(X_keep, X[valid_index])
            try:
                y_pred[valid_index] = self.fit_predict_p(X_train_scaled, y_keep, X_valid_scaled)
            except: pass
        return y_pred

    def fit_inliers(self, featuresets, factor=1.1, results=None):
        if results is None:
            results = Results(self)
        for f in featuresets:
            y_pred = self.fit_inliers1(f, factor=factor)
            fs = FeatureSet(f)
            results.score(fs, self.y, y_pred)
        return results

    def fit_inliers_x(self, featuresets, factor=1.1, results=None):
        x = []
        for f in featuresets:
            y_pred = self.fit_inliers1(f, factor=factor)
            x.append(y_pred)
        return np.vstack(x).transpose()

    def fit_inliers_x_p(self, featuresets, factor=1.1, results=None):
        x = []
        for f in featuresets:
            y_pred = self.fit_inliers1_p(f, factor=factor)
            x.append(y_pred)
        return np.vstack(x).transpose()

    def fit_inliers_ensemble(self, featureset, factor=1.1, results=None, name=None, print_failed_models = False):
        if results is None:
            results = Results(self)
        if name is None:
            name = str(FeatureSet(featureset))
        y_pred = [ self.fit_inliers1(f, factor=factor, print_failed_models = print_failed_models) for f in featureset ]
        y_pred = np.array([ max( y ) for y in zip(*y_pred) ])
        results.score(name, self.y, y_pred)
        return results
    
    def fit_inliers_factor_ensemble(self, featureset, results=None, name=None):
        if results is None:
            results = Results(self)
        if name is None:
            name = str(FeatureSet(featureset))
        y_pred = [ self.fit_inliers1(feature, factor=factor) for feature, factor in featureset.items() ]
        y_pred = np.array([ max( y ) for y in zip(*y_pred) ])
        results.score(name, self.y, y_pred)
        return results
    
    def fit_inliers_auto_ensemble_auto_factor(self, step=0.05, import_from=None, plot_results=False):
        """
        Fits an auto ensemble for every 0.05 factor between 1.0 and 2.0.
        Returns the ensemble with the lowest amount of false negatives.
        """
        features = self.cols
        
        f_results = {}
        new_step = int(step * 100)
        if import_from is None:
            for i in range(100, 205 ,new_step):
                f = i/100
                results = self.fit_inliers_auto_ensemble(features, factor=f, print_featureset=False)
                f_results[f] = {'fn': len(results.get_false_negatives('Auto ensemble')), 'results': results}
        else:
            for i in range(100, 205 ,new_step):
                f = i/100
                try:
                    results = Results.load(f'{import_from}/results_f{f}.pickle')
                    f_results[f] = {'fn': len(results.get_false_negatives('Auto ensemble')), 'results': results}
                    print(f'{import_from}/results_f{f}.pickle file loaded')
                except:
                    print(f'{import_from}/results_f{f}.pickle file not found')
                    
        min_fn = min(values['fn'] for values in f_results.values())
        keys_min_fn = [k for k in f_results.keys() if f_results[k]['fn'] == min_fn]
        
        if plot_results == True:
            ortho_plot.plot_results(f_results)
            
        return keys_min_fn[-1], f_results[keys_min_fn[-1]]['results']
    
    def fit_inliers_auto_factor_ensemble(self, 
                                  features, 
                                  results=None, 
                                  name=None, 
                                  thresholds = (0.0, 1.0, 0.0), 
                                  show_progress = True,
                                  print_featureset = True):
        if results is None:
            results = Results(self)
        auto_ensemble = {}
        excluded = {}
        
        if show_progress is True:
            _features = log_progress(features, name=f'Features')
        else:
            _features = features
        
        for fs in _features:
            name = fs
            f_results = {}
            
            for i in range(100, 205 ,5):
                f = i/100
                _results = self.fit_inliers_ensemble([fs], factor=f)
                
                if thresholds is not None:
                    if _results.precision(name) < thresholds[1]:
                        excluded[name] = _results
                    elif _results.recall(name) < thresholds[0]:
                        excluded[name] = _results
                    elif _results.accuracy(name) < thresholds[2]:
                        excluded[name] = _results
                    else:
                        f_results[f] = {'tp': len(_results.get_true_positives(name)), 'results': _results}
                else:
                    f_results[f] = {'tp': len(_results.get_true_positives(name)), 'results': _results}
            if len(f_results) == 0:
                continue

            max_tp = max(values['tp'] for values in f_results.values())
            keys_max_tp = [k for k in f_results.keys() if f_results[k]['tp'] == max_tp]
            
            results = self.fit_inliers_ensemble([fs], results=results, factor=keys_max_tp[0])
            auto_ensemble[fs] = keys_max_tp[0]
                
        results.auto_factor_ensemble = auto_ensemble

        if print_featureset is True:
            print (f'Auto ensemble ({len(auto_ensemble)}/{len(features)}): [\n\'' + '\',\n\''.join(auto_ensemble.keys()) + ']')
        
        return self.fit_inliers_factor_ensemble(auto_ensemble, results = results, name = 'Auto ensemble')
    
    def fit_inliers_auto_ensemble(self, 
                                  features, 
                                  factor=1.1, 
                                  results=None, 
                                  name=None, 
                                  thresholds = (0.0, 1.0, 0.0), 
                                  show_progress = True,
                                  print_featureset = True, 
                                  print_failed_models = False):
        if results is None:
            results = Results(self)
        auto_ensemble = []
        excluded = {}
        
        if show_progress is True:
            _features = log_progress(features, name=f'Features factor {factor}')
        else:
            _features = features
        
        for fs in _features:
            name = fs
            _results = self.fit_inliers_ensemble([fs], factor=factor, print_failed_models = print_failed_models)
            
            if thresholds is not None:
                if _results.precision(name) < thresholds[1]:
                    excluded[name] = _results
                elif _results.recall(name) < thresholds[0]:
                    excluded[name] = _results
                elif _results.accuracy(name) < thresholds[2]:
                    excluded[name] = _results
                else:
                    results = self.fit_inliers_ensemble([fs], results=results, factor=factor, print_failed_models = print_failed_models)
                    auto_ensemble.append(fs)
            else:
                results = self.fit_inliers_ensemble([fs], results=results, factor=factor, print_failed_models = print_failed_models)
                auto_ensemble.append(fs)
        
        auto_ensemble = set(auto_ensemble)
        results.auto_ensemble = list(auto_ensemble)
#         features = set(features)
        if print_featureset is True:
            print (f'Auto ensemble ({len(auto_ensemble)}/{len(features)}): [\n\'' + '\',\n\''.join(auto_ensemble) + ']')
        
        return self.fit_inliers_ensemble(auto_ensemble, factor = factor, results = results, name = 'Auto ensemble', print_failed_models = print_failed_models)

    def keep_focussing(self, featureset, p=0.5):
        results = Results(self)
        results_gt = Results(self)
        ids = self.exercises.patients
        fs = FeatureSet(featureset)
        X = self.X(fs).to_numpy()
        keep = np.ones(self.y.shape, dtype=bool)
        y = self.y
        for i in range(5):
            y_pred = self.fit_loo_p(X[keep], y[keep])
            keep1 = np.zeros(self.y.shape, dtype=bool)
            keep1[keep] = ((y[keep] == 0)|(y_pred >= p))
            dont_keep = 1 - keep
            X_train, X_valid = self.scale(X[keep], X[dont_keep])
            y_pred = self.fit_predict_p(X_train, y[keep], X_valid)
            keep1[dont_keep] = ((y[dont_keep] == 0)|(y_pred >= p))
            keep = keep1
            if sum(y[keep]) == 0:
                return None
            #print(y[keep])
        return keep

    def keep_alternating(self, *features, p=0.5):
        assert len(features) == 2
        results = Results(self)
        results_gt = Results(self)
        ids = self.exercises.patients
        fs = [ FeatureSet(f) for f in features ]
        Xs = [ self.X(f).to_numpy() for f in fs ]
        keep = np.ones(self.y.shape, dtype=bool)
        y = self.y
        for i in range(10):
            X = Xs[i % 2]
            y_pred = self.fit_loo_p(X[keep], y[keep])
            keep1 = np.zeros(self.y.shape, dtype=bool)
            keep1[keep] = ((y[keep] == 0)|(y_pred >= p))
            dont_keep = 1 - keep
            X_train, X_valid = self.scale(X[keep], X[dont_keep])
            y_pred = self.fit_predict_p(X_train, y[keep], X_valid)
            keep1[dont_keep] = ((y[dont_keep] == 0)|(y_pred >= p))
            keep = keep1
            #print(y[keep])
        return keep

    def fit_alternating2(self, f1, f2, p=0.5):
        keep = self.keep_alternating(f1, f2, p=p)
        dont_keep = 1 - keep
        fs = FeatureSet([f1, f2])
        X = self.X(fs).to_numpy()
        y_pred = np.zeros(self.y.shape)
        X_keep = X[keep]
        y_keep = self.y[keep]
        y_pred[keep] = self.fit_loo(X_keep, y_keep)
        X_train, X_valid = self.scale(X_keep, X[dont_keep])
        y_pred[dont_keep] = self.fit_predict(X_train, self.y[keep], X_valid)
        return y_pred

    def fit_focussing1(self, featureset, p=0.5):
        keep = self.keep_focussing(featureset, p=p)
        if keep is None:
            return np.zeros(self.y.shape)
        dont_keep = 1 - keep
        fs = FeatureSet(featureset)
        X = self.X(fs).to_numpy()
        y_pred = np.zeros(self.y.shape)
        X_keep = X[keep]
        y_keep = self.y[keep]
        if sum(y_keep) < 2:
            return np.zeros(self.y.shape)
        y_pred[keep] = self.fit_loo(X_keep, y_keep)
        X_train, X_valid = self.scale(X_keep, X[dont_keep])
        y_pred[dont_keep] = self.fit_predict(X_train, self.y[keep], X_valid)
        return y_pred

    def fit_alternating(self, featuresets, p=0.5, results=None):
        if results is None:
            results = Results(self)
        y_pred = [ self.fit_alternating2(f1, f2, p=p) for f1, f2 in featuresets ]
        y_pred = np.array([ max( y ) for y in zip(*y_pred) ])
        results.score('ensemble', self.y, y_pred)
        return results

    def fit_focussing(self, featuresets, p=0.5, results=None):
        if results is None:
            results = Results(self)
        for f in featuresets:
            y_pred = self.fit_focussing1(f, p=p)
            fs = FeatureSet(f)
            results.score(fs, self.y, y_pred)
        return results

    def fit_focussing_ensemble(self, featuresets, p=0.5, results=None):
        if results is None:
            results = Results(self)
        y_pred = [ self.fit_focussing1(f, p=p) for f in featuresets ]
        y_pred = np.array([ max( y ) for y in zip(*y_pred) ])
        results.score('ensemble', self.y, y_pred)
        return results
    
    
    
    
class FeatureSet(list):
    def __init__(self, f):
        if type(f) is str:
            self.append(f)
        else:
            self.extend(set(f))

    def __str__(self):
        return ','.join(self)    
        
class FilesCategory:
    """
    lees alle oefeningen per patient uit een directory van een categorie.
    """
    
    def __init__(self, path):
        self.exercise_types = ['AB', 'AF', 'EL', 'EH', 'RF', 'overig']
        self.path = path
        if not self.path.endswith('/'):
            self.path = self.path + '/'
        self.category = self.path.split('/')[-2]
        self.patients = []
        
        # iterate patients
        for patient in os.listdir(self.path):
            if os.path.isdir(self.path + patient): 
                tmp_patient_path = self.path + patient
                tmp_file_dict = {self.exercise_types[0] : [], \
                               self.exercise_types[1] : [], \
                               self.exercise_types[2] : [], \
                               self.exercise_types[3] : [], \
                               self.exercise_types[4] : [], \
                               self.exercise_types[5] : [] }
                
                # iterate files. Sort exercises in tmpfiledict
                for exercise in os.listdir(tmp_patient_path):
                    tmp_exercise_path = tmp_patient_path + '/' +  exercise
                    if os.path.isfile(tmp_exercise_path) and tmp_exercise_path.endswith('.txt'):
                        """
                        kijk enkel naar de gemeenschappelijke oefeningen, sla de rest over.
                        """
                        if exercise.startswith(self.exercise_types[0]) or exercise.startswith(self.exercise_types[1]) or \
                           exercise.startswith(self.exercise_types[2]) or exercise.startswith(self.exercise_types[3]) or \
                           exercise.startswith(self.exercise_types[4]):                            
                            tmp_file_dict[exercise[:2]].append(exercise)
                        else:
                            tmp_file_dict[self.exercise_types[-1]].append(exercise)
                
                self.patients.append({'patient' : int(patient), 'files': tmp_file_dict})
        
    def get_patient_ids(self):
        """
        return een lijst met patiend ID's als int
        """
        patient_ids = []
        for p in self.patients:
            patient_ids.append(p['patient'])
        return patient_ids

    def get_exercises(self, pat_id, ex_type=''):
        """
        return een lijst met oefeningen van een patient met gegeven ID, en oeveningtype (self.exercisetypes)
        als oeftype leeg is, geef dan alle oefeningen behalve uit de categorie 'overig'
        """
        for p in self.patients:
            if p['patient'] == pat_id:
                if len(ex_type) == 0:
                    exercises = []
                    for x in self.exercise_types[:-1]:  
                        exercises = exercises + p['files'][x]
                    return exercises
                else:
                    return p['files'][ex_type]
        return []

    def fullpath(self, pat_id = None, exercise = None, ex_type=None):
        """
        PatID = sting: patient ID
        exercise = string: vollegige naam van een oefening dus bv "AB1.txt"
        exertype = string: type oefeving dus bv 'AB'. Wordt genegeerd als exercise != None
        """
        if pat_id is None:
            r = [self.fullpath(p, exercise, ex_type) for p in self.get_patient_ids()]
            if type(r[0]) is list:
                return list(itertools.chain(*r))
            else:
                return r
        
        if exercise is None:
            if ex_type is None:
                return [self.fullpath(pat_id, e, ex_type) for e in self.get_exercises(pat_id)]
            else:
                return [self.fullpath(pat_id, e, ex_type) for e in self.get_exercises(pat_id, ex_type)]
        
        tmp = self.path + str(pat_id) + '/' + exercise
        if os.path.isfile(tmp): 
            return tmp
        return ""
    
    def all_exercises(self):
        """
        laad alle exercises 1.txt
        idee: 2.txt houden als validatie set?
        """
        for patient in self.get_patient_ids():
            for ex_type in self.exercise_types[:-1]:
                for exercise in self.get_exercises(patient, ex_type):
                    if len(exercise) == 7 and '1.txt' in exercise:
                        yield self.fullpath(patient, exercise, ex_type), self.category, ex_type, patient 
            
    @staticmethod
    def from_categories(*categories):
        return [ f for category in categories for f in FilesCategory(f'transformed_data/{category}').all_exercises() ]

    
class Frame(dict):
    """
    klasse om frames uit een oefening op te slaan
    """
    
    def __init__(self, frame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ruwe x, y, x coordinaten
        self['x'] = []
        self['y'] = []
        self['z'] = []

        # welke sensor en welk frame de opname is
        self['sensor'] = []
        self['frame'] = []
        self.Frame = frame
    
    def contains(self, sensor):
        return sensor in self['sensor']
    
    def add_position(self, sensor, x, y, z):
        """
        voegt de eerste rij met sensornr en ruwe x,y,x coordinaten toe aan een frame
        """
        self['x'].append(float(x))
        self['y'].append(float(y))
        self['z'].append(float(z))
        self['sensor'].append(sensor)
        self['frame'].append(self.Frame)

    def to_df(self):
        """
        zet de gelezen data om naar een dataframe 
        """
        return pd.DataFrame.from_dict(self)
    
    
class FrameRotation(Frame):
    """
    klasse om frames incl rotaties uit een oefening op te slaan
    """
    
    def __init__(self, frame, *args, **kwargs):
        super().__init__(frame, *args, **kwargs)
        # rotatie matrix 
        self['x0'] = []
        self['x1'] = []
        self['x2'] = []

        self['y0'] = []
        self['y1'] = []
        self['y2'] = []
        
        self['z0'] = []
        self['z1'] = []
        self['z2'] = []

    def add_rotation(self, x, y, z):
        """
        voegt een rij aan de rotatie matrix toe, de matrix staat in rijen in de textfile
        dus de eerste rij komt in x0,y0,z0, de tweede in x1,y,z1 etc.
        """
        if len(self['x0']) < len(self['x']):
            self['x0'].append(float(x))
            self['y0'].append(float(y))
            self['z0'].append(float(z))
        elif len(self['x1']) < len(self['x0']):
            self['x1'].append(float(x))
            self['y1'].append(float(y))
            self['z1'].append(float(z))
        elif len(self['x2']) < len(self['x1']):
            self['x2'].append(float(x))
            self['y2'].append(float(y))
            self['z2'].append(float(z))

            
class Patient(dict):
    """
    A Patient dictionary
    Used to construct a dataframe with the features AB, AF, EL, RF, which correspond with important exercise shorthands.
    """

    fs = {
        'angle_left_shoulder_xz': {'operations': ['max'], 'exercises': ['AF', 'RF']},
        'angle_right_shoulder_xz': {'operations': ['max'], 'exercises': ['AF', 'RF']},
        'angle_left_shoulder_yz': {'operations': ['max'], 'exercises': ['AB']},
        'angle_right_shoulder_yz': {'operations': ['max'], 'exercises': ['AB']},
#         'lowest_max_shoulder_angle_xz': {'operations': ['max'], 'exercises': ['AF', 'RF']},
#         'lowest_max_shoulder_angle_yz': {'operations': ['max'], 'exercises': ['AB']},

        # symmetry x, left right
        'diff_x_wrist': {'operations': ['std'], 'exercises': ['EL', 'AF', 'RF']},
        'diff_x_elbow': {'operations': ['std'], 'exercises': ['EL', 'AF', 'RF']},
        'diff_x_shoulder': {'operations': ['std'], 'exercises': ['EL', 'AF', 'RF']},

        # symmetry y, left right
        'diff_y_wrist': {'operations': ['std'], 'exercises': ['EL', 'AF', 'RF', 'AB']},
        'diff_y_elbow': {'operations': ['std'], 'exercises': ['AB']},
        
        # symmetry z, left right
        'diff_z_wrist': {'operations': ['std'], 'exercises': ['AB', 'AF', 'RF']},
        'diff_z_elbow': {'operations': ['std'], 'exercises': ['AB', 'AF', 'RF']},
        'diff_z_shoulder': {'operations': ['std'], 'exercises': ['AB', 'AF', 'RF']},

        'z_elbow': {'operations': ['max'], 'exercises': ['AB', 'AF', 'RF']},
        'z_wrist': {'operations': ['max'], 'exercises': ['AB', 'AF', 'RF']},
        'x_wrist': {'operations': ['max'], 'exercises': ['EL']},
        
        # velocities wrists x z, left right
        'vel_wrists_x_l': {'operations': ['std'], 'exercises': ['EL']},
        'vel_wrists_x_r': {'operations': ['std'], 'exercises': ['EL']},
        
        # velocties elbows z, left right
        'vel_elbows_z_l': {'operations': ['std'], 'exercises': ['AF', 'RF', 'AB']},
        'vel_elbows_z_r': {'operations': ['std'], 'exercises': ['AF', 'RF', 'AB']},

        # accelerations wrists x z, left right
        'acc_wrists_x_l': {'operations': ['mean', 'std'], 'exercises': ['EL']},
        'acc_wrists_x_r': {'operations': ['mean', 'std'], 'exercises': ['EL']},
        
        # accelerations elbows z, left right
        'acc_elbows_z_l': {'operations': ['mean', 'std'], 'exercises': ['AF', 'RF', 'AB']},
        'acc_elbows_z_r': {'operations': ['mean', 'std'], 'exercises': ['AF', 'RF', 'AB']},

        # angular velocity en acceleration elbow, left right
        'angular_vel_xz_elbow_l': {'operations': ['std'], 'exercises': ['AF', 'RF']},
        'angular_vel_xz_elbow_r': {'operations': ['std'], 'exercises': ['AF', 'RF']},
        'angular_acc_xz_elbow_l': {'operations': ['mean', 'std'], 'exercises': ['AF', 'RF']},
        'angular_acc_xz_elbow_r': {'operations': ['mean', 'std'], 'exercises': ['AF', 'RF']},
        
        'angular_vel_yz_elbow_l': {'operations': ['std'], 'exercises': ['AB']},
        'angular_vel_yz_elbow_r': {'operations': ['std'], 'exercises': ['AB']},
        'angular_acc_yz_elbow_l': {'operations': ['mean', 'std'], 'exercises': ['AB']},
        'angular_acc_yz_elbow_r': {'operations': ['mean', 'std'], 'exercises': ['AB']},
    }
    

    def remap(self, series):

        def transform_name(x):
            if not isinstance(x,str):
                return x

            split = x.split('_')
            if len(split) > 2:
                return split[0] + '_' + '_'.join(split[1:])
            else:
                return split[0] + '_' + '_'.join(split[1:])

        mapper = { x: transform_name(x) for x in series.index }
        return series.rename(mapper)

    def to_df(self):
        dfs = []
        
        for f, opts in self.fs.items():
            for extype in opts['exercises']:
                if len(opts['operations']) > 0:
                    for op in opts['operations']:
                        if extype in self:
                            if f in self[extype].df.columns:
                                if hasattr(self[extype].df[[f]], op):
                                    method = getattr(self[extype].df[[f]], op)
                                    dfs.append(method().add_suffix(f'_{op}_{extype}'))
                                else:
                                    print('skipping', f, extype, 'because invalid operation', op)
                            else:
                                print('skipping', f, extype, 'because', f, 'not in', self[extype].df.columns)
                        else:
                            print('skipping', f, 'because', extype, 'not in', self.keys())
                elif extype in self:
                    if f in self[extype].df.columns:
                        dfs.append(self[extype].df[[f]].add_suffix(f'_{extype}'))
                    else:
                        print('skipping', f, 'because', f, 'not in', self[extype].df.columns)
                else:
                    print('skipping', f, 'because', extype, 'not in', self.keys())
        series = pd.concat(dfs, axis=0)
        return self.remap(series)

    def to_series(self):
        dfs = []
        
        for f, opts in self.fs.items():
            for extype in opts['exercises']:
                if len(opts['operations']) > 0:
                    for op in opts['operations']:
                        if extype in self:
                            if f in self[extype].df.columns:
                                if hasattr(self[extype].df[[f]], op):
                                    method = getattr(self[extype].df[[f]], op)
                                    dfs.append(method().add_suffix(f'_{op}_{extype}'))
                                else:
                                    print('skipping', f, extype, 'because invalid operation', op)
                            else:
                                print('skipping', f, extype, 'because', f, 'not in', self[extype].df.columns)
                        else:
                            print('skipping', f, 'because', extype, 'not in', self.keys())
                elif extype in self:
                    if f in self[extype].df.columns:
                        if not isinstance(f, int):
                            a = self[extype].df[[f]].to_numpy()
                            if self[extype].df[[f]].values.shape[0] == 1 or (a[0] == a).all(0):
                                dfs.append(self[extype].df[[f]].add_suffix(f'_{extype}'))
                            else:
                                print('skipping', f, 'because no operation is given, and the series can\'t be aggregated.')
                        else:
                            print('skipping', f, 'because it is an integer')
                    else:
                        print('skipping', f, 'because', f, 'not in', self[extype].df.columns)
                else:
                    print('skipping', f, 'because', extype, 'not in', self.keys())
        series = pd.concat(dfs, axis=0)
        return series
    
    def to_df_min(self):
        dfmin = [ self[extype].df[[f]].min().add_suffix(f'_min_{extype}') for extype, fs in self.features.items() for f in fs ]
        return self.remap(pd.concat( dfmin, axis=0 ))
    
    def to_df_max(self):
        dfmin = [ self[extype].df[[f]].max().add_suffix(f'_ax_{extype}') for extype, fs in self.features.items() for f in fs ]
        return self.remap(pd.concat( dfmin, axis=0 ))         

    @classmethod
    def add_features(cls, features, exercises = [], operations = []):
        if isinstance(features,dict):
            cls.fs.update(exercises)
            return

        for op in operations:
            if op not in ['min', 'max', 'std', 'mean', 'var', 'prod']:
                raise Exception(f'Invalid operation {op}, must be one of: ' + ', '.join(['min', 'max', 'std', 'mean', 'var', 'prod']))

        if type(features) == str:
            features = [features]
        for f in features:
            if f not in cls.fs:
                cls.fs[f] = {'operations': [], 'exercises': []}
            cls.fs[f]['operations'] = cls.fs[f]['operations'] + list(set(operations) - set(cls.fs[f]['operations']))
            cls.fs[f]['exercises'] = cls.fs[f]['exercises'] + list(set(exercises) - set(cls.fs[f]['exercises']))
    
    
class Results(dict):
    def __init__(self, experiment):
        self.experiment = experiment
        self.exercises = experiment.exercises
        self.func = self.positives
        self.auto_ensemble = []
        self.auto_factor_ensemble = {}
    
    def positives(self, x):
        return x['tp'] + x['fp']
    
    def check_patients(self, patients):
        features = self.analyze_ensemble_features()
        
        features_found = []
        
        for feature, f_patients in features.items():
            if all(elem in f_patients  for elem in patients):
                features_found.append(feature)
        return features_found
    
    def analyze_tps(self):
        counts = {}
        
        features = self.get_true_positives()
        for pat_id, f in features:
            if pat_id not in counts:
                counts[pat_id] = []
            counts[pat_id].append(f)
            
        sorted_counts = {k: v for k, v in sorted(counts.items(), key=lambda item: len(item[1]))}
        return sorted_counts
    
    def analyze_ensemble_features(self):
        counts = {}
        
        features = self.get_true_positives()
        for pat_id, f in features:
            if f != 'Auto ensemble':
                if f not in counts:
                    counts[f] = []
                counts[f].append(pat_id)
        sorted_counts = {k: v for k, v in sorted(counts.items(), key=lambda item: len(item[1]))}
        return sorted_counts
    
    def count_ensemble_features(self):
        features = self.analyze_ensemble_features()
        count_features = {feature: len(patients) for feature, patients in features.items()}
        sorted_count_features = {k: v for k, v in sorted(count_features.items(), key=lambda item: item[1], reverse=True)}
        return sorted_count_features
    
    def least_obvious(self, tps):
        count_list = [len(v) for v in tps.values()]
        min_count = min(count_list)
        least_obvious = [p for p, l in tps.items() if len(l) == min_count]
        
        for lo in log_progress(least_obvious):
            print(f'features selected for {lo}:', tps[lo])
        return least_obvious
    
    def most_obvious(self, tps):
        count_list = [len(v) for v in tps.values()]
        max_count = max(count_list)
        most_obvious = [p for p, l in tps.items() if len(l) == max_count]
        
        for mo in log_progress(most_obvious):
            print(f'features selected for {mo}:', tps[mo])
        return most_obvious

    def store(self, p_id, feature, tp, tn, fp, fn):
        self[(p_id, feature)] = {'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn }
       
    def score(self, fs, y_valid, y_pred):
        tp = ((y_valid == 1) & (y_pred == 1)) * 1.0
        tn = ((y_valid == 0) & (y_pred == 0)) * 1.0
        fp = ((y_valid == 0) & (y_pred == 1)) * 1.0
        fn = ((y_valid == 1) & (y_pred == 0)) * 1.0
        for id, *p in zip(self.exercises.patients, tp, tn, fp, fn):
            self.store( id, str(fs), *p)

    @property
    def ids(self):
        try:
            return self._ids
        except:
            self._ids = self.exercises.patients
            return self._ids

    def feature(self, f, func=None):
        if func is None:
            func = self.func
        return [ func(self[(i, f)]) for i in self.ids ]
    
    def feature_df(self, f):
        return pd.DataFrame(self.feature(f))

    @property
    def columns(self):
        try:
            return self._columns
        except:
            self._columns = list({ c for i, c in self.keys() })
            return self._columns

    def accuracy(self, f):
        r = self.feature(f, func = lambda x: x['tp'] + x['tn'])
#         if len(r) == 0 or math.isnan(len(r)):
#             return 0
        return sum(r) / len(r)
    
    def recall(self, f):
        tpfn = self.feature(f, func = lambda x: x['tp'] + x['fn'])
#         if math.isnan(tpfn) or tpfn == 0:
#             return 0
        return sum(self.feature(f, func = lambda x: x['tp'])) / sum(self.feature(f, func = lambda x: x['tp'] + x['fn']))
    
    def f1(self, f):
        precision  = self.precision(f)
        recall  = self.recall(f)
        if precision == 0 and recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def precision(self, f):
        tpfp = self.feature(f, func = lambda x: x['tp'] + x['fp'])
        if math.isnan(sum(tpfp)) or sum(tpfp) == 0:
            return 0
        return sum(self.feature(f, func = lambda x: x['tp'])) / sum(self.feature(f, func = lambda x: x['tp'] + x['fp']))
   
    def report(self, sort=[], ascending=False, include_f1 = False):
        if type(sort) is str:
            sort = [sort]
        if include_f1 is True:
            r = [ [f, self.recall(f), self.precision(f), self.accuracy(f), self.f1(f)] for f in self.columns ]
            cols = ['features', 'recall', 'precision', 'accuracy', 'f1']
        else:
            r = [ [f, self.recall(f), self.precision(f), self.accuracy(f)] for f in self.columns ]
            cols = ['features', 'recall', 'precision', 'accuracy']
        df = pd.DataFrame( r, columns=cols )
        if len(sort) > 0:
            df = df.sort_values(*sort, ascending=ascending)
            df.reset_index(inplace=True, drop=True)
        return df
    
    def report_auto_factor(self, sort=[], ascending=False, include_f1 = False):
        self.auto_factor_ensemble['Auto ensemble'] = '-'
        if type(sort) is str:
            sort = [sort]
        if include_f1 is True:
            r = [ [f, self.recall(f), self.precision(f), self.accuracy(f), self.f1(f)] for f in self.columns ]
            cols = ['features', 'recall', 'precision', 'accuracy', 'f1']
        else:
            r = [ [f, self.auto_factor_ensemble[f], self.recall(f), self.precision(f), self.accuracy(f)] for f in self.columns ]
            cols = ['features', 'factor', 'recall', 'precision', 'accuracy']
        df = pd.DataFrame( r, columns=cols )
        if len(sort) > 0:
            df = df.sort_values(*sort, ascending=ascending)
            df.reset_index(inplace=True, drop=True)
        return df


    def cross_cronbach(self, cols = None):
        if cols is None:
            cols = self.columns
        res = []
        for c1 in cols:
            for c2 in cols:
                r = self.cronbach(c1, c2)
                acc1 = f'{self.accuracy(c1):0.2f}'
                acc2 = f'{self.accuracy(c2):0.2f}'
                res.append( [ len(res), c1 + acc1, c2 + acc2, r])
        return pd.DataFrame(res).pivot(index=1, columns=2, values=3)
   
    def heatmap(self, cols = None, figsize = (10,10), annot = True):
        plt.figure(figsize = figsize)
        sns.heatmap(self.cross_cronbach(cols), annot = annot )


    def heatmap_plotly(self, cols = None): 
        df = self.cross_cronbach(cols)

        fig = go.Figure(data=go.Heatmap(
                   z=df.values,
                   x=df.columns.to_numpy(),
                   y=df.index.to_numpy(),
                   hoverongaps = False))
        fig.show()

    def feature_vector(self, f = lambda x: x['tp'] + x['fp']):
        cols = self.columns
        ids = { i for i, c in self.keys() }
        data = [ [ f(self[(i, c)]) for i in ids ] for c in cols ]
        data = np.array(data)
        return data

    def tsne(self, perplexity=2, figsize=(10,10)):
        X = self.feature_vector()
        labels = self.columns
        X_embedded = TSNE(n_components=2, perplexity=perplexity).fit_transform(X)
        plt.figure(figsize=figsize)
        for i, label_id in enumerate(labels):
            X_label = X_embedded[i]
            plt.plot(X_label[0],
                        X_label[1],
                        marker='o',
                        linewidth='1',
                        alpha=0.8)
            plt.text(X_label[0]+10,
                        X_label[1]+10, label_id)
#         return fig

    def result_dist(self, label, figsize=(10,1), out=''):
        tp = []
        tn = []
        fp = []
        fn = []
        ids = self.ids
        df = self.experiment.df
        for (id, l), v in self.items():
            if l == label:
                f = df[label][ids.index(id)]
                if v['tp']:
                    tp.append(f)
                if v['fp']:
                    fp.append(f)
                if v['tn']:
                    tn.append(f)
                if v['fn']:
                    fn.append(f)
        plt.figure(figsize=figsize)
        plt.scatter(tp,
                 [1] * len(tp),
                 marker='o', color='g')
        plt.scatter(tn,
                 [0] * len(tn),
                 marker='o', color='r')
        plt.scatter(fp,
                 [0] * len(fp),
                 marker='o', color='g')
        plt.scatter(fn,
                 [1] * len(fn),
                 marker='o', color='r')
        plt.ylabel('y')
        plt.xlabel(label)
        if out != '':
            plt.tight_layout()
            plt.savefig(out)
        plt.show()
        return plt

    def cronbach(self, f1, f2):
        df = pd.concat([self.feature_df(f1), self.feature_df(f2)], axis=1)
        df_corr = df.corr()
        N = df.shape[1]
        rs = np.array([])
        for i, col in enumerate(df_corr.columns):
            sum_ = df_corr[col][i+1:].values
            rs = np.append(sum_, rs)
            mean_r = np.mean(rs)
        cronbach_alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
        return cronbach_alpha
    
    def get_true_positives(self, feature = None):
        return [(p_id, f) for (p_id, f), confusion in self.items() if confusion['tp'] == 1 and (f == feature or feature == None)]
        
    def get_true_negatives(self, feature = None):
        return [(p_id, f) for (p_id, f), confusion in self.items() if confusion['tn'] == 1 and (f == feature or feature == None)]
    
    def get_false_positives(self, feature = None):
        return [(p_id, f) for (p_id, f), confusion in self.items() if confusion['fp'] == 1 and (f == feature or feature == None)]
        
    def get_false_negatives(self, feature = None):
        return [(p_id, f) for (p_id, f), confusion in self.items() if confusion['fn'] == 1 and (f == feature or feature == None)]
    
    def confusion_matrix(self, feature='Auto ensemble'):
        print(f"""
        total patients for feature '{feature}': {len(self.ids)} (r: {round(self.recall(feature),3)}, p: {round(self.precision(feature), 3)}, a: {round(self.accuracy(feature), 3)}, f1: {round(self.f1(feature),3)})
        true positives: {len(self.get_true_positives(feature))}
        true negatives: {len(self.get_true_negatives(feature))}
        false positives: {len(self.get_false_positives(feature))}
        false negatives: {len(self.get_false_negatives(feature))}
        """)
        
    def dump(self, file):
        with open(file, 'wb') as fout:
            pickle.dump(self, fout)
   
    @staticmethod
    def load(file):
        with open(file, 'rb') as fin:
            return pickle.load(fin)
    
def angle(df, s1, s2, s3):
    return angle4(df, s1, s2, s3, s2)

def angle_AB(df, s1, s2):
    y1 = df['y_' + s1]
    z1 = df['z_' + s1]
    y2 = df['y_' + s2]
    z2 = df['z_' + s2]
    l_arm = ((y1-y2)**2+(z1-z2)**2)**0.5
    l_3 = abs(z1-z2)
    
    angle = []
    for index, row in df.iterrows():
        if z1[index] > z2[index]:
            angle.append(np.arccos(l_3[index] / l_arm[index]))
        elif z1[index] == z2[index]:
            angle.append(90 * math.pi / 180)
        else:
            angle.append(math.pi - np.arccos(l_3[index] / l_arm[index]))
    return pd.Series(angle)

def angle_AF_RF(df, s1, s2):
    x1 = df['x_' + s1]
    z1 = df['z_' + s1]
    x2 = df['x_' + s2]
    z2 = df['z_' + s2]
    l_arm = ((x1-x2)**2+(z1-z2)**2)**0.5
    l_3 = abs(z1-z2)
    
    angle = []
    for index, row in df.iterrows():
        if z1[index] > z2[index]:
            angle.append(np.arccos(l_3[index] / l_arm[index]))
        elif z1[index] == z2[index]:
            angle.append(90 * math.pi / 180)
        else:
            angle.append(math.pi - np.arccos(l_3[index] / l_arm[index]))
    return pd.Series(angle)

def angle4(df, s1, s2, s3, s4):
    """ compute the angle between 2 3d vectors """
    x1, y1, z1 = vnorm(df['x_' + s1], df['y_' + s1], df['z_' + s1], df['x_' + s2], df['y_' + s2], df['z_' + s2])
    x2, y2, z2 = vnorm(df['x_' + s3], df['y_' + s3], df['z_' + s3], df['x_' + s4], df['y_' + s4], df['z_' + s4])
    res = x1 * x2 + y1 * y2 + z1 * z2
    angle = np.arccos(res)
    return 180 * angle / math.pi

def calculate_lr_diffs(df):
    """
    Calculate differences in left and right sides of the patient.
    Calculates x,y,z differences, as well as average xy,yz,xz difference and total average difference
    """
    df[['diff_x', 'diff_y', 'diff_z', 'diff_xz', 'diff_xy', 'diff_yz', 'diff']] = 0

    sensor_pairs = {
        '4': '7', # shoulders
        '5': '8', # elbows
        '6': '9', # wrists
        '7': '4', # shoulders
        '8': '5', # elbows
        '9': '6'  # wrists
    }

    # Calculate difference columns
    for i in sensor_pairs:
        s_values = df[df['sensor'] == i]
        
        df_sensor_compl = df[df['sensor'] == sensor_pairs[i]]
        
        # get complementary coords
        compl_x = df_sensor_compl.x.values
        compl_y = df_sensor_compl.y.values
        compl_z = df_sensor_compl.z.values

        s_values.loc[s_values.index,'diff_x'] = (s_values.x - compl_x).abs()
        s_values.loc[s_values.index,'diff_y'] = (s_values.y - 1 + compl_y).abs()
        s_values.loc[s_values.index,'diff_z'] = (s_values.z - compl_z).abs()
        s_values.loc[s_values.index,'diff_xz'] = (s_values.diff_x + s_values.diff_z).values / 2
        s_values.loc[s_values.index,'diff_xy'] = (s_values.diff_x + s_values.diff_y).values / 2
        s_values.loc[s_values.index,'diff_yz'] = (s_values.diff_y + s_values.diff_z).values / 2
        s_values.loc[s_values.index,'diff'] = (s_values.diff_x + s_values.diff_y + s_values.diff_z).values / 3

        df[df['sensor'] == i] = s_values
        
    return df

def create_dfframe(category, pat_id, exercise, data_dir = 'transformed_data', rotation= None, dfframe_class = DFFrame):
    """
    Creates an instance of a dataframe class, by default the DFFrame class, for 1 exercise from 1 patient from 1 category.
    If rotation = None, the rotation matrices will be excluded.
    If rotation != None, the rotation matrices will be included and the euler angles will be calculated.
    """
    if isinstance(category, int):
        category = 'Category_' + str(category)
        
    file = FilesCategory(os.path.join(data_dir, category)).fullpath(pat_id = pat_id, exercise = exercise)
    
    dfs = None
    if rotation is None:
        df = exercise_to_df(file)
        
        # split the dataframe in separate dataframes per sensor
        dfs = [ df_sensors(df, s) for s in '23456789' ] 
    else:
        df = exercise_to_df_with_rotation(file)
        df = get_df_with_euler(df, rotation)
                    
        # split the dataframe in separate dataframes per sensor
        dfs = [ df_rotation_sensors(df, s) for s in '23456789' ] 

    # define reduction function
    join2 = lambda l, r: pd.merge(l, r, on='frame')

    # merge the dataframes back to a single dataframe, with columns {x,y,z}_{sensor_num}
    merged = reduce( join2, dfs )
    
    # construct a DFFrame to calculate the arm, elbow and shoulder angles
    return dfframe_class(merged)

def create_dfframes(category = ['Category_1', 'Category_2', 'Category_3', 'Category_4'], 
                    pat_id = None, 
                    exercises = None, 
                    extype = 'AB', 
                    data_dir = 'sliced_transformed_data', 
                    dfframe_class = DFFrame, 
                    rotation = None,
                    show_progress = True, 
                    print_errors = True):
    """
    Create a pandas DataFrame for the given catergories.

    Specific patients and exercises can be selected with `pat_id` and `exercises`, both are None by default.
    Specific data directory can be set with `data_dir`, default `'transformed_data'`.
    Specific DataFrame class can be set with `dfframe_class`, default `DFFrame`.
    Filter out exercises you don't want with `extype`, can either be string or list of strings, defaul `'AB'`.

    Shows error messages and progress bar by default

    """
    # fix category names
    if isinstance(category, list):
        numbers = list(filter(lambda e: isinstance(e, int), category))
        if len(numbers) == 0:
            cats = category
        else:
            cats = ['Category_' + str(c) for c in category]
    elif isinstance(category, str):
        cats = [category]
    elif isinstance(category, int):
        cats = ['Category_' + str(category)]
    else:
        raise Exception('Invalid type for category argument')

    if isinstance(extype, str):
        extype = [extype]
    elif not isinstance(extype, list):
        raise Exception('Invalid type for category argument')
    
    files = [FilesCategory(os.path.join(data_dir, c)).fullpath(pat_id = pat_id, exercise = exercises) for c in cats]
    #flatten array
    files = [l for fs in files for l in fs]
    
    # log progress
    if show_progress == True:
        files = log_progress(files, every = 1, name = 'Files')
    
    dffs = {}
    
    p = r'(?:.*\/|)[\w_]+\/Category_(\d)\/(\d+)\/(\w+)\.txt'
    pattern = re.compile(p)

    for file in files:
        try:
            cat, pat_id, ex = pattern.match(file).groups()
            cat = str(cat)
            et = str(ex[0:2])
            if et in extype:

                dfs = None
                if rotation is None:
                    df = exercise_to_df(file)
                    
                    # split the dataframe in separate dataframes per sensor
                    dfs = [ df_sensors(df, s) for s in '23456789' ] 
                else:
                    df = exercise_to_df_with_rotation(file)
                    df = get_df_with_euler(df, rotation)
                    
                    # split the dataframe in separate dataframes per sensor
                    dfs = [ df_rotation_sensors(df, s) for s in '23456789' ] 
                
                # define reduction function
                join2 = lambda l, r: pd.merge(l, r, on='frame')

                # merge the dataframes back to a single dataframe, with columns {col}_{sensor_num}
                merged = reduce( join2, dfs )

                if dffs.get(cat) is None:
                    dffs[cat] = []
                    
                _extype = et
                try:
                    dffs[cat].append((int(pat_id), _extype, dfframe_class(merged)))
                except:
                    if print_errors is True:
                        print('Appending to ' + str(cat) + 'failed, dffs is now ', dffs)
                    dffs[cat] = []
                    dffs[cat].append((int(pat_id), _extype, dfframe_class(merged)))
            else:
                if print_errors is True:
                    print('Excluding ' + file + ', as it is not of type ' + extype)
        except Exception as e:
            if print_errors is True:
                print('Error while determining the values for category, patient or exercise in file ', file, type(e), e)
#                 exc_type, exc_obj, exc_tb = sys.exc_info()
# #                 fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#                 print(file, exc_tb.tb_lineno)
#     if show_progress == True:
#         Audio('snd/tring.mp3', autoplay=True)

    return dffs


        

def create_experiment(categories = [1,3], 
                      ex_type = 'AB', 
                      data_dir = 'transformed_data', 
                      dfframe_class = DFFrame):
    dffs = create_dfframes(categories, 
                           data_dir = data_dir, 
                           extype = ex_type, 
                           dfframe_class = dfframe_class, 
                           print_errors = False)
    exercises = dffs_to_exercises(dffs)
    return Experiment(exercises)

def df_rotation_sensors(df, s):
    """
    return dataframe with renamed x,y,z data for one sensor.
    x coördinate for sensor 2 becomes 'x_2'.
    Includes the 3 euler angles in radians.
    """
    return df[df.sensor == s][['frame', 'x', 'y', 'z', 'rad1', 'rad2', 'rad3']].rename(columns={'x':'x_' + s, 'y':'y_' + s,
                                                                                                'z':'z_' + s, 'rad1':'rad1_' + s, 
                                                                                                'rad2':'rad2_' + s, 'rad3':'rad3_' + s})

def df_sensors(df, s):
    """
    return dataframe with renamed x,y,z data for one sensor.
    x coördinate for sensor 2 becomes 'x_2'.
    Excludes the 3 euler angles.
    """
    return df[df.sensor == s][['frame', 'x', 'y', 'z']].rename(columns={'x':'x_' + s, 'y':'y_' + s, 'z':'z_' + s})

def dffs_to_exercises(dffs):
    if not isinstance(dffs, dict):
        raise Exception('dffs should be a dictionary')
        
    exs = []
    for cat in dffs:
        for pat_id, ex, dff in dffs[cat]:
            exs.append(Exercise(dff, 'Category_' + str(cat), ex, pat_id))
         
    exercises = Exercises.from_exercise_objects(exs)
    exercises = filter_incomplete_patients(exercises)
    
    return exercises

def diff(df, sl, sr):
    """
    Calculates the absolute difference between 2 sensors. Used for x and z-coördinates.
    """
    return abs(df[sl] - df[sr])

def diff_mirrored(df, sl, sr):
    """
    Calculates the mirrored absolute difference between 2 sensors. Used for y-coördinates.
    """
    return abs(df[sl] + df[sr])

def exercise_from_df(df, category, extype, patient):
    return Exercise(df, category, extype, patient)

def exercise_from_file( file, category, extype, patient):
    """
    Read the exercise file and return a dataframe.
    """
    df = exercise_to_df(file)
    dfs = [ df_sensors(df, s) for s in '23456789' ] 
    join2 = lambda l, r: pd.merge(l, r, on='frame')
    df = DFFrame(reduce( join2, dfs ))
    return Exercise(df, category, extype, patient)

def exercise_to_df(file, invert_z=False):
    """
    Converts a .txt exercise file to a pandas dataframe.
    Excludes the rotation matrices.
    """
    frames = []
    f = Frame(0)
    
    with open(file) as fin:
        for line in fin:
            words = line.split()
            try:
                sensor, x, y, z = words
                if f.contains(sensor):
                    frames.append(f.to_df())
                    f = Frame(len(frames))                    
                f.add_position(sensor, x, y, z)
            except: pass
                    
    frames.append(f.to_df())
    df = pd.concat(frames)
    if invert_z:
        df.z = -df.z
    return df

def exercise_to_df_with_rotation(file, invert_z=False):
    """
    Converts a .txt exercise file to a pandas dataframe.
    Includes the rotation matrices.
    """
    frames = []
    f = FrameRotation(0)
    
    with open(file) as fin:
        for line in fin:
            words = line.split()
            try:
                sensor, x, y, z = words
                if f.contains(sensor):
                    frames.append(f.to_df())
                    f = FrameRotation(len(frames))                    
                f.add_position(sensor, x, y, z)
            except: 
                try:
                    x, y, z = words
                    f.add_rotation(x ,y ,z)
                except: pass
                    
    frames.append(f.to_df())
    df = pd.concat(frames)
    if invert_z:
        df.z = -df.z
    return df

def filter_incomplete_patients(exercises, exercise_types = ['AF', 'AB', 'EL', 'RF']):
    """ 
    Drop patients who did not perform all exercise_types.
    Return only patients who did all exercise_types.
    """ 
    new_exercises = copy.deepcopy(exercises)
    for patient in exercises.patients:
        keys = exercises[patient].keys()
        for ex in exercise_types:
            if ex not in keys:
                new_exercises = new_exercises.drop_patient(patient)
                print('dropped ' + str(patient) + ' because there is (a) missing exercise(s): ', list(keys))

    return new_exercises

def get_acceleration(df, sensor):
    """ 
    Calculates and returns the acceleration of the given sensor (pass: dataframe and sensor).
    """ 
    velocity = get_velocity(df, sensor)
    acceleration = get_velocity(velocity, sensor)
    acceleration = acceleration.dropna()
    return acceleration

def get_velocity(df, sensor):
    """ 
    Calculates and returns the velocity of the given sensor (pass: dataframe and sensor).
    """ 
    velocity = df[[sensor]].diff(axis=0)
    velocity = velocity.dropna()
    return velocity

def get_df_with_euler(df, rotation):
    """ 
    Compute the euler angles in radians and degrees for all sensors and frames and add these to a new column.
    """
    df = df.reset_index(drop=True)
    for index, row in df.iterrows():
        r = R.from_matrix([[row['x0'], row['y0'], row['z0']],
                  [row['x1'], row['y1'], row['z1']],
                  [row['x2'], row['y2'], row['z2']]])
        angle1, angle2, angle3 = r.as_euler(rotation, degrees=True)
        rad1, rad2, rad3 = r.as_euler(rotation, degrees=False)
        
        df.loc[index, 'angle1'] = angle1
        df.loc[index, 'angle2'] = angle2
        df.loc[index, 'angle3'] = angle3
        df.loc[index, 'rad1'] = rad1
        df.loc[index, 'rad2'] = rad2
        df.loc[index, 'rad3'] = rad3
    return df

def log_progress(sequence, every=None, size=None, name='Items'):
    """
    https://github.com/kuk/log-progress.
    """
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

def highest_min_lr(df, sl, sr):
    """ 
    Compute the highest value between the min values of 2 sensors.
    """
    return max(np.min(df[sl]), np.min(df[sr]))
        
def lowest_max_lr(df, sl, sr):
    """ 
    Compute the lowest value between the max values of 2 sensors.
    """
    return min(np.max(df[sl]), np.max(df[sr]))

def merge_dffs(df1, *dfs):
    df = {}
    for df2 in dfs:
        for key, val in df1.items():
            if key in df2:
                df[key] = df1[key] + df2[key]
    return df

def pyth(df, s1, s2):
    return (df[s1]**2 + df[s2]**2)**0.5

def length_vector(df, s1, s2):
    return ((df['x_' + s1]-df['x_' + s2])**2 + (df['y_' + s1]-df['y_' + s2])**2 + (df['z_' + s1]-df['z_' + s2])**2)**0.5

def shoulder_at_max_elbow(df, shoulder, elbow):
    """ 
    Compute the shoulder height at the max elbow height. 
    """
    max_elbow = np.max(df[elbow])
    sh =  df[df[elbow] == max_elbow][shoulder]
    return sh

def vnorm(x1, y1, z1, x2, y2, z2):
    """ 
    Compute the normalized difference between two 3d vectors. 
    """
    vx = x1 - x2
    vy = y1 - y2
    vz = z1 - z2
    
    mag = np.sqrt(vx * vx + vy * vy + vz * vz)
    return vx / mag, vy / mag, vz / mag        