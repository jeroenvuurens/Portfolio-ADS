Patient.diff
```diff
- class patient(dict):
-     features = { 'AB':['left_shoulder', 'right_shoulder'], 'AF':['left_shoulder', 'right_shoulder'], 
-                  'EL':['left_arm', 'right_arm'], 'RF':['left_shoulder', 'right_shoulder']}
---
+ class Patient(dict):
+     """
+     A Patient dictionary
+     Used to construct a dataframe with the features AB, AF, EL, RF, which correspond with important exercise shorthands.
+     """
+ 
+     fs = {
+         'angle_left_shoulder_xz': {'operations': ['max'], 'exercises': ['AF', 'RF']},
+         'angle_right_shoulder_xz': {'operations': ['max'], 'exercises': ['AF', 'RF']},
+         'angle_left_shoulder_yz': {'operations': ['max'], 'exercises': ['AB']},
+         'angle_right_shoulder_yz': {'operations': ['max'], 'exercises': ['AB']},
+ #         'lowest_max_shoulder_angle_xz': {'operations': ['max'], 'exercises': ['AF', 'RF']},
+ #         'lowest_max_shoulder_angle_yz': {'operations': ['max'], 'exercises': ['AB']},
+ 
+         # symmetry x, left right
+         'diff_x_wrist': {'operations': ['std'], 'exercises': ['EL', 'AF', 'RF']},
+         'diff_x_elbow': {'operations': ['std'], 'exercises': ['EL', 'AF', 'RF']},
+         'diff_x_shoulder': {'operations': ['std'], 'exercises': ['EL', 'AF', 'RF']},
+ 
+         # symmetry y, left right
+         'diff_y_wrist': {'operations': ['std'], 'exercises': ['EL', 'AF', 'RF', 'AB']},
+         'diff_y_elbow': {'operations': ['std'], 'exercises': ['AB']},
+         
+         # symmetry z, left right
+         'diff_z_wrist': {'operations': ['std'], 'exercises': ['AB', 'AF', 'RF']},
+         'diff_z_elbow': {'operations': ['std'], 'exercises': ['AB', 'AF', 'RF']},
+         'diff_z_shoulder': {'operations': ['std'], 'exercises': ['AB', 'AF', 'RF']},
+ 
+         'z_elbow': {'operations': ['max'], 'exercises': ['AB', 'AF', 'RF']},
+         'z_wrist': {'operations': ['max'], 'exercises': ['AB', 'AF', 'RF']},
+         'x_wrist': {'operations': ['max'], 'exercises': ['EL']},
+         
+         # velocities wrists x z, left right
+         'vel_wrists_x_l': {'operations': ['std'], 'exercises': ['EL']},
+         'vel_wrists_x_r': {'operations': ['std'], 'exercises': ['EL']},
+         
+         # velocties elbows z, left right
+         'vel_elbows_z_l': {'operations': ['std'], 'exercises': ['AF', 'RF', 'AB']},
+         'vel_elbows_z_r': {'operations': ['std'], 'exercises': ['AF', 'RF', 'AB']},
+ 
+         # accelerations wrists x z, left right
+         'acc_wrists_x_l': {'operations': ['mean', 'std'], 'exercises': ['EL']},
+         'acc_wrists_x_r': {'operations': ['mean', 'std'], 'exercises': ['EL']},
+         
+         # accelerations elbows z, left right
+         'acc_elbows_z_l': {'operations': ['mean', 'std'], 'exercises': ['AF', 'RF', 'AB']},
+         'acc_elbows_z_r': {'operations': ['mean', 'std'], 'exercises': ['AF', 'RF', 'AB']},
+ 
+         # angular velocity en acceleration elbow, left right
+         'angular_vel_xz_elbow_l': {'operations': ['std'], 'exercises': ['AF', 'RF']},
+         'angular_vel_xz_elbow_r': {'operations': ['std'], 'exercises': ['AF', 'RF']},
+         'angular_acc_xz_elbow_l': {'operations': ['mean', 'std'], 'exercises': ['AF', 'RF']},
+         'angular_acc_xz_elbow_r': {'operations': ['mean', 'std'], 'exercises': ['AF', 'RF']},
+         
+         'angular_vel_yz_elbow_l': {'operations': ['std'], 'exercises': ['AB']},
+         'angular_vel_yz_elbow_r': {'operations': ['std'], 'exercises': ['AB']},
+         'angular_acc_yz_elbow_l': {'operations': ['mean', 'std'], 'exercises': ['AB']},
+         'angular_acc_yz_elbow_r': {'operations': ['mean', 'std'], 'exercises': ['AB']},
+     }
-     @property
-     def category(self):
-         return self['AB'].category
-   
-         mapper = { x:x.split('_')[0] + '_' + '_'.join(x.split('_')[2:]) for x in series.index }
---
+ 
+         def transform_name(x):
+             if not isinstance(x,str):
+                 return x
+ 
+             split = x.split('_')
+             if len(split) > 2:
+                 return split[0] + '_' + '_'.join(split[1:])
+             else:
+                 return split[0] + '_' + '_'.join(split[1:])
+ 
+         mapper = { x: transform_name(x) for x in series.index }
-         dfmin = [ self[extype].df[[f]].min().add_suffix(f'_min_{extype}') for extype, fs in self.features.items() for f in fs ]
-         dfmax = [ self[extype].df[[f]].max().add_suffix(f'_max_{extype}') for extype, fs in self.features.items() for f in fs ]
-         series = pd.concat( dfmin + dfmax, axis=0 )
-         mapper = { x:x.split('_')[0] + '_' + '_'.join(x.split('_')[2:]) for x in series.index }
---
+         dfs = []
+         
+         for f, opts in self.fs.items():
+             for extype in opts['exercises']:
+                 if len(opts['operations']) > 0:
+                     for op in opts['operations']:
+                         if extype in self:
+                             if f in self[extype].df.columns:
+                                 if hasattr(self[extype].df[[f]], op):
+                                     method = getattr(self[extype].df[[f]], op)
+                                     dfs.append(method().add_suffix(f'_{op}_{extype}'))
+                                 else:
+                                     print('skipping', f, extype, 'because invalid operation', op)
+                             else:
+                                 print('skipping', f, extype, 'because', f, 'not in', self[extype].df.columns)
+                         else:
+                             print('skipping', f, 'because', extype, 'not in', self.keys())
+                 elif extype in self:
+                     if f in self[extype].df.columns:
+                         dfs.append(self[extype].df[[f]].add_suffix(f'_{extype}'))
+                     else:
+                         print('skipping', f, 'because', f, 'not in', self[extype].df.columns)
+                 else:
+                     print('skipping', f, 'because', extype, 'not in', self.keys())
+         series = pd.concat(dfs, axis=0)
+     def to_series(self):
+         dfs = []
+         
+         for f, opts in self.fs.items():
+             for extype in opts['exercises']:
+                 if len(opts['operations']) > 0:
+                     for op in opts['operations']:
+                         if extype in self:
+                             if f in self[extype].df.columns:
+                                 if hasattr(self[extype].df[[f]], op):
+                                     method = getattr(self[extype].df[[f]], op)
+                                     dfs.append(method().add_suffix(f'_{op}_{extype}'))
+                                 else:
+                                     print('skipping', f, extype, 'because invalid operation', op)
+                             else:
+                                 print('skipping', f, extype, 'because', f, 'not in', self[extype].df.columns)
+                         else:
+                             print('skipping', f, 'because', extype, 'not in', self.keys())
+                 elif extype in self:
+                     if f in self[extype].df.columns:
+                         if not isinstance(f, int):
+                             a = self[extype].df[[f]].to_numpy()
+                             if self[extype].df[[f]].values.shape[0] == 1 or (a[0] == a).all(0):
+                                 dfs.append(self[extype].df[[f]].add_suffix(f'_{extype}'))
+                             else:
+                                 print('skipping', f, 'because no operation is given, and the series can\'t be aggregated.')
+                         else:
+                             print('skipping', f, 'because it is an integer')
+                     else:
+                         print('skipping', f, 'because', f, 'not in', self[extype].df.columns)
+                 else:
+                     print('skipping', f, 'because', extype, 'not in', self.keys())
+         series = pd.concat(dfs, axis=0)
+         return series
+     
-         dfmin = [ self[extype].df[[f]].min().add_suffix(f'_min_{extype}') for extype, fs in self.features.items() for f in fs ]
-         return self.remap(pd.concat( dfmin, axis=0 ))
---
+         dfmin = [ self[extype].df[[f]].max().add_suffix(f'_ax_{extype}') for extype, fs in self.features.items() for f in fs ]
+         return self.remap(pd.concat( dfmin, axis=0 ))         
+ 
+     @classmethod
+     def add_features(cls, features, exercises = [], operations = []):
+         if isinstance(features,dict):
+             cls.fs.update(exercises)
+             return
+ 
+         for op in operations:
+             if op not in ['min', 'max', 'std', 'mean', 'var', 'prod']:
+                 raise Exception(f'Invalid operation {op}, must be one of: ' + ', '.join(['min', 'max', 'std', 'mean', 'var', 'prod']))
+ 
+         if type(features) == str:
+             features = [features]
+         for f in features:
+             if f not in cls.fs:
+                 cls.fs[f] = {'operations': [], 'exercises': []}
+             cls.fs[f]['operations'] = cls.fs[f]['operations'] + list(set(operations) - set(cls.fs[f]['operations']))
+             cls.fs[f]['exercises'] = cls.fs[f]['exercises'] + list(set(exercises) - set(cls.fs[f]['exercises']))

```