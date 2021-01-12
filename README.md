# Portfolio Minor Applied Data Science<!-- omit in toc -->
### Student name: Beau Fiechter<!-- omit in toc -->
### Student number: 14152584<!-- omit in toc -->

This document describes my progress, contributions and reflection of the OrthoEyes 2020 project, as well as background information about the project.

# <a id="table-of-contents"></a>Table of Contents <!-- omit in toc -->
- [Obligatory Criteria](#obligatory-criteria)
  - [Datacamp assignments](#datacamp-assignments)
  - [Personal Reflection](#personal-reflection)
  - [Personal Learning Objectives](#personal-learning-objectives)
  - [Evaluation of the group project](#evaluation-of-the-group-project)
- [1. The Project](#1-the-project)
  - [Future work](#future-work)
  - [Conculsions](#conculsions)
  - [Planning](#planning)
- [2. Predictive Analysis](#2-predictive-analysis)
  - [Dichotomous Tree with the OR-Ensemble](#dichotomous-tree-with-the-or-ensemble)
  - [Multinomial Logistic Regression](#multinomial-logistic-regression)
  - [Final OR-ensemble experiments](#final-or-ensemble-experiments)
- [3. Domain Knowledge](#3-domain-knowledge)
  - [Literature](#literature)
  - [Terminology](#terminology)
- [4. Data Preprocessing](#4-data-preprocessing)
  - [Exploring and explanation of the dataset](#exploring-and-explanation-of-the-dataset)
  - [Exploring and explanation of existing code](#exploring-and-explanation-of-existing-code)
  - [Visualization (and further exploration of the data)](#visualization-and-further-exploration-of-the-data)
- [5. Communication](#5-communication)
  - [Presentations](#presentations)
  - [Paper](#paper)
  - [Feedback](#feedback)

---

# Obligatory Criteria
## Datacamp assignments

- Pandas Foundation

    [Statement of accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/191ae7b0e88d5a01c2fe3b170f0de4223b59cea8)

- Manipulating DataFrames with pandas

    [Statement of accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/c3fa774860c87b6e84dd4389164b5b472eb69b42)

- Introduction to Data Visualization in Python

    [Statement of accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/44b629ff307d16bfd3f76bd95e689fe5f1c18a40)

- Preprocessing for Preprocessing for Machine Learning in Python
    
    [Statement of accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/9b11e6d7a7554ad8ff1963889a07c87a40b4aab5)

- Data Types for Data Science in Python

    [Statement of accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/2d8bb26091fa49e453ef56034ca17a901b70df19)

- Cleaning Data in Python

    [Statement of accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/70fbf61dba08f8b162f4939ea49e03425dbedd11)

- Introduction to Python

    [Statement of accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/f9c4ed638870ff224beb6a52894423fe728bdc3d)

- Intermediate Python

    [Statement of accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/6c07b815a7cb2b59032cddb0983cb114a85c1e48)

- Python Data Science Toolbox (Part 1)

    [Statement of accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/767c4e5be465ac5faf0716dcb92563fd3fa3727f)


[Back to Table of Contents](#table-of-contents)
## Personal Reflection
When I started with the project, part of the group was already a week in. This was due to me joining the group at a later time because there was a participation spot that came free when I started the other minor I was signed in for. This wasn't really a problem though, because I could easily catch up with the rest of the group.

I didn't really have a specific role in the group. I did act like a 'back-end engineer' together with Danny. Together we did most of the coding for the [convenience library](code/lib/ortho_lib.py) we put together. I also already had some experience in machine learning and neural networks, so at times (mostly at the beginning) I could provide some extra information when it was needed. We worked in a free form of agile development from week 3 or 4 on, so we divided tasks at the start of the week, and we just carried them out.

Other tasks I worked on:
- [Reconstructing the results from the original paper](notebooks/Reconstructie%20Paper%20Model.ipynb)
- [A lot of visualization during the project](#visualization-and-further-exploration-of-the-data)
- [Presentations](#presentations)
- Feature selection
- [Constructing the dichotomous classification tree](#dichotomous-tree-or-ensemble)
- [Multinomial Logistic Regression](#multinomial-logistic-regression)
- [Constructing the final experiment and results](notebooks/final_experiment.ipynb)
- [Writing, but mostly reviewing, the paper](#paper)

Here I will reflect on a situation using STARR:
### **Updating the library to support more aggregation methods** <!-- omit in toc -->
#### **Situation** <!-- omit in toc -->
To generate more kinds features, an expansion of the `ortho_lib` library is needed. All sequence aggregation methods of the `pandas` `DataFrame` class should be supported. Currently, the only supported methods are `min()` and `max()`. The method should still support specifying for which exercise types the feature should be generated.

#### **Task** <!-- omit in toc -->
To rewrite the `Patient` class to support all of the sequence aggregation methods.

#### **Action** <!-- omit in toc -->
I rewrote the `Patient` class to have a new input structure, in which the basic feature name could be specified, together with the mathematical operations to aggregate the column values. The exercise types could be specified as well. I also made a change to the naming of the generated feature.

#### **Results** <!-- omit in toc -->
With this relatively small change, a lot more kinds of features can be generated. Some of the existing notebooks will have to be changed a little to support this change. 

#### **Reflection** <!-- omit in toc -->
In this situation, I think I made a pretty large impact on the feature engineering part of the project. Where we could first only generate minimal and maximal column values through our library, we can now do any aggregation method. It has made it easy to generate large numbers of features, and I think that the group really profitted from this action, as it saves a lot of manual work.

## Personal Learning Objectives
For this minor I mainly wanted to expand my knowledge on machine learning and neural networks. I was really happy when I received the message that I could join the minor, even after it started, because this was my favorite choice of minor in the first place. I already had some knowledge on ML and NN through online articles I read and YouTube videos I watched. I also ported a so-called Intent Parser (parses an intent from a written sentence) called Padatious from Python to Java. This system used a simple neural network. I did learn a lot during this minor though. The intricacies of developing, using and evaluating a machine learning algorithm were not new to me, but I did gain in-depth knowledge and I learned to apply the different methods correctly. 

I also learned more about programming in Python. I mainly program in Java, which is a lot more strict than Python, and PHP, but I did know a little Python beforehand. Through the Datacamp assignments I learned some of the intricacies of Python, like list comprehensions, and I can read and write Python code quite well now, if I say so myself. This is why Danny and me were very able to compile and expand the library we used to suite the needs of the group and the project and made running nicely structured experiments a trivial task.


Here I will reflect on a situation using STARR:
### **Building and evaluating the final model** <!-- omit in toc -->
#### Situation <!-- omit in toc -->
We've done the evaluation of the trainingset, which yielded very promising results. We now have to build the final model and evaluate the test data we separated beforehand. The training using the OR-ensemble generated lots of models for each feature, because of the cross validation method we used. We initially wanted extract these models from the training results. Later we realised we just needed to create a totally new model, without validating and evaluate it on the test set.

#### Task <!-- omit in toc -->
Train a model on the training data and evaluate the test data on the trained model.

#### Action <!-- omit in toc -->
The library we compiled was not built to generate results for this case. I needed to extract of the some of the classes and methods and make some modifications to get them to return model and scaler objects. Our results were previously always generated by the `Experiment` class of our library, so I also needed to modify the class to not need the `Experiment` object to be constructed.

#### Results <!-- omit in toc -->
The result of this situation were also the final results of our project. We got a scoring of our model on data it has not seen before, which gives us an unbiased result.

#### Reflection <!-- omit in toc -->
I was really struggling when I was extracting models from the library at first. I should have known this wasn't the correct way to get our final results. I really needed some help with this, and none of my groupmembers could really help me with it. In the end I got the answer by asking mr. Andrioli and mr. Vuurens. I think this final experiment shouldn't have taken as long as it did, and we could have gotten our results at an earlier time.

## Evaluation of the group project
In my personal opinion, this was the nicest project team I've ever been part of. Every member of the team was able give their opinion and was also able to react in an adult way. We did have our differences from time to time, maybe because our group is a mix of multiple disciplines. When we came accross a conflict, it was address right way or at the weekly retrospective.

I can be a *little direct* at times, especially when I want to be right and I'm sure I'm right. This can sometimes result people thinking I'm angry or something like that, which is never my intention. I've had some conflicts with some of the group members regarding this fact. The nice thing is that they were not scared to share their opinion or to address their experience. I can really appreciate this, because I can then correct my own behaviour. 

The group had some really hard workers, with great work mentality. At times I felt a little inadequate in this aspect. But these hard workers were also very able to motivate me in the same way. I think I worked harder because I saw them work hard.

At the start of the project we worked at the THUAS on wednesdays. Later, because of the new measures for the pandemic, we had to cancel this. In stead we decided to meet (almost) all day in the Teams chat. I think whis helped a lot at keeping up with the work, because I tend to slack off when I work on my own.

Overall, I'm very happy with the way our team communicated and worked together. 

Here I will reflect on a situation using STARR:
### **Leading a retrospective** <!-- omit in toc -->

#### Situation <!-- omit in toc -->
At the end of a week, we always held a retrospective to reflect on our week. Not only to reflect on the work we had done, but also on the ambiance in the group. Every week we switch the leader of the retrospective.

#### Task <!-- omit in toc -->
Make sure the group fills their retrospective points in a serious way and the discussions following these points are done in an adult and serious manner.

#### Action <!-- omit in toc -->
Walk through the retrospective in a structured way. Let everyone fill out their positive and negative points to discuss later. Let everyone explain their points and make sure no one interupts. After everyone has done their explanation, discussion can be started about the points.

#### Results <!-- omit in toc -->
The retrospectives I led were sometimes more structured than the other. I feel like a strict retrospective is not per say necessary, as long as everyone can still follow the discussions.

#### Reflection <!-- omit in toc -->
Overall, I think the retrospectives I led were done quite well. I gave enough space for people to react on eachother. I also was active in these discussions and was able to solve conflicts when necessary.


[Back to Table of Contents](#table-of-contents)
# 1. The Project

The OrthoEyes project is a project that explores the possibilities of identifying people with shoulder injuries, in particular rotator cuff tears. The project has had several iterations in the last years. This year we focussed on the expansion of the featureset for the so-called 'OR-ensemble', which is a machine learning ensembling method developed and explored by [A. Andrioli and J. Vuurens](pdf/ortho.pdf). 

The project is of value to the medical domain, in specific the orthopedic domain. If this project can provide an efficient and reliable way of indicating whether someone has a rotator cuff tear, a lot of time and funds would be saved.

The team this year consisted of originally four students, but after a week I joined the team and a week later our team had another addition.

At the start of the project it was quite unclear what the goal of the project was. Later we were able to define the following research question and sub-questions.

    Research Question:

    Which models can be added to the current OR-ensemble to further improve the ability to properly identify individuals as part of the correct patient group, using the existing data gained from a Flock of Birds system?

    1. Based on the 3D animations from the Flock of Birds system, which features are promising for classifying shoulder injuries from a physiotherapeutic perspective?

    2. Based on the data from the Flock of Birds system, which features are promising for classifying shoulder injuries from a data science perspective? 

    3. What is the certain added value of the parameters that were chosen in addition to the existing ensemble?
   
    4. Which classification models fit best with the chosen features?
    
    5. How do patients refrain certain parts of movements and how can this be recognized in the dataset?

During the course of the project, we did deviate from these questions a little bit. WE dropped question 4, because we decided to only use the integrated Logistic Regression model used in the OR-ensemble. The goal remained the same: Expanding and improving the OR-ensemble by add more features. 

Somewhere in the first weeks of the project, we wrote a so-called 'Plan of Attack', which is a document in which we document what possible challenges and the preferred results of the project were. This document can be found [here](pdf/Plan%20of%20Attack.pdf). Note that the project deviated from this plan, as we noticed we needed to make some changes after a couple of weeks, and we did not update this document.


## Future work
I think we made a lot of progress in this iteration of the project. As the project will likely be continued, I think there are lots of opportunities for the future.

One of the improvements to be made is data collection. The dataset we received is very small, and that's why I'm convinced the results did not meet my expectations. I really thought the model would generate nearly perfect results at separating patients from non-patients, because the training evaulation yielded such good metrics. I'm convinced that when the model is fed more data, the results will improve. 
If there is more data, the method of cross-validating can be changed as well. I wouldn't know the impact of this, but it might be worth a try. 

We did a lot of feature engineering. But there always could be some features we didn't think of. So further feature exploration might be interesting for future work.

The OR-ensemble creates a Logistic Regression model for each feature in the featureset. However, all of the (hyper)parameters are left at default. We did not explore tweaking any of these parameters. Future work could explore this as well.

The dataset also contained CSV files with calibrated and transformed data about angles of the bones of individuals. We did not use these files, because we calculated any angle we used as feature using the sensor positions. I think there are some features in these files which are not covered by our featureset.

## Conculsions
To answer our main research question, we need to explain that features are different for each exercise type. Below is a table of the combinations of feature bases and the mathematical operations that are performed on these values sequences. 

<details>
<summary>The models(features) that can be added to the OR-ensemble are:</summary>

|     Feature basis              |     Mathematical operation    |     Axis / plane    |     AB    |     AF    |     RF    |     EL    | Description                                                                                |
|--------------------------------|-------------------------------|---------------------|-----------|-----------|-----------|-----------|--------------------------------------------------------------------------------------------|
|     angle_left_shoulder_xz     |     Max                       |     XZ              |           |     ✔     |     ✔     |           | Maximal angle of the right shoulder in the XZ plane                                        |
|     angle_right_shoulder_xz    |     Max                       |     XZ              |           |     ✔     |     ✔     |           | Maximal angle of the left shoulder in the XZ plane                                         |
|     angle_left_shoulder_yz     |     Max                       |     YZ              |     ✔     |           |           |           | Maximal angle of the right shoulder in the YZ plane                                        |
|     angle_right_shoulder_yz    |     Max                       |     YZ              |     ✔     |           |           |           | Maximal angle of the left shoulder in the YZ plane                                         |
|                                |                               |                     |           |           |           |           |                                                                                            |
|     diff_x_wrist               |     Std                       |     X               |           |     ✔     |     ✔     |     ✔     | Standard Deviation of the symmetry of the wrists on the X-axis                             |
|     diff_x_elbow               |     Std                       |     X               |           |     ✔     |     ✔     |     ✔     | Standard Deviation of the symmetry of the elbows on the X-axis                             |
|     diff_x_shoulder            |     Std                       |     X               |           |     ✔     |     ✔     |     ✔     | Standard Deviation of the symmetry of the shoulders on the X-axis                          |
|                                |                               |                     |           |           |           |           |                                                                                            |
|     diff_y_wrist               |     Std                       |     Y               |     ✔     |     ✔     |     ✔     |     ✔     | Standard Deviation of the symmetry of the wrists on the Y-axis                             |
|     diff_y_elbow               |     Std                       |     Y               |     ✔     |           |           |           | Standard Deviation of the symmetry of the elbows on the Y-axis                             |
|                                |                               |                     |           |           |           |           |                                                                                            |
|     diff_z_wrist               |     Std                       |     Z               |     ✔     |     ✔     |     ✔     |           | Standard Deviation of the symmetry of the wrists on the Z-axis                             |
|     diff_z_elbow               |     Std                       |     Z               |     ✔     |     ✔     |     ✔     |           | Standard Deviation of the symmetry of the elbows on the Z-axis                             |
|     diff_z_shoulder            |     Std                       |     Z               |     ✔     |     ✔     |     ✔     |           | Standard Deviation of the symmetry of the shoulders on the Z-axis                          |
|                                |                               |                     |           |           |           |           |                                                                                            |
|     z_elbow                    |     Max                       |     Z               |     ✔     |     ✔     |     ✔     |           | Maximal Height of the elbows                                                               |
|     z_wrist                    |     Max                       |     Z               |     ✔     |     ✔     |     ✔     |           | Maximal Height of the wrists                                                               |
|     x_wrist                    |     Max                       |     X               |           |           |           |     ✔     | Maximal Depth of the elbows                                                                |
|                                |                               |                     |           |           |           |           |                                                                                            |
|     vel_wrists_x_l             |     Std                       |     X               |           |           |           |     ✔     | Standard Deviation of the velocity of the left wrist on the X-axis                         |
|     vel_wrists_x_r             |     Std                       |     X               |           |           |           |     ✔     | Standard Deviation of the velocity of the right wrist on the X-axis                        |
|                                |                               |                     |           |           |           |           |                                                                                            |
|     vel_elbows_z_l             |     Std                       |     Z               |     ✔     |     ✔     |     ✔     |           | Standard Deviation of the velocity of the left elbow on the Z-axis                         |
|     vel_elbows_z_r             |     Std                       |     Z               |     ✔     |     ✔     |     ✔     |           | Standard Deviation of the velocity of the right elbow on the Z-axis                        |
|                                |                               |                     |           |           |           |           |                                                                                            |
|     acc_wrists_x_l             |     Mean, std                 |     X               |           |           |           |     ✔     | Mean and Standard Deviation of the velocity of the left wrist on the X-axis                |
|     acc_wrists_x_r             |     Mean, std                 |     X               |           |           |           |     ✔     | Mean and Standard Deviation of the velocity of the right wrist on the Z-axis               |
|                                |                               |                     |           |           |           |           |                                                                                            |
|     acc_elbows_z_l             |     Mean, std                 |     Z               |     ✔     |     ✔     |     ✔     |           | Mean and Standard Deviation of the acceleration of the left elbow on the Z-axis            |
|     acc_elbows_z_r             |     Mean, std                 |     Z               |     ✔     |     ✔     |     ✔     |           | Mean and Standard Deviation of the acceleration of the right elbow on the Z-axis           |
|                                |                               |                     |           |           |           |           |                                                                                            |
|     angular_vel_xz_elbow_l     |     Std                       |     XZ              |           |     ✔     |     ✔     |           | Standard Deviation of the angular velocity of the left elbow on the XZ plane               |
|     angular_vel_xz_elbow_r     |     Std                       |     XZ              |           |     ✔     |     ✔     |           | Standard Deviation of the angular velocity of the right elbow on the XZ plane              |
|     angular_acc_xz_elbow_l     |     Mean, std                 |     XZ              |           |     ✔     |     ✔     |           | Mean and Standard Deviation of the angular acceleration of the left elbow on the XZ plane  |
|     angular_acc_xz_elbow_r     |     Mean, std                 |     XZ              |           |     ✔     |     ✔     |           | Mean and Standard Deviation of the angular acceleration of the right elbow on the XZ plane |
|                                |                               |                     |           |           |           |           |                                                                                            |
|     angular_vel_yz_elbow_l     |     Std                       |     YZ              |     ✔     |           |           |           | Standard Deviation of the angular velocity of the left elbow on the YZ plane               |
|     angular_vel_yz_elbow_r     |     Std                       |     YZ              |     ✔     |           |           |           | Standard Deviation of the angular velocity of the right elbow on the YZ plane              |
|     angular_acc_yz_elbow_l     |     Mean, std                 |     YZ              |     ✔     |           |           |           | Mean and Standard Deviation of the angular acceleration of the left elbow on the YZ plane  |
|     angular_acc_yz_elbow_r     |     Mean, std                 |     YZ              |     ✔     |           |           |           | Mean and Standard Deviation of the angular acceleration of the right elbow on the YZ plane |

</details>

<details>
<summary>Results: Identifying patients

The model outperforms AdaBoost, bagging and SVC in identifying patients/non-patients. The Random Forest ensemble method still outperforms the OR-ensemble.
</summary>

|                  |     OR-ensemble     f=1.7    |     AdaBoost    |     Random Forest    |     Bagging    |     SVC     |
|------------------|------------------------------|-----------------|----------------------|----------------|-------------|
|     Precision    |     0.9                      |     0.74        |     0.93             |     0.8        |     0.87    |
|     Recall       |     0.95                     |     0.76        |     0.92             |     0.88       |     0.88    |
|     Accuracy     |     0.88                     |     0.76        |     0.91             |     0.88       |     0.88    |

</details>
<details>
<summary>Results: Identifying rotator cuff tears

In identifying patients with/without rotator cuff tears the OR-ensemble is outperformed by Baggin and SVC, but not by Random Forest or AdaBoost.
</summary>

|                  |     OR-ensemble     f=1.45    |     AdaBoost    |     Random Forest    |     Bagging    |     SVC     |
|------------------|-------------------------------|-----------------|----------------------|----------------|-------------|
|     Precision    |     0.78                      |     0.40        |     0.47             |     0.73       |     0.73    |
|     Recall       |     0.58                      |     0.63        |     0.52             |     0.75       |     0.73    |
|     Accuracy     |     0.63                      |     0.63        |     0.49             |     0.75       |     0.73    |

</details>


## Planning 

We made a global planning to get a feeling for the course of the project. In my experience, global plannings quickly deviate from reality. That is why I mainly tried to keep up with weekly plannings.

Throughout the project, we used a Trello board to manage our tasks in an agile-like manner. We created a small backlog at the beginning of the project and made additions to it when we did our weekly planning-moment on mondays. Tasks were also added by the retrospectives to held every friday (since week 5). 

Some tasks in the Trello board had deadlines, to make sure these tasks were carried out that week or day. Even though these deadlines were sometimes changed, I think we did a pretty good job at planning this project.

<details>
<summary>Trello board screenshot</summary>
<img src="img/trello.png" />
</details>

[Back to Table of Contents](#table-of-contents)
# 2. Predictive Analysis

## Dichotomous Tree with the OR-Ensemble
[Notebook](notebooks/Dichotomous%20Tree.ipynb)

At some point in the project, we decided we wanted to classify patients into a specific category. We thought about using a dichotomous tree like the one below. By process of elimination a person would be classified. We later abandoned this idea, as we mainly focussed on two One-vs-Rest classification for patients and non-patients, and also separating Category 2 from 3 and 4. Even though we abandoned the idea, I already created a notebook to facilitate the tree structure. 
<details><summary>Tree visualization</summary>
<center><img src="img/tree.png"></center>
</details>

In the notebook, different category orders are created. The tree is walked one step, the system runs and experiment with the OR-ensemble. Then, with all true positive patients, the next experiment is run, et cetera. 

Note that the notebook is dated and made before we cleaned the data. The results are not realy bad, but it might be possible that the data is still littered with noise and recording mistakes. The features used in this notebook can also deviate from the final set of features we used for our final model.

<details>
<summary>Results (notebook output)</summary>

```
for order 1-2-3 factors 1.8-1.8-1.8 we got 84/123 (68.293%) in the correct category
    Category 1: 26/26
    Category 2: 36/36
    Category 3: 4/34
    Category 4: 18/27
    

for order 1-2-4 factors 1.8-1.8-1.8 we got 83/123 (67.47999999999999%) in the correct category
    Category 1: 26/26
    Category 2: 36/36
    Category 4: 18/27
    Category 3: 3/34
    

for order 1-3-2 factors 1.8-1.8-1.8 we got 89/123 (72.358%) in the correct category
    Category 1: 26/26
    Category 3: 34/34
    Category 2: 6/36
    Category 4: 23/27
    

for order 1-3-4 factors 1.8-1.8-1.8 we got 89/123 (72.358%) in the correct category
    Category 1: 26/26
    Category 3: 34/34
    Category 4: 23/27
    Category 2: 6/36
    

for order 1-4-3 factors 1.8-1.8-1.8 we got 74/123 (60.163%) in the correct category
    Category 1: 26/26
    Category 4: 27/27
    Category 3: 14/34
    Category 2: 7/36
    

for order 1-4-2 factors 1.8-1.8-1.8 we got 77/123 (62.602000000000004%) in the correct category
    Category 1: 26/26
    Category 4: 27/27
    Category 2: 15/36
    Category 3: 9/34
```

</details>

## Multinomial Logistic Regression
[Notebook](notebooks/multinomial%20logistic%20regression.ipynb)

The notebook above contains my experiments with a Multinomial Logistic Regression model. In the notebook, a lot of different combinations of hyperparameters are tried and the best results (results with the highest weighted averages) are then printed. I built it in this way to avoid manually tweaking hyperparameters. 

<details>
<summary>Results (best model)</summary>

```
CV: RepeatedKFold
solver: lbfgs | penalty: l2 | C: 0.1 | l1_ratio: None
              precision    recall  f1-score   support

  Category 1       0.76      0.96      0.85        23
  Category 2       0.63      0.69      0.66        32
  Category 3       0.58      0.48      0.53        29
  Category 4       0.89      0.74      0.81        23

    accuracy                           0.70       107
   macro avg       0.72      0.72      0.71       107
weighted avg       0.70      0.70      0.70       107
```

</details>

## Final OR-ensemble experiments
[Notebook 1: Category 1 vs Categories 2,3,4](notebooks/final_experiment.ipynb)

[Notebook 2: Category 2 vs Categories 3,4](notebooks/final_experiment_234.ipynb)

The notebooks in the links above contain our final experiments. Both are almost exactly the same notebooks. The only difference is that the Y-condition for the experiment is set on a different category. In the second experiment, category 1 is not included in the training and evaluation.

<a id="get-featureset"></a>In the experiment, the first two cells fits and evaluates a model on the training set. This results in a set of features that are included in the ensemble and have precision values of `1.0`. We then begin our actual final experiment.

The model is trained on the full training set, and the model and scaler are returned. The scaler is included in the output of the model training, as we need to scale our test data using this scaler. The test data is then scaled and predictions are made. For each of the features in the set collected from the [first two cells](#get-featureset) this procedure is repeated. The results for each feature/model are then combined by getting the maximum value for each column (patient), replicating a Boolean OR operation, ergo, the OR-ensemble. The results are then reported and we can review our evaluation.

<details>
<summary>Results: Category 1 vs. Categories 2, 3, 4</summary>

| Recall   | Precision | Accuracy |
|----------|-----------|----------|
| 0.947368 | 0.9       | 0.88     |

</details>
<details>
<summary>Results: Category 2 vs. Categories 3, 4</summary>

| Recall   | Precision | Accuracy |
|----------|-----------|----------|
| 0.583333 | 0.777778  | 0.631579 |

</details>
<br>
For closer inspection of the results of each feature/model, I included a distribution graph in which we can observe the datapoints of the training set, outliers and the test set, as well as the regression line. Below is an example of one of these plots.

<details>
<summary><a id="logit-viz"></a>Regression visualization example</summary>
<img src="img/logit_viz.png"/>
</details>

[Back to Table of Contents](#table-of-contents)
# 3. Domain Knowledge

This projects applies data science to the medical field. The medical field has some requirements that other domains may not have. In the medical field predictions have to be very precise. This means we want to avoid false diagnoses as much as possible. It is also very nice if we can explain our prediction, so that medical examiners can use this explanation in their diagnosis. In our project, we implemented a training requirement for our system to never base our final prediction on a feature that has given false positive results to accomodate this requirement. In this way, we tried to make our predictions as trustworthy as possible.

## Literature
Firstly, I have read the papers of previous iterations of the project:

- [One-Sided Clissification for Interpretable Predictions](pdf/ortho.pdf)
- [Kinematic analysis of shoulder motion for diagnostic purposes](pdf/Ortho_Eyes_Paper%20(1).pdf)
- [Kinematic analysis of shoulder motion for diagnostic purposes](pdf/paper_ortho_eyes2018-2019%20(1).pdf)

I also read the following papers to gain more knowledge on this specific domain of research:

- [3D Analysis of Upper Limbs Motion during Rehabilitation Exercises Using the KinectTM Sensor: Development, Laboratory Validation and Clinical Application](pdf/sensors-18-02216.pdf)
- [Detection of Outliers and Influential Observations in Binary Logistic Regression: An Emprical Study](pdf/outliers.pdf)
- [Classification of K-Pop Dance Movements Based on Skeleton Information Obtained by a Kinect Sensor](pdf/sensors-17-01261-v2.pdf)

And the following articles:
 - [A 2019 guide to 3D Human Pose Estimation](https://nanonets.com/blog/human-pose-estimation-3d-guide/)

## Terminology
<details>
<summary>
In the project there are some terms and jargon that might need to be explained:
</summary>

- OR-ensemble

    The novel ensemble method explored in this project. Combines multiple binary machine learning models using a Boolean OR-operation.

- `ortho_lib`

    The library our team compiled from old code and made additions and modification to to accomodate the needs of the project.
</details>

[Back to Table of Contents](#table-of-contents)
# 4. Data Preprocessing
Because the project has had several iterations, there was already a lot of work done with the data, including some code which made it a little more convenient to work with our dataset.

## Exploring and explanation of the dataset

The provided dataset consisted of directories with the patients and their executed exercises, split up into their corresponding category directories:
```
+-- Category_1          # Category folder
|   +-- 1               # Patient/Person
|       +-- AB1.txt     # Single exercise file (TXT)
|       +-- AB1.csv     # Single exercise file (CSV)
|       +-- AB2.txt
|       +-- AB2.csv
|       +-- ...
|   +-- 2
|       +-- AB1.txt
|       +-- ...
|   +-- ...
+-- Category_2
|   +-- ...
+-- ...
```

Wherein the `*.txt` files included the positional XYZ-data of each sensor and the rotation matrix for that sensor. The file contains data in the following format:
```
[sensor_number] [position_x]  [position_y]  [position_z]
                [rotation_x1] [rotation_y1] [rotation_z1]
                [rotation_x2] [rotation_y2] [rotation_z2]
                [rotation_x3] [rotation_y3] [rotation_z3]
```
This format is repeated for every sensor number and repeated for each recorded frame for the exercise.

 The `*.csv` files include calibrated euler angles for each of the following bones both left and right:
 - Thorax (3 parts)
 - Clavicula (3 parts)
 - Scapula (3 parts)
 - Humerus (3 parts)
 - Elbow (1 angle)

## Exploring and explanation of existing code

From the code that was delivered to us at the start of the project, I compiled a library file called `ortho_lib`. This library was used throughout the project with most of the tasks we needed to execute. The library is found [here](code/ortho_lib.py).

Of course, we made some additions to the code to expand the functionality and possibilties. This file has been modified throughout the project by both my teammembers and myself to suite the needs of the project at the time.

### Additions to the code <!-- omit in toc -->
One of these changes was in the processing of feauture value generation. This was originally done by just calculating the maximum value of a feature. I rewrote the `Patient` class to accept all column and row aggregation methods included in `pandas` (`max`, `min`, `avg`, `std` etc.). The diff for this can be found [here](code/diffs/patient.md). The new version of the `Patient` class also includes the final combinations for features and mathematical operations for specific exercises. It might not be the most efficient code, but it works and is not that slow.

Another big and very usefull change I made was the `fit_inliers_auto_ensemble()` method in the `Experiment` class. This function automatically leaves out any feature in the ensemble which evaluated recall, precision or accuracy does not meet the required threshold value, which is an argument for the function. This resulted in any featureset you feed into the system to have an ensemble precision of `1.0` (if the threshold is `1.0` and any of the evaulation had precision `1.0`) to ensure our requirements.

We also made additions to the `DFFrame` class, which is a simple subclass of the `pandas` `DataFrame` class with pre-made calculations that are used as the basis for features (generated in using the `Patient` class feature definitions).

When we started, I wrote a guide for the rest of the group to understand the class structure and functions of the library. This guide can be found [here](notebooks/__Using_ortho_lib.ipynb). The guide is a bit outdated, as it hasn't been updated since week 7 or so.

## Visualization (and further exploration of the data)
At the start of the project I wanted to visualize the data as much as possible, to understand the stucture and get a better feeling for the data. 

For convenience I wrote a small library file called `ortho_plot`. This library also contains functions to easily plot a single exercise (in Pandas dataframe) in 2d and 3d. This file has been modified throughout the project by both me and my teammembers to suit the needs of the project at the time.

The library is found [here](code/ortho_plot.py).

When we started, I wrote a guide for the rest of the group to understand the class structure and functions of the library. This guide can be found [here](notebooks/__Using_ortho_plot.ipynb).

Below I describe visualizations for the data that may or may not be included in the library.

#### **Animated visualization of an individual** <!-- omit in toc -->
[Notebook](notebooks/movement_all.ipynb)

To better get a feeling for what information we could get from the data, I wanted to make a 3-dimensional visualization of the data by plotting it in a `plotly` 3D graph. Luckily, in a previous iteration of the project, a basic notebook existed for this. I wasn't happy with the notebook, because of the anchor sensor that was included and the fact that sensors weren't connected. I [changed the notebook](notebooks/movement_all.ipynb) so that the anchor is removed in the plot and the plot came out as a sort of 'stick figure'.

This notebook has helped a lot when we were cleaning the data to remove the noise.

##### **Plotting rotational data** <!-- omit in toc -->
[Notebook](notebooks/plot_rotational_data.ipynb)

To better understand the rotational data, I wanted to plot each sensor's data in the 3D graphs described above. I wanted to have an arrow or something like an arrow pointing in the direction the rotation matrices described. I succeeded in plotting `plotly` cones in the directions, but later I realized we wouldn't really need these visualizations and I abandonded the [notebook](notebooks/plot_rotational_data.ipynb).

#### **Regression model (and related) visualizations** <!-- omit in toc -->
[Notebook](notebooks/visualize_factor.ipynb)

The model created uses a simple method of temporarily removing individuals from the training data to fit a Logistic Regression model. It uses a single hyperparameter to separate the datapoints for `y=0` and `y=1`. The visualization for this separation can be found [here](notebooks/visualize_factor.ipynb). It includes the training (or loading results) of models for 20 intervals of the `factor` parameter in order to fetch the features that are included in the model. It then outputs distibution graphs for which of the datapoints are discarded/included for a certain factor.

At the end of the project we realized some of the features we selected gave some inconsistent results when evaluating on the test set. This was due to the way we setup the [final experiment](notebooks/final_experiment.ipynb), we didn't scale our test set properly. So I created a method to visualize the regression models in a distribution graph. Later the scaling was fixed, and we had some nice visualizations for the Logistic Regression models like the one [I showed above](#logit-viz).

#### **Visualization of velocity and acceleration** <!-- omit in toc -->
[Notebook](notebooks/transform_data_velocities.ipynb)

Two feature groups we wanted to explore were the velocity and accelaration of movements. We calculated the velocity by calculating the difference of positions over each frame, and the accelation by calculating the difference of velocities over each velocity frame. The notebook for visualizing these values can be found [here](notebooks/transform_data_velocities.ipynb). Note that half of the visualizations do not work anymore, as this notebook was created in an early stage of the project.


[Back to Table of Contents](#table-of-contents)
# 5. Communication

## Presentations
- [Week 2 - Weekly Presentation](presentations/Week%202%20presentation.pptx)
- [Week 10 - Weekly Presentation](presentations/Week%2010%20presentation%20(9-11)%20(1).pptx)
- [Public Presentation Week 16](presentations/OrthoEyes%20Final%20Public%20Presentation.pptx)

## Paper
When we started writing the paper, I was still busy carrying out other tasks, like the Multinomial Logistic Regression model. This resulted in the paper being mostly writting already before I even finished my task. In the end, we didn't even add the MLR model and results to the paper, so that was a little waste of time. I wrote a small section covering the MLR, but it didn't end up in the paper.

I wrote some other some small pieces for the paper, including some of the discussion and some additions to pieces others wrote. 

Because I had the feeling I wasn't really contributing to the paper at the start, I mainly focussed on reviewing and reformulating pieces the others wrote. We've also had multiple session in which we all worked together on reviewing and improving the paper. I think these sessions were very useful, because we all were able to give an opinion on the pieces we were reviewing and come to a consensus.

## Feedback
During the project we were given the assignment to give each member of the team a 360-degree feedback form. 
<details>
<summary>Feedback from others</summary>
<img src="img/feedback_360.png" title="Feedback 360" />
</details>

<details>
<summary>Feedback on others</summary>

<details>
<summary>Myself</summary>
<img src="img/feedback%20(2).png" title="Feedback Me" />
</details>

<details>
<summary>Tim Bekema</summary>
<img src="img/feedback%20(5).png" title="Feedback Tim" />
</details>

<details>
<summary>Jasmijn Heupers</summary>
<img src="img/feedback%20(6).png" title="Feedback Jasmijn" />
</details>

<details>
<summary>Donna van Grunsven</summary>
<img src="img/feedback%20(3).png" title="Feedback Donna" />
</details>

<details>
<summary>Emma Korf</summary>
<img src="img/feedback%20(4).png" title="Feedback Emma" />
</details>

<details>
<summary>Danny Vink</summary>
<img src="img/feedback%20(1).png" title="Feedback Danny" />
</details>
</details>