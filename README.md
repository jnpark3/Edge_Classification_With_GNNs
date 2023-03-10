# CERN GSoC Machine Learning & Statistics Task:

### Main Contents of Repository

```
task1.ipynb
```

Jupyter notebook described in the deliverable section for this task. Contains exploratory data analysis of provided dataset, training of model, and evaluation of model. Additionally, I provide justification, observations, and explanations for the steps that I take.

```
utils/dataStorage.py
```

Contains the class `dataStorage`, which is used for storing graph data obtained from files. This class extends the class described in the task document by adding a method to append to the graph list, and also performing an automatic 80-20 train-test split of the provided dataset. New methods were added to support this split functionality.

```
utils/model.py
```

Contains the class `EdgeMode`, the model used in `task1.ipynb`. This model is further described below in the model section.

### Running the Contents of this Repository

As the dataset file is not included in this repository in the interest of space, in order to run the contents of the Jupyter notebook, a file named `dataset` must be added to the root folder containing the batch folders `batch_1_0`, `batch_1_1`, ..., `batch_1_9`. Additionally, ensure the necessary imports are installed (listed on the first two cells of the notebook).

### Summary of Results



### Model Design



### Clarifications on Commits/Additional notes

It is noted in the task document that the commits for this repository may be observed, so I wanted to explain 

 - I decided to work on this task on my sibiling's more powerful computer, but I had some issues with setting up the repository at the start. As a result, I accidentally sent a few commits all titled "First Commit." This are all unintentional, and I noticed these commits too late to undo them.

 - While I was solving issues related to git, I forked one of my previously existing (and irrelevant) repositories to get this repository started. As a result, there are two commits on this rep originating before March; these are completely disconnected from my submission.

 - Otherwise, this task was very enjoyable and I had a good time. Thank you for your consideration of my application.