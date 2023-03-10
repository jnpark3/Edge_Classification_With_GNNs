# HSF GSoC Machine Learning & Statistics Task:

### Main Contents of Repository

```
task1.ipynb
```

Jupyter notebook described in the deliverable section for this task. Contains exploratory data analysis of provided dataset, training of model, and evaluation of model. Additionally, I provide justification, observations, and explanations for the steps that I take.

```
utils/dataStorage.py
```

Contains the class `DataStorage`, which is used for storing graph data obtained from files. This class extends the class described in the task document by adding a method to append additional to the graphs to the object after initialization. Also, this class performs an automatic 80-20 train-test split of the provided dataset. New methods were added to support this splitting functionality.

```
utils/model.py
```

Contains the class `EdgeModel`, the model used in `task1.ipynb`. This model is further described below in the model section.

```
software_task_files/copy.ipynb
```

Contains a copy of all classes and functions created as a part of Task 1 of the Software Engineering tasks, including unit testing operations. The original implementations of the main function can be found in `gnn_tracking/models/edge_classifier.py` within my separate task 2 submission.

### Running the Contents of this Repository

As the dataset file is not included in this repository in the interest of space, in order to run the contents of the Jupyter notebook, a folder named `dataset` must be added to the root folder containing the batch folders `batch_1_0`, `batch_1_1`, ..., `batch_1_9`. Additionally, ensure the necessary imports are installed (listed on the first two cells of the notebook).

### Summary of Results

During the exploratory data analysis, the most notable observation I made was that the edge classifications were imbalanced in favor of the negatie classification. I thus focused on developing a model that was able to overcome this challenge. During the training step, I chose the loss function to be the binary cross entropy function with an added weight to the positive case in order to account for the imbalance and encourage the model to properly classify positive edges. During the evaluation step, I compared the model's performance to a selection of naive models that took advantage of the imbalanced nature of the data. Namely, I compared the trained model to a selection of models which always or almost always chose the negatie case. In the end, my model had a 60% lower loss value on the test dataset compared to the best-performing naive algorithm. Although it is impossible to know whether or not this model is "good" given there is no provided context to the dataset, from the loss improvement and the visualizations of classification results, it is clear the provided model performs classifications with non-trivial accuracy. In the conclusion section, I note several further developments which I hope to make in the future.

### Model Design

The GNN model class I created for task, `EdgeModel`, is comprised of two stages. 

In the first stage of a forward pass, convolutions are performed to aggregate features from neighboring nodes and edges. Specifically, the `NNConv` convolution operator, also known as the "edge-conditioned convolution," is used for this convolution stage. This operator collects both node and edge attribute data as it performs a convolution, enabling the model to utilize both the node and edge attributes in the dataset effectively. Between each convolution stage, the RELU operation is applied. Lastly, when initializing `EdgeModel` it is possible to customize he number of layers and hidden layer sizes, but for the model trained in the notebook used the default two layers. At the end of the last convolution layer, each node is associated with a tensor of activation values.

In the second stage, several fully connected layers are used in order to make the final classification predictions for each edge. For each edge, its original edge attribute tensor is concatenated to the activation values tensor of the node which the edge originates from. This tensor is then passed through three linear layers (with RELU in between) to finally generate a single value used to predict the classification for the given edge. This process of creating a classification prediction is applied to every edge (using the same three linear layers) to generate the final prediction tensor for the edge classifications.

The intuition behind this model was that at the end of the convolution stage, each node will possess an activation tensor that contains some aggregate collection of relevant features obtained from neighboring nodes and edges. The fully connected layers will then hopefully be trained to identify pairings of edge attributes and the node aggregate attributes which correlate to positive and negative edge classifications.

### Clarifications on Commits/Additional notes

It is noted in the task document that the commits for this repository may be observed, so I wanted to explain a few unintentional errors in the list of commits for this repository.

 - I decided to work on this task on my sibiling's more powerful computer, but I had some issues with setting up the repository at the start. As a result, I accidentally sent a few commits all titled "First Commit." This are all unintentional, and I noticed these commits too late to undo them.

 - While I was solving issues related to git, I forked one of my previously existing (and irrelevant) repositories to get this repository started. As a result, there are two commits on this rep originating before March; these are completely disconnected from my submission.

 - See my other submission on the software engineering task to see my model included in `gnn_tracking/models/`.
 
 - Otherwise, this task was surprisingly enjoyable and I had a good time. Thank you for your time and consideration.