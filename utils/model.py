import torch
from torch.nn.functional import relu
from torch_geometric.nn import NNConv

class EdgeModel(torch.nn.Module):
    def __init__(self, sample, hidden_layers = [16], intra_hidden = 16):
        """
        Initialization for EdgeModel class
            sample (torch_geometric.data.Data): one element from the database
            hidden_layers (int list): Size of each 
            intra_hidden (int): Size of intermediate 
        Returns: None
        """

        super().__init__()

        # Folder to store all convolutional layers
        self.conv = []

        if len(hidden_layers) == 0:
            # Case when no hidden layers are passed
            nn = self.simple_nn(sample.num_edge_features, intra_hidden, intra_hidden * sample.num_node_features)
            self.conv.append(NNConv(sample.num_node_features, intra_hidden, nn))

        else:
            # Case when hidden layers are passed

            # First convolution NNConv layer uses a simple nn to incorporate the edge attributes into the activation values
            nn = self.simple_nn(sample.num_edge_features, intra_hidden, sample.num_node_features * hidden_layers[0])
            self.conv.append(NNConv(sample.num_node_features, hidden_layers[0], nn))

            for i in range(len(hidden_layers) - 1):
                # Intermediate convolution NNConv layers also use a simple nn
                nn = self.simple_nn(sample.num_edge_features, intra_hidden, hidden_layers[i] * hidden_layers[i + 1])
                self.conv.append(NNConv(hidden_layers[i], hidden_layers[i + 1], nn))    
            
            # Final convolution layer also use a simple nn
            nn = self.simple_nn(sample.num_edge_features, intra_hidden, intra_hidden * hidden_layers[-1])
            self.conv.append(NNConv(hidden_layers[-1], intra_hidden, nn))

        # Last uses edge attributes and the fimal layer of activation values to generate one value for predicting the edge classification.
        # This function has three linear layers and two RELU operations.
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(sample.num_edge_features + intra_hidden, sample.num_edge_features + intra_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(sample.num_edge_features + intra_hidden, intra_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(intra_hidden, intra_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(intra_hidden, 1)
        )

        # Enable torch to recognize self.conv as weights that can be updated
        self.conv_module = torch.nn.ModuleList(self.conv)
    
    def simple_nn(self, a_1, a_2, a_3):
        """
        A function for creating a simple neural network with two linear layers and one layer of RELU
        Parameters:
            a_1 (int): Size of input layer
            a_2 (int): Size of intermediate layer
            a_3 (int): Size of output layer
        Returns: torch.nn.Sequential object containing specified neural network
        """

        return torch.nn.Sequential(
            torch.nn.Linear(a_1, a_2),
            torch.nn.ReLU(),
            torch.nn.Linear(a_2, a_3)
        )

    def forward(self, data):
        """
        Forward pass of the model given a graph
        Parameters:
            data (torch_geometric.data.Data): graph to be passed forward
        Returns: Tensor of length data.num_edges denoting our predictions
        """

        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
    
        for i in range(len(self.conv)):
            # Pass the result of every convolution layer through REUL
            x = relu(self.conv[i](x, edge_index, edge_attr))

        # For each edge, get the activation values assigned to the node which originates this edges and append it to the edge attributes
        src_node_attrs = x[edge_index[0]]
        conv_attributes = torch.cat((edge_attr, src_node_attrs), dim=1)

        # Generate edge scores using the edge_predictor neural network given the convolution attribute arrays previously derived
        edge_scores = self.edge_predictor(conv_attributes)

        # Change 2-dimensional tensor to 1-dimension
        return edge_scores.squeeze()