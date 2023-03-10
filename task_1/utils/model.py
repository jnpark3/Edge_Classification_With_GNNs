import torch
from torch.nn.functional import relu
from torch_geometric.nn import NNConv

class EdgeModel(torch.nn.Module):
    def __init__(self, sample, hidden_layers = [16], intra_hidden = 10):
        super().__init__()

        self.conv = []
        self.linear_weights = torch.randn(2 * sample.num_node_features + sample.num_edge_features)

        if len(hidden_layers) == 0:
            # Case when no hidden layers are passed

            nn = torch.nn.Sequential(
                torch.nn.Linear(sample.num_edge_features, intra_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(intra_hidden, sample.num_node_features)
            )
            self.conv.append(NNConv(sample.num_node_features, 1, nn))

        else:
            # Case when hidden layers are passed

            nn = self.simple_nn(sample.num_edge_features, intra_hidden, sample.num_node_features * hidden_layers[0])
            self.conv.append(NNConv(sample.num_node_features, hidden_layers[0], nn))

            for i in range(len(hidden_layers) - 1):
                nn = self.simple_nn(sample.num_edge_features, intra_hidden, hidden_layers[i] * hidden_layers[i + 1])
                self.conv.append(NNConv(hidden_layers[i], hidden_layers[i + 1], nn))    
            
            nn = self.simple_nn(sample.num_edge_features, intra_hidden, intra_hidden * hidden_layers[-1])
            self.conv.append(NNConv(hidden_layers[-1], intra_hidden, nn))

        # Last layer for making edge predictions from node + edge data
        self.edge_predictor = self.simple_nn(sample.num_edge_features + intra_hidden, intra_hidden, 1)

        self.conv_module = torch.nn.ModuleList(self.conv)
    
    def simple_nn(self, a_1, a_2, a_3):
        return torch.nn.Sequential(
            torch.nn.Linear(a_1, a_2),
            torch.nn.ReLU(),
            torch.nn.Linear(a_2, a_3)
        )

    def forward(self, data):

        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        '''
        # Concatenate the edge attributes with the source node attributes
        '''
    
        for i in range(len(self.conv)):
            x = relu(self.conv[i](x, edge_index, edge_attr))

        src_node_attrs = x[edge_index[0]]
        conv_attributes = torch.cat((edge_attr, src_node_attrs), dim=1)
        edge_scores = self.edge_predictor(conv_attributes)

        return edge_scores.squeeze()