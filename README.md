# Microlearning Project

- Members: Jeeyoung Yoon, Elisee Amegassi, Ian Deal, Shinvanshu Srivastava, Victor Martins, Stephen Chen, Sungjae Cho

## First proposal

Our project is about evaluating the effect that learning rules have on how neural networks internally represent data and how this affects downstream tasks.
We are particularly interested in evaluating how increasingly biologically plausible learning rules affect these representations.

Towards this goal we will be asking:
1. Comparing artificial and biology inspired data embedding methods (how models represent data in the initial layer) and measure this affects task performance?
2. How do different biologically plausible learning rules affect internal representations of a set of models performing the same task? Especially, will a learning rule based on local weight adjustments significantly change what features are represented compared to backprop.
3. How do networks that are trained with biologically plausible learning rules compare to data collected from human image recognition tasks?

To do this we will be constructing a standard network architecture and training it to perform multiple tasks. For each task, an instance of the model will be trained with each of our selected learning rules and a set of metrics will be computed comparing each instance's representations and task performance.

![image](image/README/workflow-1.png "Workflow-1")

## Baseline model

```python
# @title MNIST Model
# modified and reorganized from https://github.com/pytorch/examples/blob/main/mnist/main.py
#lenet model
# Dropout has been left out.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #self.dropout1 = nn.Dropout(0.25)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

## Progress report for NeuroAI

Our project aims to compare the backpropagation learning rule with more biologically plausible learning rules in terms of their mechanism, performance, and internal representations. Our progress and next steps are as follows:

The **deep predictive coding network** (Wen et al., 2018) has been employed from the authors’ GitHub repository. The implementation has been adapted to the newest PyTorch version and the MNIST and CIFAR-10 datasets of our interest. The network achieved 99% test accuracy for MNIST and 95% for CIFAR-10. Further analysis on the representations of each layer is to be continued.

The **direct difference target propagation learning rule** (Meulemans et al., 2020) has been implemented in the standard convolutional network and trained on MNIST, RDM’s and dimensionality reduction based metrics have been computed to compare the the learning rule’s effect on the internal representations within the model compared to backpropagation.

The **non-negative matrix factorization** (Seung & Lee, 1999) has been implemented on the MNIST dataset. This feature representation was compared with that of a CNN trained with backpropagation. Further analysis on the CIFAR-10 dataset will be conducted to compare the resemblance of these two approaches with the biological visual system's stimuli representation.

The **Kolen-Pollack approach** (Kolen & Pollack, 1994) updates forward and backward weights using only locally available information. Neuromatch tutorials implemented this approach in a single fully connected layer, and we extended it to multiple layers. Then we compared the representational geometries from this method to those in neural networks trained with error backpropagation, weight mirror, and feedback alignment learning rules. Going forward we aim to further extend our modeling capabilities with it to convolutional layers.

### References

1. Wen, H., Han, K., Shi, J., Zhang, Y., Culurciello, E., & Liu, Z. (2018). Deep predictive coding network for object recognition. International Conference on Machine Learning.
2. Ernoult, M., Normandin, F., Moudgil, A., Spinney, S., Belilovsky, E., Rish, I., Richards, B.A., & Bengio, Y. (2022). Towards scaling difference target propagation by learning backprop targets. International Conference on Machine Learning.
3. Lee, D., & Seung, H.S. (1999). Learning the parts of objects by non-negative matrix factorization. Nature.
4. Kolen, J.F., & Pollack, J.B. (1994). Backpropagation without weight transport. Proceedings of 1994 IEEE International Conference on Neural Networks.