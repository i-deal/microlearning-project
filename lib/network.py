import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from lib import utils
import pandas as pd
import warnings

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class DTPLayer(nn.Module):
    """ An abstract base class for a layer of an MLP that will be trained by the
    differece target propagation method. Child classes should specify which
    activation function is used.

    Attributes:
        weights (torch.Tensor): The forward weight matrix :math:`W` of the layer
        bias (torch.Tensor): The forward bias :math:`\mathbf{b}`
            of the layer.
            Attribute is ``None`` if argument ``bias`` was passed as ``None``
            in the constructor.
        feedback_weights (torch.Tensor): The feedback weight matrix :math:`Q`
            of the layer. Warning: if we use the notation of the theoretical
            framework, the feedback weights are actually $Q_{i-1}$ from the
            previous layer!! We do this because this makes the implementation
            of the reconstruction loss and training the feedback weights much
            easier (as g_{i-1} and hence Q_{i-1} needs to approximate
            f_i^{-1}). However for the direct feedback connection layers, it
            might be more logical to let the feedbackweights represent Q_i
            instead of Q_{i-1}, as now only direct feedback connections exist.
        feedback_bias (torch.Tensor): The feedback bias :math:`\mathbf{b}`
            of the layer.
            Attribute is ``None`` if argument ``bias`` was passed as ``None``
            in the constructor.
        forward_requires_grad (bool): Flag indicating whether the computational
            graph with respect to the forward parameters should be saved. This
            flag should be True if you want to compute BP or GN updates. For
            TP updates, computational graphs are not needed (custom
            implementation by ourselves)
        reconstruction_loss (float): The reconstruction loss of this layer
            evaluated at the current mini-batch.
        forward_activation (str): String indicating the forward nonlinear
            activation function used by the layer. Choices: 'tanh', 'relu',
            'linear'.
        feedback_activation (str): String indicating the feedback nonlinear
            activation function used by the layer. Choices: 'tanh', 'relu',
            'linear'.
        target (torch.Tensor or None): The target for this layer on the current
            minibatch. During normal training, it is not needed
            to save the targets so this attribute will stay None. If the user
            wants to compute the angle between (target - activation) and a
            BP update or GN update, the target needs to be saved in the layer
            object to use later on to compute the angles. The saving happens in
            the backward method of the network.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive
            bias.
        forward_requires_grad (bool): Flag indicating whether the forward
            parameters require gradients that can be computed with autograd.
            This might be needed when comparing the DTP updates with BP updates
            and GN updates.
        forward_activation (str): String indicating the forward nonlinear
            activation function used by the layer. Choices: 'tanh', 'relu',
            'linear'.
        feedback_activation (str): String indicating the feedback nonlinear
            activation function used by the layer. Choices: 'tanh', 'relu',
            'linear'.
    """

    def __init__(self, in_features, out_features, bias=True,
                 forward_requires_grad=False, forward_activation='tanh',
                 feedback_activation='tanh', initialization='orthogonal',device=device):
        nn.Module.__init__(self)
        self.device = device
        self._weights = nn.Parameter(torch.Tensor(out_features, in_features),
                                     requires_grad=forward_requires_grad)
        self._feedbackweights = nn.Parameter(torch.Tensor(in_features,
                                                          out_features),
                                             requires_grad=False)
        if bias:
            self._bias = nn.Parameter(torch.Tensor(out_features),
                                      requires_grad=forward_requires_grad)
            self._feedbackbias = nn.Parameter(torch.Tensor(in_features),
                                              requires_grad=False)
        else:
            self._bias = None
            self._feedbackbias = None

        # Initialize the weight matrices following Lee 2015
        # TODO: try other initializations, such as the special one to mimic
        # batchnorm
        if initialization == 'orthogonal':
            gain = np.sqrt(6. / (in_features + out_features))
            nn.init.orthogonal_(self._weights, gain=gain)
            nn.init.orthogonal_(self._feedbackweights, gain=gain)
        elif initialization == 'xavier':
            nn.init.xavier_uniform_(self._weights)
            nn.init.xavier_uniform_(self._feedbackweights)
        elif initialization == 'xavier_normal':
            nn.init.xavier_normal_(self._weights)
            nn.init.xavier_normal_(self._feedbackweights)
        elif initialization == 'teacher':
            nn.init.xavier_normal_(self._weights, gain=3.)
            nn.init.xavier_normal_(self._feedbackweights)
        else:
            raise ValueError('Provided weight initialization "{}" is not '
                             'supported.'.format(initialization))

        if bias:
            nn.init.constant_(self._bias, 0)
            nn.init.constant_(self._feedbackbias, 0)

        self._activations = None
        self._linearactivations = None
        self._reconstruction_loss = None
        self._forward_activation = forward_activation
        self._feedback_activation = feedback_activation
        self._target = None

    @property
    def weights(self):
        """Getter for read-only attribute :attr:`weights`."""
        return self._weights

    @property
    def bias(self):
        """Getter for read-only attribute :attr:`bias`."""
        return self._bias

    @property
    def feedbackweights(self):
        """Getter for read-only attribute :attr:`feedbackweights`."""
        return self._feedbackweights

    @property
    def feedbackbias(self):
        """Getter for read-only attribute :attr:`feedbackbias`."""
        return self._feedbackbias

    @property
    def activations(self):
        """Getter for read-only attribute :attr:`activations` """
        return self._activations

    @activations.setter
    def activations(self, value):
        """ Setter for the attribute activations"""
        self._activations = value

    @property
    def linearactivations(self):
        """Getter for read-only attribute :attr:`linearactivations` """
        return self._linearactivations

    @linearactivations.setter
    def linearactivations(self, value):
        """Setter for the attribute :attr:`linearactivations` """
        self._linearactivations = value

    @property
    def reconstruction_loss(self):
        """ Getter for attribute reconstruction_loss."""
        return self._reconstruction_loss

    @reconstruction_loss.setter
    def reconstruction_loss(self, value):
        """ Setter for attribute reconstruction_loss."""
        self._reconstruction_loss = value

    @property
    def forward_activation(self):
        """ Getter for read-only attribute forward_activation"""
        return self._forward_activation

    @property
    def feedback_activation(self):
        """ Getter for read-only attribute feedback_activation"""
        return self._feedback_activation

    @property
    def target(self):
        """ Getter for attribute target"""
        return self._target

    @target.setter
    def target(self, value):
        """ Setter for attribute target"""
        self._target = value

    def get_forward_parameter_list(self):
        """ Return forward weights and forward bias if applicable"""
        parameterlist = []
        parameterlist.append(self.weights)
        if self.bias is not None:
            parameterlist.append(self.bias)
        return parameterlist

    def forward_activationfunction(self, x):
        """ Element-wise forward activation function"""
        if self.forward_activation == 'tanh':
            return torch.tanh(x)
        elif self.forward_activation == 'relu':
            return F.relu(x)
        elif self.forward_activation == 'linear':
            return x
        elif self.forward_activation == 'leakyrelu':
            return F.leaky_relu(x, 0.2)
        elif self.forward_activation == 'sigmoid':
            return torch.sigmoid(x)
        else:
            raise ValueError('The provided forward activation {} is not '
                             'supported'.format(self.forward_activation))

    def feedback_activationfunction(self, x):
        """ Element-wise feedback activation function"""
        if self.feedback_activation == 'tanh':
            return torch.tanh(x)
        elif self.feedback_activation == 'relu':
            return F.relu(x)
        elif self.feedback_activation == 'linear':
            return x
        elif self.feedback_activation == 'leakyrelu':
            return F.leaky_relu(x, 5)
        elif self.feedback_activation == 'sigmoid':
            if torch.sum(x < 1e-12) > 0 or torch.sum(x > 1-1e-12) > 0:
                warnings.warn('Input to inverse sigmoid is out of'
                                 'bound: x={}'.format(x))
            inverse_sigmoid = torch.log(x/(1-x))
            if utils.contains_nan(inverse_sigmoid):
                raise ValueError('inverse sigmoid function outputted a NaN')
            return torch.log(x/(1-x))
        else:
            raise ValueError('The provided feedback activation {} is not '
                             'supported'.format(self.feedback_activation))

    def compute_vectorized_jacobian(self, a):
        """ Compute the vectorized Jacobian of the forward activation function,
        evaluated at a. The vectorized Jacobian is the vector with the diagonal
        elements of the real Jacobian, as it is a diagonal matrix for element-
        wise functions. As a is a minibatch, the output will also be a
        mini-batch of vectorized Jacobians (thus a matrix).
        Args:
            a (torch.Tensor): linear activations
        """
        if self.forward_activation == 'tanh':
            return 1. - torch.tanh(a)**2
        elif self.forward_activation == 'relu':
            J = torch.ones_like(a)
            J[a < 0.] = 0.
            return J
        elif self.forward_activation == 'leakyrelu':
            J = torch.ones_like(a)
            J[a < 0.] = 0.2
            return J
        elif self.forward_activation == 'linear':
            return torch.ones_like(a)
        elif self.forward_activation == 'sigmoid':
            s = torch.sigmoid(a)
            return s * (1 - s)
        else:
            raise ValueError('The provided forward activation {} is not '
                             'supported'.format(self.forward_activation))

    def requires_grad(self):
        """ Set require_grad attribute of the activations of this layer to
        True, such that the gradient will be saved in the activation tensor."""
        self._activations.requires_grad = True

    def forward(self, x):
        """Compute the output activation of the layer.

        This method applies first a linear mapping with the
        parameters ``weights`` and ``bias``, after which it applies the
        forward activation function.

        Args:
            x: A mini-batch of size B x in_features with input activations from
            the previous layer or input.

        Returns:
            The mini-batch of output activations of the layer.
        """

        h = x.mm(self.weights.t().to(self.device))
        if self.bias is not None:
            h += self.bias.unsqueeze(0).expand_as(h).to(self.device)
        self.linearactivations = h

        self.activations = self.forward_activationfunction(h)
        return self.activations

    def dummy_forward(self, x):
        """ Same as the forward method, besides that the activations and
        linear activations are not saved in self."""
        h = x.mm(self.weights.t().to(self.device))
        if self.bias is not None:
            h += self.bias.unsqueeze(0).expand_as(h).to(self.device)
        h = self.forward_activationfunction(h)
        return h

    def dummy_forward_linear(self, x):
        """ Propagate the input of the layer forward to the linear activation
        of the current layer (so no nonlinearity applied), without saving the
        linear activations."""
        a = x.mm(self.weights.t().to(self.device))
        if self.bias is not None:
            a += self.bias.unsqueeze(0).expand_as(a).to(self.device)

        return a

    def propagate_backward(self, h):
        """ Propagate the activation h backward through the backward mapping
        function g(h) = t(Q*h + d)
        Args:
            h (torch.Tensor): a mini-batch of activations
        """
        h = h.mm(self.feedbackweights.t().to(self.device))
        if self.feedbackbias is not None:
            h += self.feedbackbias.unsqueeze(0).expand_as(h).to(self.device)
        return self.feedback_activationfunction(h)


    def backward(self, h_target, h_previous, h_current):
        """Compute the target activation for the previous layer, based on the
        provided target.


        Args:
            h_target: a mini-batch of the provided targets for this layer.
            h_previous: the activations of the previous layer, used for the
                DTP correction term.
            h_current: the activations of the current layer, used for the
                DTP correction term.

        Returns:
            h_target_previous: The mini-batch of target activations for
                the previous layer.
        """

        h_target_previous = self.propagate_backward(h_target)
        h_tilde_previous = self.propagate_backward(h_current)
        h_target_previous = h_target_previous + h_previous - h_tilde_previous

        return h_target_previous

    def compute_forward_gradients(self, h_target, h_previous, norm_ratio=1.):
        """ Compute the gradient of the forward weights and bias, based on the
        local mse loss between the layer activation and layer target.
        The gradients are saved in the .grad attribute of the forward weights
        and forward bias.
        Args:
            h_target (torch.Tensor): the DTP target of the current layer
            h_previous (torch.Tensor): the rate activation of the previous
                layer
            norm_ratio (float): Depreciated.
        """

        if self.forward_activation == 'linear':
            teaching_signal = 2 * (self.activations - h_target)
        else:
            vectorized_jacobians = self.compute_vectorized_jacobian(
                self.linearactivations)
            teaching_signal = 2 * vectorized_jacobians * (
                    self.activations - h_target)
        batch_size = h_target.shape[0]
        bias_grad = teaching_signal.mean(0).to(self.device)
        weights_grad = 1./batch_size * teaching_signal.t().mm(h_previous)

        temp_device = self.device
        if self.bias is not None:
            if self.bias.get_device() == -1:
                temp_device = 'cpu'
                
            self._bias.grad = bias_grad.detach().to(temp_device)
            self._bias.grad.to(self.device)
        self._weights.grad = weights_grad.detach().to(temp_device)
        self._weights.grad.to(self.device)

    def set_feedback_requires_grad(self, value):
        """
        Sets the 'requires_grad' attribute of the feedback weights and bias to
        the given value
        Args:
            value (bool): True or False
        """
        if not isinstance(value, bool):
            raise TypeError('The given value should be a boolean.')
        self._feedbackweights.requires_grad = value
        if self._feedbackbias is not None:
            self._feedbackbias.requires_grad = value

    def compute_feedback_gradients(self, h_previous_corrupted, sigma):
        """ Compute the gradient of the backward weights and bias, based on the
        local reconstruction loss of a corrupted sample of the previous layer
        activation. The gradients are saved in the .grad attribute of the
        feedback weights and feedback bias."""

        self.set_feedback_requires_grad(True)

        h_current = self.dummy_forward(h_previous_corrupted)
        h = self.propagate_backward(h_current)

        if sigma < 0.02:
            scale = 1/0.02**2
        else:
            scale = 1/sigma**2
        reconstruction_loss = scale * F.mse_loss(h, h_previous_corrupted)

        self.save_feedback_gradients(reconstruction_loss)

        self.set_feedback_requires_grad(False)

    def save_feedback_gradients(self, reconstruction_loss):
        """
        Compute the gradients of the reconstruction_loss with respect to the
        feedback parameters by help of autograd and save them in the .grad
        attribute of the feedback parameters
        Args:
            reconstruction_loss: the reconstruction loss

        """
        self.reconstruction_loss = reconstruction_loss.item()
        if self.feedbackbias is not None:
            grads = torch.autograd.grad(reconstruction_loss, [
                self.feedbackweights, self.feedbackbias], retain_graph=False)
            self._feedbackbias.grad = grads[1].detach()
        else:
            grads = torch.autograd.grad(reconstruction_loss,
                                        self.feedbackweights,
                                        retain_graph=False
                                        )
        self._feedbackweights.grad = grads[0].detach()

    def compute_bp_update(self, loss, retain_graph=False):
        """ Compute the error backpropagation update for the forward
        parameters of this layer, based on the given loss.
        Args:
            loss (nn.Module): network loss
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        """

        if self.bias is not None:
            grads = torch.autograd.grad(loss, [self.weights, self.bias],
                                        retain_graph=retain_graph)
        else:
            grads = torch.autograd.grad(loss, self.weights,
                                        retain_graph=retain_graph)

        return grads

    def compute_gn_update(self, output_activation, loss, damping=0.,
                          retain_graph=False):
        """
        Compute the Gauss Newton update for the parameters of this layer based
        on the current minibatch.
        Args:
            output_activation (torch.Tensor): The tensor containing the output
                activations of the network for the current mini-batch
            loss (torch.Tensor): The 0D tensor containing the loss value of the
                current mini-batch.
            damping (float): the damping coefficient to damp the GN curvature
                matrix J^TJ. Default: 0.
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.

        Returns (tuple): A tuple containing the Gauss Newton updates for the
            forward parameters (at index 0 the weight updates, at index 1
            the bias updates if the layer has a bias)

        """
        output_error = torch.autograd.grad(loss, output_activation,
                                           retain_graph=True)[0].detach()
        parameters = self.get_forward_parameters()
        jacobian = utils.compute_jacobian(parameters, output_activation,
                                          retain_graph=retain_graph)

        gn_updates = utils.compute_damped_gn_update(jacobian, output_error,
                                                    damping)

        if self.bias is not None:
            weight_update_flattened = gn_updates[:self.weights.numel(), :]
            bias_update_flattened = gn_updates[self.weights.numel():, :]
            weight_update = weight_update_flattened.view_as(self.weights)
            bias_update = bias_update_flattened.view_as(self.bias)
            return (weight_update, bias_update)
        else:
            weight_update = gn_updates.view(self.weights.shape)
            return (weight_update, )

    def compute_gn_activation_updates(self, output_activation, loss,
                                      damping=0., retain_graph=False,
                                      linear=False):
        """
        Compute the Gauss Newton update for activations of the layer. Target
        propagation tries to approximate these updates by the difference between
        the layer targets and the layer activations.
        Args:
            output_activation (torch.Tensor): The tensor containing the output
                activations of the network for the current mini-batch
            loss (torch.Tensor): The 0D tensor containing the loss value of the
                current mini-batch.
            damping (float): the damping coefficient to damp the GN curvature
                matrix J^TJ. Default: 0.
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
            linear (bool): Flag indicating whether the GN update for the
                linear activations should be computed instead of for the
                nonlinear activations.

        Returns (torch.Tensor): A tensor containing the Gauss-Newton updates
            for the layer activations of the current mini-batch. The size is
            minibatchsize x layersize

        """
        output_error = torch.autograd.grad(loss, output_activation,
                                           retain_graph=True)[0].detach()
        if linear:
            activations = self.linearactivations
        else:
            activations = self.activations
        activations_updates = torch.Tensor(activations.shape)
        layersize = activations.shape[1]

        # compute the GN update for each batch sample separate, as we are now
        # computing 'updates' for the activations of the layer instead of the
        # parameters of the layers
        for batch_idx in range(activations.shape[0]):
            # print(batch_idx)
            #  compute jacobian for one batch sample:
            if batch_idx == activations.shape[0] - 1:
                retain_graph_flag = retain_graph
            else:
                # if not yet at the end of the batch, we should retain the graph
                # used for computing the jacobian, as the graph needs to be
                # reused for the computing the jacobian of the next batch sample
                retain_graph_flag = True
            jacobian = utils.compute_jacobian(activations,
                                              output_activation[batch_idx,
                                              :],
                                            retain_graph=retain_graph_flag)
            # torch.autograd.grad only accepts the original input tensor,
            # not a subpart of it. Thus we compute the jacobian to all the
            # batch samples from activations and then select the correct
            # part of it
            jacobian = jacobian[:, batch_idx*layersize:
                                   (batch_idx+1)*layersize]

            gn_updates = utils.compute_damped_gn_update(jacobian,
                                                output_error[batch_idx, :],
                                                        damping)
            activations_updates[batch_idx, :] = gn_updates.view(-1)
        return activations_updates

    def compute_bp_activation_updates(self, loss, retain_graph=False,
                                      linear=False):
        """ Compute the error backpropagation teaching signal for the
        activations of this layer, based on the given loss.
        Args:
            loss (nn.Module): network loss
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
            linear (bool): Flag indicating whether the GN update for the
                linear activations should be computed instead of for the
                nonlinear activations.
        Returns (torch.Tensor): A tensor containing the BP updates for the layer
            activations for the current mini-batch.

                """

        if linear:
            activations = self.linearactivations
        else:
            activations = self.activations
        grads = torch.autograd.grad(loss, activations,
                                    retain_graph=retain_graph)[0].detach()
        return grads

    def compute_gnt_updates(self, output_activation, loss, h_previous, damping=0.,
                            retain_graph=False, linear=False):
        """ Compute the angle with the GNT updates for the parameters of the
        network."""
        gn_activation_update = self.compute_gn_activation_updates(output_activation=output_activation,
                                                                  loss=loss,
                                                                  damping=damping,
                                                                  retain_graph=retain_graph,
                                                                  linear=linear)

        if not linear:
            vectorized_jacobians = self.compute_vectorized_jacobian(
                self.linearactivations)
            teaching_signal = 2 * vectorized_jacobians * (
                        gn_activation_update)
        else:
            teaching_signal = 2 * gn_activation_update

        batch_size = self.activations.shape[0]
        bias_grad = teaching_signal.mean(0)
        weights_grad = 1. / batch_size * teaching_signal.t().mm(h_previous)

        if self.bias is not None:
            return (weights_grad, bias_grad)
        else:
            return (weights_grad, )

    def compute_nullspace_relative_norm(self, output_activation, retain_graph=False):
        """ Compute the norm of the components of weights.grad that are in the nullspace
        of the jacobian of the output with respect to weights, relative to the norm of
        weights.grad."""
        J = utils.compute_jacobian(self.weights, output_activation,
                                   structured_tensor=False,
                                   retain_graph=retain_graph)
        weights_update_flat = self.weights.grad.view(-1)
        relative_norm = utils.nullspace_relative_norm(J, weights_update_flat)
        return relative_norm

    def save_logs(self, writer, step, name, no_gradient=False,
                  no_fb_param=False):
        """
        Save logs and plots of this layer on tensorboardX
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            name (str): The name of the layer
            no_gradient (bool): flag indicating whether we should skip saving
                the gradients of the feedback weights.
            no_fb_param (bool): don't log the feedback parameters

        """
        forward_weights_norm = torch.norm(self.weights)
        writer.add_scalar(tag='{}/forward_weights_norm'.format(name),
                          scalar_value=forward_weights_norm,
                          global_step=step)
        if self.weights.grad is not None:
            forward_weights_gradients_norm = torch.norm(self.weights.grad)
            writer.add_scalar(tag='{}/forward_weights_gradients_norm'.format(name),
                              scalar_value=forward_weights_gradients_norm,
                              global_step=step)
        if self.bias is not None:
            forward_bias_norm = torch.norm(self.bias)

            writer.add_scalar(tag='{}/forward_bias_norm'.format(name),
                              scalar_value=forward_bias_norm,
                              global_step=step)
            if self.bias.grad is not None:
                forward_bias_gradients_norm = torch.norm(self.bias.grad)
                writer.add_scalar(tag='{}/forward_bias_gradients_norm'.format(name),
                                  scalar_value=forward_bias_gradients_norm,
                                  global_step=step)
        if not no_fb_param:
            feedback_weights_norm = torch.norm(self.feedbackweights)
            writer.add_scalar(tag='{}/feedback_weights_norm'.format(name),
                              scalar_value=feedback_weights_norm,
                              global_step=step)
            if self.feedbackbias is not None:
                feedback_bias_norm = torch.norm(self.feedbackbias)
                writer.add_scalar(tag='{}/feedback_bias_norm'.format(name),
                                  scalar_value=feedback_bias_norm,
                                  global_step=step)

            if not no_gradient and self.feedbackweights.grad is not None:
                feedback_weights_gradients_norm = torch.norm(
                    self.feedbackweights.grad)
                writer.add_scalar(
                    tag='{}/feedback_weights_gradients_norm'.format(name),
                    scalar_value=feedback_weights_gradients_norm,
                    global_step=step)
                if self.feedbackbias is not None:
                    feedback_bias_gradients_norm = torch.norm(
                        self.feedbackbias.grad)
                    writer.add_scalar(
                        tag='{}/feedback_bias_gradients_norm'.format(name),
                        scalar_value=feedback_bias_gradients_norm,
                        global_step=step)

    def save_feedback_batch_logs(self, writer, step, name, no_gradient=False,
                                 init=False):
        """
        Save logs for one minibatch.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            name (str): The name of the layer
            no_gradient (bool): flag indicating whether we should skip saving
                the gradients of the feedback weights.
            init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
        """
        if not init:
            if not no_gradient and self.reconstruction_loss is not None:
                writer.add_scalar(
                    tag='{}/reconstruction_loss'.format(name),
                    scalar_value=self.reconstruction_loss,
                    global_step=step)
        else:
            if not no_gradient and self.reconstruction_loss is not None:
                writer.add_scalar(
                    tag='{}/reconstruction_loss_init'.format(name),
                    scalar_value=self.reconstruction_loss,
                    global_step=step)

    def get_forward_parameters(self):
        """ Return a list containing the forward parameters."""
        if self.bias is not None:
            return [self.weights, self.bias]
        else:
            return [self.weights]

    def get_forward_gradients(self):
        """ Return a tuple containing the gradients of the forward
        parameters."""

        if self.bias is not None:
            return (self.weights.grad, self.bias.grad)
        else:
            return (self.weights.grad, )

class DTPDRLLayer(DTPLayer):
    """ Class for difference target propagation combined with the
    difference reconstruction loss, but without the minimal norm update."""

    def compute_feedback_gradients(self, h_previous_corrupted,
                                   h_current_reconstructed,
                                   h_previous, sigma):
        """
        Compute the gradient of the feedback weights and bias, based on the
        difference reconstruction loss (p16 in theoretical framework). The
        gradients are saved in the .grad attribute of the feedback weights and
        feedback bias.
        Args:
            h_previous_corrupted (torch.Tensor): the initial corrupted sample
                of the previous layer that was propagated forward to the output.
            h_current_reconstructed (torch.Tensor): The reconstruction of the
                corrupted sample (by propagating it backward again in a DTP-like
                fashion to this layer)
            h_previous (torch.Tensor): the initial non-corrupted sample of the
                previous layer
        """
        self.set_feedback_requires_grad(True)

        h_previous_reconstructed = self.backward(h_current_reconstructed,
                                             h_previous,
                                             self.activations)
        if sigma <= 0:
            raise ValueError('Sigma should be greater than zero when using the'
                             'difference reconstruction loss. Given sigma = '
                             '{}'.format(sigma))

        scale = 1/sigma**2
        reconstruction_loss = scale * F.mse_loss(h_previous_corrupted,
                                         h_previous_reconstructed)
        self.save_feedback_gradients(reconstruction_loss)

        self.set_feedback_requires_grad(False)

class DTPNetwork(nn.Module):
    """ A multilayer perceptron (MLP) network that will be trained by the
    difference target propagation (DTP) method.

    Attributes:
        layers (nn.ModuleList): a ModuleList with the layer objects of the MLP
        depth: the depth of the network (# hidden layers + 1)
        input (torch.Tensor): the input minibatch of the current training
                iteration. We need
                to save this tensor for computing the weight updates for the
                first hidden layer
        sigma: standard deviation of the gaussian that corrupts layer
                activations for computing the reconstruction losses.
        update_idx (None or int): the layer index of which the layer parameters
            are updated for the current mini-batch, when working in a randomized
            setting. If the randomized setting is not used, it is equal to None.

    Args:
        n_in: input dimension (flattened input assumed)
        n_hidden: list with hidden layer dimensions
        n_out: output dimension
        activation: activation function indicator for the hidden layers
        output_activation: activation function indicator for the output layer
        bias: boolean indicating whether the network uses biases or not
        sigma: standard deviation of the gaussian that corrupts layer
                activations for computing the reconstruction losses.
        forward_requires_grad (bool): Flag indicating whether the forward
            parameters require gradients that can be computed with autograd.
            This might be needed when comparing the DTP updates with BP updates
            and GN updates.
        initialization (str): the initialization method used for the forward
                and feedback matrices of the layers


    """

    def __init__(self, n_in, n_hidden, n_out, activation='relu',
                 output_activation='linear', bias=True, sigma=0.36,
                 forward_requires_grad=False,
                 initialization='orthogonal',
                 fb_activation='relu',
                 plots=None):
        nn.Module.__init__(self)

        self._depth = len(n_hidden) + 1
        self._layers = self.set_layers(n_in, n_hidden, n_out, activation,
                                       output_activation, bias,
                                       forward_requires_grad,
                                       initialization,
                                       fb_activation)
        self._input = None
        self._sigma = sigma
        self._forward_requires_grad = forward_requires_grad
        self._update_idx = None
        self._plots = plots
        if plots is not None:
            self.bp_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gn_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gnt_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.bp_activation_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gn_activation_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])

            self.reconstruction_loss_init = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.reconstruction_loss = pd.DataFrame(columns=[i for i in range(0, self._depth)])

            self.td_activation = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gn_activation = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.bp_activation = pd.DataFrame(columns=[i for i in range(0, self._depth)])

            self.nullspace_relative_norm = pd.DataFrame(columns=[i for i in range(0, self._depth)])



    def set_layers(self, n_in, n_hidden, n_out, activation, output_activation,
                   bias, forward_requires_grad, initialization,
                   fb_activation):
        """
        Create the layers of the network and output them as a ModuleList.
        Args:
            n_in: input dimension (flattened input assumed)
            n_hidden: list with hidden layer dimensions
            n_out: output dimension
            activation: activation function indicator for the hidden layers
            output_activation: activation function indicator for the output
                layer
            bias: boolean indicating whether the network uses biases or not
            forward_requires_grad (bool): Flag indicating whether the forward
            parameters require gradients that can be computed with autograd.
            This might be needed when comparing the DTP updates with BP updates
            and GN updates.
            initialization (str): the initialization method used for the forward
                and feedback matrices of the layers
            fb_activation (str): activation function indicator for the feedback
                path of the hidden layers

        """
        n_all = [n_in] + n_hidden + [n_out]
        layers = nn.ModuleList()
        for i in range(1, len(n_all) - 1):
            layers.append(
                DTPLayer(n_all[i - 1], n_all[i], bias=bias,
                         forward_activation=activation,
                         feedback_activation=fb_activation,
                         forward_requires_grad=forward_requires_grad,
                         initialization=initialization
                         ))
        layers.append(DTPLayer(n_all[-2], n_all[-1], bias=bias,
                               forward_activation=output_activation,
                               feedback_activation=fb_activation,
                               forward_requires_grad=forward_requires_grad,
                               initialization=initialization))
        return layers

    @property
    def depth(self):
        """Getter for read-only attribute :attr:`depth`."""
        return self._depth

    @property
    def layers(self):
        """Getter for read-only attribute :attr:`layers`."""
        return self._layers

    @property
    def sigma(self):
        """ Getter for read-only attribute sigma"""
        return self._sigma

    @property
    def input(self):
        """ Getter for attribute input."""
        return self._input

    @input.setter
    def input(self, value):
        """ Setter for attribute input."""
        self._input = value

    @property
    def forward_requires_grad(self):
        """ Getter for read-only attribute forward_requires_grad"""
        return self._forward_requires_grad

    @property
    def update_idx(self):
        """ Getter for attribute update_idx"""
        return self._update_idx

    @update_idx.setter
    def update_idx(self, value):
        """Setter for attribute update_idx"""
        self._update_idx = value

    def forward(self, x):
        """ Propagate the input forward through the MLP network.

        Args:
            x: the input to the network

        returns:
            y: the output of the network
            """
        self.input = x
        y = x

        for layer in self.layers:
            y = layer.forward(y)

        # the output of the network requires a gradient in order to compute the
        # target (in compute_output_target() )
        if y.requires_grad == False:
            y.requires_grad = True

        return y

    def compute_output_target(self, loss, target_lr):
        """
        Compute the output target.
        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer

        Returns: Mini-batch of output targets
        """
        output_activations = self.layers[-1].activations

        gradient = torch.autograd.grad(loss, output_activations,
                                       retain_graph=self.forward_requires_grad)\
                                        [0].detach()
        output_targets = output_activations - \
                         target_lr*gradient
        return output_targets

    def propagate_backward(self, h_target, i):
        """
        Propagate the output target backwards to layer i in a DTP-like fashion.
        Args:
            h_target (torch.Tensor): the output target
            i: the layer index to which the target must be propagated

        Returns: the target for layer i

        """
        for k in range(self.depth-1, i, -1):
            h_current = self.layers[k].activations
            h_previous = self.layers[k-1].activations
            h_target = self.layers[k].backward(h_target, h_previous, h_current)
        return h_target

    def backward_random(self, loss, target_lr, i, save_target=False,
                        norm_ratio=1.):
        """ Propagate the output target backwards through the network until
        layer i. Based on this target, compute the gradient of the forward
        weights and bias of layer i and save them in the parameter tensors.
        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer
            i: layer index to which the target needs to be propagated and the
                gradients need to be computed
            save_target (bool): flag indicating whether the target should be
                saved in the layer object for later use.
            norm_ratio (float): will only be used in children of DTPLayer for
                the minimal_norm update
        """

        self.update_idx = i

        h_target = self.compute_output_target(loss, target_lr)

        h_target = self.propagate_backward(h_target, i)

        if save_target:
            self.layers[i].target = h_target

        if i == 0: # first hidden layer needs to have the input
                   # for computing gradients
            self.layers[i].compute_forward_gradients(h_target, self.input,
                                                     norm_ratio=norm_ratio)
        else:
            self.layers[i].compute_forward_gradients(h_target,
                                                 self.layers[i-1].activations,
                                                     norm_ratio=norm_ratio)

    def backward_all(self, output_target, save_target=False, norm_ratio=1.):
        """ Propagate the output_target backwards through all the layers. Based
        on these local targets, compute the gradient of the forward weights and
        biases of all layers.

        Args:
            output_target (torch.Tensor): a mini-batch of targets for the
                output layer.
            save_target (bool): flag indicating whether the target should be
                saved in the layer object for later use.
            norm_ratio (float): will only be used in children of DTPLayer for
                the minimal_norm update
            """

        h_target = output_target

        if save_target:
            self.layers[-1].target = h_target
        for i in range(self.depth-1, 0, -1):
            h_current = self.layers[i].activations
            h_previous = self.layers[i-1].activations
            self.layers[i].compute_forward_gradients(h_target, h_previous,
                                                     norm_ratio=norm_ratio)
            h_target = self.layers[i].backward(h_target, h_previous, h_current)
            if save_target:
                self.layers[i-1].target = h_target

        self.layers[0].compute_forward_gradients(h_target, self.input,
                                                 norm_ratio=norm_ratio)

    def backward(self, loss, target_lr, save_target=False, norm_ratio=1.):
        """ Compute and propagate the output_target backwards through all the
        layers. Based on these local targets, compute the gradient of the
        forward weights and biases of all layers.

        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer
            save_target (bool): flag indicating whether the target should be
                saved in the layer object for later use.
            norm_ratio (float): will only be used in children of DTPLayer for
                the minimal_norm update
            """

        output_target = self.compute_output_target(loss, target_lr)
        self.backward_all(output_target, save_target, norm_ratio=norm_ratio)

    def compute_feedback_gradients(self):
        """ Compute the local reconstruction loss for each layer and compute
        the gradient of those losses with respect to
        the feedback weights and biases. The gradients are saved in the
        feedback parameter tensors."""

        for i in range(1, self.depth):
            h_corrupted = self.layers[i-1].activations + \
                    self.sigma * torch.randn_like(self.layers[i-1].activations)
            self.layers[i].compute_feedback_gradients(h_corrupted, self.sigma)

    def get_forward_parameter_list(self):
        """
        Args:
            freeze_ouptut_layer (bool): flag indicating whether the forward
                parameters of the output layer should be excluded from the
                returned list.
        Returns: a list with all the forward parameters (weights and biases) of
            the network.

        """
        parameterlist = []
        for layer in self.layers:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_reduced_forward_parameter_list(self):
        """
        Returns: a list with all the forward parameters of the network, except
        from the ones of the output layer.
        """
        parameterlist = []
        for layer in self.layers[:-1]:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_forward_parameters_last_two_layers(self):
        parameterlist = []
        for layer in self.layers[-2:]:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_forward_parameters_last_three_layers(self):
        parameterlist = []
        for layer in self.layers[-3:]:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_forward_parameters_last_four_layers(self):
        parameterlist = []
        for layer in self.layers[-4:]:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_forward_parameter_list_first_layer(self):
        """
        Returns: a list with only the forward parameters of the first layer.
        """
        parameterlist = []
        parameterlist.append(self.layers[0].weights)
        if self.layers[0].bias is not None:
            parameterlist.append(self.layers[0].bias)
        return parameterlist

    def get_feedback_parameter_list(self):
        """

        Returns (list): a list with all the feedback parameters (weights and
            biases) of the network. Note that the first hidden layer does not
            need feedback parameters, so they are not put in the list.

        """
        parameterlist = []
        for layer in self.layers[1:]:
            parameterlist.append(layer.feedbackweights)
            if layer.feedbackbias is not None:
                parameterlist.append(layer.feedbackbias)
        return parameterlist

    def get_BP_updates(self, loss, i):
        """
        Compute the gradients of the loss with respect to the forward
        parameters of layer i.
        Args:
            loss (torch.Tensor): the loss value of the current minibatch.
            i (int): layer index

        Returns (tuple): a tuple with the gradients of the loss with respect to
            the forward parameters of layer i, computed with backprop.

        """
        return self.layers[i].compute_bp_update(loss)

    def compute_bp_angles(self, loss, i, retain_graph=False):
        """
        Compute the angles of the current forward parameter updates of layer i
        with the backprop update for those parameters.
        Args:
            loss (torch.Tensor): the loss value of the current minibatch.
            i (int): layer index
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.

        Returns (tuple): Tuple containing the angle in degrees between the
            updates for the forward weights at index 0 and the forward bias
            at index 1 (if bias is not None).

        """
        bp_gradients = self.layers[i].compute_bp_update(loss,
                                                        retain_graph)
        gradients = self.layers[i].get_forward_gradients()
        if utils.contains_nan(bp_gradients[0].detach()):
            print('bp update contains nan (layer {}):'.format(i))
            print(bp_gradients[0].detach())
        if utils.contains_nan(gradients[0].detach()):
            print('weight update contains nan (layer {}):'.format(i))
            print(gradients[0].detach())
        if torch.norm(gradients[0].detach(), p='fro') < 1e-14:
            print('norm updates approximately zero (layer {}):'.format(i))
            print(torch.norm(gradients[0].detach(), p='fro'))
            print(gradients[0].detach())
        if torch.norm(gradients[0].detach(), p='fro') == 0:
            print('norm updates exactly zero (layer {}):'.format(i))
            print(torch.norm(gradients[0].detach(), p='fro'))
            print(gradients[0].detach())

        weights_angle = utils.compute_angle(bp_gradients[0].detach(),
                                            gradients[0])
        if self.layers[i].bias is not None:
            bias_angle = utils.compute_angle(bp_gradients[1].detach(),
                                             gradients[1])
            return (weights_angle, bias_angle)
        else:
            return (weights_angle, )

    def compute_gn_angles(self, output_activation, loss, damping, i,
                          retain_graph=False):
        """
        Compute the angles of the current forward parameter updates of layer i
        with the GN update for those parameters.
        Args:
            see lib.dtp_layers.DTPLayer.compute_gn_updates(...)
            i (int): layer index
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.

        Returns (tuple): Tuple containing the angle in degrees between the
            updates for the forward weights at index 0 and the forward bias
            at index 1 (if bias is not None).

        """
        gn_gradients = self.layers[i].compute_gn_update(output_activation,
                                                        loss,
                                                        damping,
                                                        retain_graph)
        gradients =self.layers[i].get_forward_gradients()
        weights_angle = utils.compute_angle(gn_gradients[0],
                                            gradients[0])
        if self.layers[i].bias is not None:
            bias_angle = utils.compute_angle(gn_gradients[1],
                                             gradients[1])
            return (weights_angle, bias_angle)
        else:
            return (weights_angle,)

    def compute_gn_activation_angle(self, output_activation, loss,
                                    damping, i, step,
                                    retain_graph=False,
                                    linear=False):
        """
        Compute the angle between the difference between the target and layer
        activation and the gauss-newton update for the layers activation
        Args:
            see lib.dtp_layers.DTPLayer.compute_gn_activation_updates(...)
            i (int): layer index
            step (int): epoch step, just for logging
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
            linear (bool): Flag indicating whether the GN update for the
                linear activations should be computed instead of for the
                nonlinear activations.

        Returns: The average angle in degrees

        """
        if linear:
            target_difference = self.layers[i].linearactivations - \
                                self.layers[i].target
        else:
            target_difference = self.layers[i].activations - \
                                self.layers[i].target
        gn_updates = self.layers[i].compute_gn_activation_updates(
            output_activation,
            loss,
            damping,
            retain_graph=retain_graph,
            linear=linear
        )
        # print(f"Layer {i}:")
        # print(torch.mean(target_difference).item())
        # print(torch.mean(gn_updates).item())
        if self._plots is not None:
            self.td_activation.at[step, i] = torch.mean(target_difference).item()
            self.gn_activation.at[step, i] = torch.mean(gn_updates).item()

        # exit()
        gn_activationav = utils.compute_average_batch_angle(target_difference, gn_updates)
        return gn_activationav

    def compute_bp_activation_angle(self, loss, i, retain_graph=False,
                                    linear=False):
        """
        Compute the angle between the difference between the target and layer
        activation and the backpropagation update for the layers activation
        Args:
            loss (torch.Tensor): the loss value of the current minibatch.
            i (int): layer index
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
            linear (bool): Flag indicating whether the GN update for the
                linear activations should be computed instead of for the
                nonlinear activations.

        Returns : The average angle in degrees
        """
        if linear:
            target_difference = self.layers[i].linearactivations - \
                                self.layers[i].target
        else:
            target_difference = self.layers[i].activations - \
                                self.layers[i].target
        bp_updates = self.layers[i].compute_bp_activation_updates(
            loss=loss,
            retain_graph=retain_graph,
            linear=linear
        ).detach()

        angle = utils.compute_average_batch_angle(target_difference.detach(),
                                                  bp_updates)

        return angle

    def compute_gnt_angle(self, output_activation, loss, damping,
                          i, step, retain_graph=False, linear=False):
        if i == 0:
            h_previous = self.input
        else:
            h_previous = self.layers[i-1].activations

        gnt_updates = self.layers[i].compute_gnt_updates(
            output_activation=output_activation,
            loss=loss,
            h_previous=h_previous,
            damping=damping,
            retain_graph=retain_graph,
            linear=linear
        )

        gradients = self.layers[i].get_forward_gradients()
        weights_angle = utils.compute_angle(gnt_updates[0], gradients[0])
        if self.layers[i].bias is not None:
            bias_angle = utils.compute_angle(gnt_updates[1], gradients[1])
            return (weights_angle, bias_angle)
        else:
            return (weights_angle, )


    def save_logs(self, writer, step):
        """ Save logs and plots for tensorboardX.

        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            """

        for i in range(len(self.layers)):
            name = 'layer {}'.format(i+1)
            self.layers[i].save_logs(writer, step, name,
                                     no_gradient=i==0)

    def save_feedback_batch_logs(self, writer, step, init=False):
        """
        Save the logs for the current minibatch on tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
        """
        for i in range(len(self.layers)):
            name = 'layer {}'.format(i+1)
            self.layers[i].save_feedback_batch_logs(writer, step, name,
                                     no_gradient=i == 0, init=init)

    def save_bp_angles(self, writer, step, loss, retain_graph=False):
        """
        Save the angles of the current forward parameter updates
        with the backprop update for those parameters on tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            loss (torch.Tensor): the loss value of the current minibatch.
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.

        """
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i+1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angles = self.compute_bp_angles(loss, i, retain_graph_flag)
            writer.add_scalar(
                tag='{}/weight_bp_angle'.format(name),
                scalar_value=angles[0],
                global_step=step
            )

            if self._plots is not None:
                self.bp_angles.at[step, i] = angles[0].item()


            if self.layers[i].bias is not None:
                writer.add_scalar(
                    tag='{}/bias_bp_angle'.format(name),
                    scalar_value=angles[1],
                    global_step=step
                )

    def save_gn_angles(self, writer, step, output_activation, loss, damping,
                       retain_graph=False):
        """
        Save the angles of the current forward parameter updates of layer i
        with the backprop update for those parameters. on tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            see lib.dtp_layers.DTPLayer.compute_gn_updates(...)
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.

        """
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i+1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                                        # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angles = self.compute_gn_angles(output_activation, loss, damping,
                                            i, retain_graph_flag)
            writer.add_scalar(
                tag='{}/weight_gn_angle'.format(name),
                scalar_value=angles[0],
                global_step=step
            )

            if self._plots is not None:
                self.gn_angles.at[step, i] = angles[0].item()

            if self.layers[i].bias is not None:
                writer.add_scalar(
                    tag='{}/bias_gn_angle'.format(name),
                    scalar_value=angles[1],
                    global_step=step
                )

    def save_gnt_angles(self, writer, step, output_activation, loss,
                        damping, retain_graph=False, custom_result_df=None):
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        # print('saving gnt angles')
        if self.update_idx is None:
            layer_indices = range(len(self.layers)-1)
        else:
            layer_indices = [self.update_idx]

        # assign a damping constant for each layer for computing the gnt angles
        if isinstance(damping, float):
            damping = [damping for i in range(self.depth)]
        else:
            # print(damping)
            # print(len(damping))
            # print(layer_indices)
            # print(len(layer_indices))
            assert len(damping) == len(layer_indices)

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angles = self.compute_gnt_angle(output_activation=output_activation,
                                            loss=loss,
                                            damping=damping[i],
                                            i=i,
                                            step=step,
                                            retain_graph=retain_graph_flag)
            if custom_result_df is not None:
                custom_result_df.at[step,i] = angles[0].item()
            else:
                writer.add_scalar(
                    tag='{}/weight_gnt_angle'.format(name),
                    scalar_value=angles[0],
                    global_step=step
                )

                if self._plots is not None:
                    # print('saving gnt angles')
                    # print(angles[0].item())
                    self.gnt_angles.at[step, i] = angles[0].item()

                if self.layers[i].bias is not None:
                    writer.add_scalar(
                        tag='{}/bias_gnt_angle'.format(name),
                        scalar_value=angles[1],
                        global_step=step
                    )

    def save_nullspace_norm_ratio(self, writer, step, output_activation,
                                  retain_graph=False):
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                                        # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph

            relative_norm = self.layers[i].compute_nullspace_relative_norm(
                output_activation,
                retain_graph=retain_graph_flag
            )

            writer.add_scalar(
                tag='{}/nullspace_relative_norm'.format(name),
                scalar_value=relative_norm,
                global_step=step
            )

            if self._plots is not None:
                self.nullspace_relative_norm.at[step, i] = relative_norm.item()


    def save_bp_activation_angle(self, writer, step, loss,
                                 retain_graph=False):
        """
        Save the angle between the difference between the target and layer
        activation and the backpropagation update for the layers activation
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            see lib.dtp_layers.DTPLayer.compute_bp_activation_updates(...)
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        """
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angle = self.compute_bp_activation_angle(loss, i,
                                                      retain_graph_flag)


            writer.add_scalar(
                tag='{}/activation_bp_angle'.format(name),
                scalar_value=angle,
                global_step=step
            )
            if self._plots is not None:
                self.bp_activation_angles.at[step, i] = angle.item()
        return

    def save_gn_activation_angle(self, writer, step, output_activation, loss,
                                 damping, retain_graph=False):
        """
        Save the angle between the difference between the target and layer
        activation and the gauss-newton update for the layers activation
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            see lib.dtp_layers.DTPLayer.compute_bp_activation_updates(...)
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        """
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angle = self.compute_gn_activation_angle(output_activation, loss,
                                                     damping, i, step,
                                                     retain_graph_flag)
            writer.add_scalar(
                tag='{}/activation_gn_angle'.format(name),
                scalar_value=angle,
                global_step=step
            )

            if self._plots is not None:
                self.gn_activation_angles.at[step, i] = angle.item()
        return


    def get_av_reconstruction_loss(self):
        """ Get the average reconstruction loss of the network across its layers
        for the current mini-batch.
        Args:
            network (networks.DTPNetwork): network
        Returns (torch.Tensor):
            Tensor containing a scalar of the average reconstruction loss
        """
        reconstruction_losses = np.array([])

        for layer in self.layers[1:]:
            reconstruction_losses = np.append(reconstruction_losses,
                                              layer.reconstruction_loss)

        return np.mean(reconstruction_losses)

class DTPDRLNetwork(DTPNetwork):
    """
    A class for networks that contain DTPDRLLayers.
    #FIXME: now the target for the nonlinear output is computed and the path
        is trained for propagating the nonlinear output target to the hidden
        layers. I think training will go better if you compute a target for the
        linear output activation and train the feedback path to map linear
        output targets towards the hidden layers.
    """

    def set_layers(self, n_in, n_hidden, n_out, activation, output_activation,
                   bias, forward_requires_grad, initialization,
                   fb_activation):
        """
        Create the layers of the network and output them as a ModuleList.
        Args:
            n_in: input dimension (flattened input assumed)
            n_hidden: list with hidden layer dimensions
            n_out: output dimension
            activation: activation function indicator for the hidden layers
            output_activation: activation function indicator for the output
                layer
            bias: boolean indicating whether the network uses biases or not
            forward_requires_grad (bool): Flag indicating whether the forward
            parameters require gradients that can be computed with autograd.
            This might be needed when comparing the DTP updates with BP updates
            and GN updates.
            initialization (str): the initialization method used for the forward
                and feedback matrices of the layers
            fb_activation (str): activation function indicator for the feedback
                path of the hidden layers

        """
        n_all = [n_in] + n_hidden + [n_out]
        layers = nn.ModuleList()
        for i in range(1, len(n_all) - 1):
            layers.append(
                DTPDRLLayer(n_all[i - 1], n_all[i], bias=bias,
                            forward_activation=activation,
                            feedback_activation=fb_activation,
                            forward_requires_grad=forward_requires_grad,
                            initialization=initialization
                            ))
        layers.append(DTPDRLLayer(n_all[-2], n_all[-1], bias=bias,
                                  forward_activation=output_activation,
                                  feedback_activation=fb_activation,
                                  forward_requires_grad=forward_requires_grad,
                                  initialization=initialization))
        return layers

    def compute_feedback_gradients(self, i):
        """
        Compute the difference reconstruction loss for layer i of the network
        and compute the gradient of this loss with respect to the feedback
        parameters. The gradients are saved in the .grad attribute of the
        feedback parameter tensors.

        """
        # save the index of the layer for which the reconstruction loss is
        # computed.
        self.reconstruction_loss_index = i

        h_corrupted = self.layers[i-1].activations + \
            self.sigma * torch.randn_like(self.layers[i - 1].activations)
        output_corrupted = self.dummy_forward(h_corrupted, i-1)
        h_current_reconstructed = self.propagate_backward(output_corrupted, i)
        self.layers[i].compute_feedback_gradients(h_corrupted,
                                                  h_current_reconstructed,
                                                  self.layers[i-1].activations,
                                                  self.sigma)

    def dummy_forward(self, h, i):
        """
        Propagates the activations h of layer i forward to the output of the
        network, without saving activations and linear activations in the layer
        objects.
        Args:
            h (torch.Tensor): activations
            i (int): index of the layer of which h are the activations

        Returns: output of the network with h as activation for layer i

        """
        y = h

        for layer in self.layers[i+1:]:
            y = layer.dummy_forward(y)

        return y

    def get_av_reconstruction_loss(self):
        """ Get the reconstruction loss of the network for the layer of which
        the feedback parameters were trained on the current mini-batch
        Returns (torch.Tensor):
            Tensor containing a scalar of the average reconstruction loss
        """
        reconstruction_loss = self.layers[self.reconstruction_loss_index].\
            reconstruction_loss
        return reconstruction_loss

class DDTPMLPLayer(DTPDRLLayer):
    """ Direct DTP layer with a fully trained multilayer perceptron (MLP) as direct feedback connections.
    DDTP-linear is a special case of DDTPMLP with a single-layer linear MLP as direct
     feedback connection."""

    def __init__(self, in_features, out_features, size_output, bias=True,
                 forward_requires_grad=False, forward_activation='tanh',
                 feedback_activation='tanh', size_hidden_fb=[100],
                 fb_hidden_activation=None,
                 initialization='orthogonal',
                 is_output=False,
                 recurrent_input=False):
        """

        Args:
            in_features:
            out_features:
            size_output:
            bias:
            forward_requires_grad:
            forward_activation:
            feedback_activation:
            size_hidden_fb:
            fb_hidden_activation:
            initialization:
            is_output:
            recurrent_input (bool): flag indicating whether the nonlinear layer
                activation should be used as a (recurrent) input to the feedback
                MLP that propagates the target to this layer.
        """
        # Warning: if the __init__ method of DTPLayer gets new/extra arguments,
        # this should also be incorporated here
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         forward_requires_grad=forward_requires_grad,
                         forward_activation=forward_activation,
                         feedback_activation=feedback_activation,
                         initialization=initialization)

        # Now we need to overwrite the initialization of the feedback weights,
        # as we now need an MLP as feedback connection from the output towards the
        # current layer. We will overwrite the feedback parameters with None and
        # create an MLP feedback connection

        self._feedbackweights = None
        self._feedbackbias = None
        self._recurrent_input = recurrent_input
        self._is_output = is_output
        self._has_hidden_fb_layers = size_hidden_fb is not None

        if fb_hidden_activation is None:
            fb_hidden_activation = feedback_activation

        if not is_output:
            if recurrent_input:
                n_in = size_output + out_features
            else:
                n_in = size_output
            self._fb_mlp = BPNetwork(n_in=n_in,
                                     n_hidden=size_hidden_fb,
                                     n_out=out_features,
                                     activation=fb_hidden_activation,
                                     output_activation=feedback_activation,
                                     bias=bias,
                                     initialization=initialization)
            self._fb_mlp.set_requires_grad(False)
        else:
            self._fb_mlp = None # output does not need to have a feedback path

    @property
    def feedbackweights(self):
        if not self._has_hidden_fb_layers:
            return self._fb_mlp.layers[0].weight
        else:
            return None

    @property
    def feedbackbias(self):
        if not self._has_hidden_fb_layers:
            return self._fb_mlp.layers[0].bias
        else:
            return None


    def set_feedback_requires_grad(self, value):
        """
        Sets the 'requires_grad' attribute of the all the parameters in the
        feedback MLP to the given value
        Args:
            value (bool): True or False
        """
        if not isinstance(value, bool):
            raise TypeError('The given value should be a boolean.')

        self._fb_mlp.set_requires_grad(value)

    def propagate_backward(self, output_target):
        if self._recurrent_input:
            in_tensor = torch.cat((output_target, self.activations), dim=1)
        else:
            in_tensor = output_target
        h = self._fb_mlp(in_tensor.to(self.device))
        return h

    def backward(self, output_target, h_current, output_activation):
        """
        Compute the target linear activation for the current layer in a DTP-like
        fashion, based on the linear output target and the output linear
        activation
        Args:
            output_target: output target
            h_current: activation of the current layer
            output_activation: Output activation

        Returns: target for linear activation of this layer

        """
        h_target = self.propagate_backward(output_target)
        h_tilde = self.propagate_backward(output_activation)
        h_target_current = h_target + h_current - h_tilde

        return h_target_current

    def compute_feedback_gradients(self, h_current_corrupted,
                                   output_corrupted,
                                   output_noncorrupted,
                                   sigma):

        """
        Compute the gradients of the feedback weights and bias, based on the
        difference reconstruction loss.
        The gradients will be saved in the .grad attributes of the feedback
        parameters.

        """

        self.set_feedback_requires_grad(True)

        h_current_noncorrupted = self.activations

        h_current_reconstructed = self.backward(output_corrupted,
                                                h_current_noncorrupted,
                                                output_noncorrupted)

        if sigma <= 0:
            raise ValueError('Sigma should be greater than zero when using the'
                             'difference reconstruction loss. Given sigma = '
                             '{}'.format(sigma))
        scale = 1/sigma**2
        reconstruction_loss = scale * F.mse_loss(h_current_reconstructed,
                                         h_current_corrupted)

        self.save_feedback_gradients(reconstruction_loss)

        self.set_feedback_requires_grad(False)

    def save_feedback_gradients(self, reconstruction_loss):
        """
        Compute the gradients of the reconstruction_loss with respect to the
        feedback parameters by help of autograd and save them in the .grad
        attribute of the feedback parameters
        Args:
            reconstruction_loss: the reconstruction loss

        """
        self.reconstruction_loss = reconstruction_loss.item()
        grads = torch.autograd.grad(reconstruction_loss,
                                    self._fb_mlp.parameters(),
                                    retain_graph=False)
        for i, param in enumerate(self._fb_mlp.parameters()):
            param.grad = grads[i].detach()

    def get_feedback_parameters(self):
        return self._fb_mlp.parameters()

    def save_logs(self, writer, step, name, no_gradient=False,
                  no_fb_param=False):
        if self._has_hidden_fb_layers or self._is_output:
            DTPLayer.save_logs(self=self, writer=writer, step=step,
                           name=name, no_gradient=no_gradient,
                           no_fb_param=True)
        else:
            DTPLayer.save_logs(self=self, writer=writer, step=step,
                               name=name, no_gradient=no_gradient,
                               no_fb_param=False)

class DDTPRHLNetwork(DTPDRLNetwork):
    """
    A class for networks that use direct feedback for providing targets
    to the hidden layers by using a shared hidden feedback representation.
    It trains its feedback parameters based on the difference reconstruction
    loss.
    """
    def __init__(self, n_in, n_hidden, n_out, activation='relu',
                 output_activation='linear', bias=True, sigma=0.36,
                 forward_requires_grad=False, hidden_feedback_dimension=500,
                 hidden_feedback_activation='tanh', initialization='orthogonal',
                 fb_activation='linear', plots=None, recurrent_input=False):
        """

        Args:
            n_in:
            n_hidden:
            n_out:
            activation:
            output_activation:
            bias:
            sigma:
            forward_requires_grad:
            hidden_feedback_dimension: The dimension of the hidden feedback
                layer
            hidden_feedback_activation: The activation function of the hidden
                feedback layer
            initialization (str): the initialization method used for the forward
                and feedback matrices of the layers
        """
        # we need to overwrite the __init__function, as we need an extra
        # argument for this network class: hidden_feedback_dimension
        nn.Module.__init__(self)

        self._depth = len(n_hidden) + 1
        self._layers = self.set_layers(n_in, n_hidden, n_out,
                                       activation, output_activation,
                                       bias, forward_requires_grad,
                                       hidden_feedback_dimension,
                                       initialization,
                                       fb_activation,
                                       recurrent_input)
        self._input = None
        self._sigma = sigma
        self._forward_requires_grad = forward_requires_grad
        self._plots = plots
        self.update_idx = None
        if plots is not None:
            self.bp_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gn_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gnt_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.bp_activation_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gn_activation_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.reconstruction_loss_init = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.reconstruction_loss = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.td_activation = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gn_activation = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.bp_activation = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.nullspace_relative_norm = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
        #TODO: set the requires_grad attribute of the weight and bias of the
        # linear layer to False.
        self._hidden_feedback_layer = nn.Linear(n_out,
                                                hidden_feedback_dimension,
                                                bias=bias)
        self._hidden_feedback_activation_function = \
            self.set_hidden_feedback_activation(
            hidden_feedback_activation)

    def set_layers(self, n_in, n_hidden, n_out, activation, output_activation,
                   bias, forward_requires_grad, hidden_feedback_dimension,
                   initialization, fb_activation, recurrent_input):
        """
        See documentation of DTPNetwork

        """
        n_all = [n_in] + n_hidden + [n_out]
        layers = nn.ModuleList()
        for i in range(1, len(n_all)-1):
            if i == len(n_all) - 2:
                hidden_fb_dimension_copy = n_out
                recurrent_input_copy = False
            else:
                hidden_fb_dimension_copy = hidden_feedback_dimension
                recurrent_input_copy = recurrent_input
            layers.append(
                DDTPRHLLayer(n_all[i - 1], n_all[i],
                             bias=bias,
                             forward_requires_grad=forward_requires_grad,
                             forward_activation=activation,
                             feedback_activation=fb_activation,
                             hidden_feedback_dimension=hidden_fb_dimension_copy,
                             initialization=initialization,
                             recurrent_input=recurrent_input_copy)
            )
        layers.append(
            DDTPRHLLayer(n_all[-2], n_all[-1],
                         bias=bias,
                         forward_requires_grad=forward_requires_grad,
                         forward_activation=output_activation,
                         feedback_activation=output_activation,
                         hidden_feedback_dimension=hidden_feedback_dimension,
                         initialization=initialization,
                         recurrent_input=recurrent_input)
        )
        return layers

    def get_feedback_parameter_list(self):
        """

        Returns (list): a list with all the feedback parameters (weights and
            biases) of the network. Note that the LAST hidden layer does not
            need feedback parameters, so they are not put in the list.

        """
        parameterlist = []
        for layer in self.layers[:-1]:
            parameterlist.append(layer.feedbackweights)
            if layer.feedbackbias is not None:
                parameterlist.append(layer.feedbackbias)
        return parameterlist

    def set_hidden_feedback_activation(self, hidden_feedback_activation):
        """ Create an activation function corresponding to the
        given string.
        Args:
            hidden_feedback_activation (str): string indicating which
            activation function needs to be created

        Returns (nn.Module): activation function object
        """
        if hidden_feedback_activation == 'linear':
            return nn.Softshrink(lambd=0)
        elif hidden_feedback_activation == 'relu':
            return nn.ReLU()
        elif hidden_feedback_activation == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.2)
        elif hidden_feedback_activation == 'tanh':
            return nn.Tanh()
        elif hidden_feedback_activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError("The given hidden feedback activation {} "
                             "is not supported.".format(
                hidden_feedback_activation))

    @property
    def hidden_feedback_layer(self):
        """ getter for attribute hidden_feedback_layer."""
        return self._hidden_feedback_layer

    @property
    def hidden_feedback_activation_function(self):
        """ getter for attribute hidden_feedback_activation"""
        return self._hidden_feedback_activation_function

    def compute_output_target(self, loss, target_lr):
        """
        Compute the output target and feed it through the hidden feedback
        layer.
        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer

        Returns: Mini-batch the hidden_feedback_layer activations resulting from
            the current output target
        """
        # We use the nonlinear activation of the output layer to compute the
        # target, such that the theory with GN optimization is consistent.
        # Note that the current direct TP network implementation expects a
        # Linear target. Therefore, after computing the nonlinear target, we
        # pass it through the exact inverse nonlinearity of the output layer.
        # The exact inverse is only implemented with sigmoid layers and the
        # linear and softmax output layer (as they both use a linear output
        # layer under the hood). For other layers, the exact inverse is not
        # implemented. Hence, when they are used, we'll throw an error to
        # prevent misusage.

        output_activations = self.layers[-1].activations
        gradient = torch.autograd.grad(
            loss, output_activations,
            retain_graph=self.forward_requires_grad)[0].detach()

        output_targets = output_activations - \
                         target_lr*gradient

        if self.layers[-1].forward_activation in ['sigmoid', 'linear']:
            output_targets = self.layers[-1].feedback_activationfunction(
                output_targets
            )
        else:
            warnings.warn('Forward activation {} not implemented yet.'
                          .format(self.layers[-1].forward_activation))


        hidden_feedback_activations = self.hidden_feedback_layer(output_targets)
        hidden_feedback_activations = self.hidden_feedback_activation_function(
            hidden_feedback_activations
        )

        # print("hfa:", hidden_feedback_activations.shape)

        return hidden_feedback_activations

    def propagate_backward(self, h_target, i):
        """
        Propagate the hidden fb layer target backwards to layer i in a
        DTP-like fashion to provide a nonlinear target for layer i
        Args:
            h_target (torch.Tensor): the hidden feedback activation, resulting
                from the output target
            i: the layer index to which the target must be propagated

        Returns (torch.Tensor): the nonlinear target for layer i

        """

        if i == self.depth - 2:
            # For propagating the output target to the last hidden layer, we
            # want to have a simple linear mapping (as the real GN target is also
            # a simple linear mapping) instead of using the random hidden layer.
            h_i_target = self.layers[i].backward(h_target,
                                                 self.layers[i].activations,
                                                 self.layers[-1].linearactivations)
        else:
            a_last = self.layers[-1].linearactivations
            a_feedback_hidden = self.hidden_feedback_layer(a_last)
            h_feedback_hidden = self.hidden_feedback_activation_function(
                a_feedback_hidden
            )
            h_i_target = self.layers[i].backward(h_target,
                                                 self.layers[i].activations,
                                                 h_feedback_hidden)
        return h_i_target

    # def propagate_backward_last_hidden_layer(self, output_target, i):
    #     """ For propagating the output target to the last hidden layer, we
    #     want to have a simple linear mapping (as the real GN target is also
    #     a simple linear mapping) instead of using the random hidden layer."""

    def compute_feedback_gradients(self, i):
        """
        Compute the difference reconstruction loss for layer i and update the
        feedback parameters based on the gradients of this loss.
        """
        self.reconstruction_loss_index = i

        h_corrupted = self.layers[i].activations + \
                      self.sigma * torch.randn_like(
            self.layers[i].activations)

        output_corrupted = self.dummy_forward_linear_output(h_corrupted, i)
        output_noncorrupted = self.layers[-1].linearactivations

        if i == self.depth - 2:
            # For propagating the output target to the last hidden layer, we
            # want to have a simple linear mapping (as the real GN target is also
            # a simple linear mapping) instead of using the random hidden layer.
            h_fb_hidden_corrupted = output_corrupted
            h_fb_hidden_noncorrupted = output_noncorrupted
        else:

            a_fb_hidden_corrupted = self.hidden_feedback_layer(
                output_corrupted)  # Voltage!
            h_fb_hidden_corrupted = self.hidden_feedback_activation_function(
                a_fb_hidden_corrupted)


            a_fb_hidden_noncorrupted = self.hidden_feedback_layer(
                output_noncorrupted)  # Voltage!
            h_fb_hidden_noncorrupted = self.hidden_feedback_activation_function(
                a_fb_hidden_noncorrupted)

        self.layers[i].compute_feedback_gradients(h_corrupted,
                                                  h_fb_hidden_corrupted,
                                                  h_fb_hidden_noncorrupted,
                                                  self.sigma)

    def compute_dummy_output_target(self, loss, target_lr, retain_graph=False):
        """ Compute a target for the nonlinear activation of the output layer.
        """
        output_activations = self.layers[-1].activations
        gradient = torch.autograd.grad(
            loss, output_activations,
            retain_graph=(self.forward_requires_grad or retain_graph))[
            0].detach()

        output_targets = output_activations - \
                         target_lr * gradient
        return output_targets

    def backward_random(self, loss, target_lr, i, save_target=False,
                        norm_ratio=1.):
        """ Propagate the output target backwards through the network until
        layer i. Based on this target, compute the gradient of the forward
        weights and bias of layer i and save them in the parameter tensors.
        Args:
            last:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer
            i: layer index to which the target needs to be propagated and the
                gradients need to be computed
            save_target (bool): flag indicating whether the target should be
                saved in the layer object for later use.
            norm_ratio (float): used for the minimal norm update (see theory)
        """

        self.update_idx = i

        if i == self.depth - 1:
            h_target = self.compute_dummy_output_target(loss, target_lr)
            if save_target:
                self.layers[i].target = h_target

            self.layers[i].compute_forward_gradients(h_target,
                                                     self.layers[
                                                         i - 1].activations,
                                                     norm_ratio=norm_ratio)
        elif i == self.depth - 2:
            # For propagating the output target to the last hidden layer, we
            # want to have a simple linear mapping (as the real GN target is also
            # a simple linear mapping) instead of using the random hidden layer.
            h_target = self.compute_dummy_output_target(loss, target_lr)
            h_target = self.layers[-1].feedback_activationfunction(
                h_target)
            h_target = self.propagate_backward(h_target, i)
            if save_target:
                self.layers[i].target = h_target

            if i == 0: # first hidden layer needs to have the input
                       # for computing gradients
                self.layers[i].compute_forward_gradients(h_target, self.input,
                                                         norm_ratio=norm_ratio)
            else:
                self.layers[i].compute_forward_gradients(h_target,
                                                     self.layers[i-1].activations,
                                                         norm_ratio=norm_ratio)

        else:
            h_target = self.compute_output_target(loss, target_lr)

            h_target = self.propagate_backward(h_target, i)

            if save_target:
                self.layers[i].target = h_target

            if i == 0: # first hidden layer needs to have the input
                       # for computing gradients
                self.layers[i].compute_forward_gradients(h_target, self.input,
                                                         norm_ratio=norm_ratio)
            else:
                self.layers[i].compute_forward_gradients(h_target,
                                                     self.layers[i-1].activations,
                                                         norm_ratio=norm_ratio)

    def backward(self, loss, target_lr, save_target=False, norm_ratio=1.):
        """ Compute the targets for all layers and update their forward
         parameters accordingly. """
        # First compute the output target, as that is computed in a different
        # manner from the output target for propagating to the hidden layers.
        output_target = self.compute_dummy_output_target(loss, target_lr,
                                                         retain_graph=True)
        if save_target:
            self.layers[-1].target = output_target

        self.layers[-1].compute_forward_gradients(output_target,
                                                 self.layers[-2].activations,
                                                 norm_ratio=norm_ratio)

        if self.depth > 1:
            # For propagating the output target to the last hidden layer, we
            # want to have a simple linear mapping (as the real GN target is also
            # a simple linear mapping) instead of using the random hidden layer.
            output_target_linear = self.layers[-1].feedback_activationfunction(
                output_target)
            h_target = self.propagate_backward(output_target_linear, self.depth - 2)
            if save_target:
                self.layers[self.depth - 2].target = h_target

            if self.depth - 2 == 0:  # first hidden layer needs to have the input
                # for computing gradients
                self.layers[self.depth - 2].compute_forward_gradients(h_target,
                                                                      self.input,
                                                         norm_ratio=norm_ratio)
            else:
                self.layers[self.depth - 2].compute_forward_gradients(h_target,
                                                         self.layers[
                                                             self.depth - 3].activations,
                                                         norm_ratio=norm_ratio)


            # Then compute the hidden feedback layer activation for the output
            # target
            hidden_fb_target = self.compute_output_target(loss, target_lr)
            self.backward_all(hidden_fb_target, save_target=save_target,
                              norm_ratio=norm_ratio)

    def backward_all(self, output_target, save_target=False, norm_ratio=1.):
        """
        Compute the targets for all hidden layers (not output layer) and
        update their forward parameters accordingly.
        """
        for i in range(self.depth - 2):
            h_target = self.propagate_backward(output_target, i)

            if save_target:
                self.layers[i].target = h_target

            if i == 0:  # first hidden layer needs to have the input
                # for computing gradients
                self.layers[i].compute_forward_gradients(h_target, self.input,
                                                         norm_ratio=norm_ratio)
            else:
                self.layers[i].compute_forward_gradients(h_target,
                                                         self.layers[
                                                             i - 1].activations,
                                                         norm_ratio=norm_ratio)


    def compute_gn_activation_angle(self, output_activation, loss,
                                    damping, i, step,
                                    retain_graph=False,
                                    linear=False):
        return DTPNetwork.compute_gn_activation_angle(
            self=self,
            output_activation=output_activation,
            loss=loss,
            damping=damping,
            i=i,
            step=step,
            retain_graph=retain_graph,
            linear=False)

    def compute_bp_activation_angle(self, loss, i, retain_graph=False,
                                    linear=False):
        return DTPNetwork.compute_bp_activation_angle(self=self,
                                                      loss=loss, i=i,
                                                   retain_graph=retain_graph,
                                                   linear=False)

    def compute_gnt_angle(self, output_activation, loss, damping,
                          i, step, retain_graph=False, linear=False):
        return DTPNetwork.compute_gnt_angle(self=self,
                                            output_activation=output_activation,
                                            loss=loss,
                                            damping=damping,
                                            i=i,
                                            step=step,
                                            retain_graph=retain_graph,
                                            linear=False)


    def dummy_forward_linear_output(self, h, i):
        """ Propagates the nonlinear activation h of layer i forward through
        the network until the linear output activation.
        THE OUTPUT NONLINEARITY IS NOT APPLIED"""
        y = h

        for layer in self.layers[i + 1:-1]:
            y = layer.dummy_forward(y)

        y = self.layers[-1].dummy_forward_linear(y)
        return y

class DDTPMLPNetwork(DDTPRHLNetwork):
    """ Network class for DDTPMLPLayers"""

    def __init__(self, n_in, n_hidden, n_out, activation='relu',
                 output_activation='linear', bias=True, sigma=0.36,
                 forward_requires_grad=False, size_hidden_fb=[100],
                 fb_hidden_activation='tanh', initialization='orthogonal',
                 fb_activation='linear', plots=None, recurrent_input=False, device =device
                 ):
        nn.Module.__init__(self)

        self._depth = len(n_hidden) + 1
        self._layers = self.set_layers(n_in, n_hidden, n_out,
                                       activation, output_activation,
                   bias, forward_requires_grad, size_hidden_fb,
                   initialization, fb_activation, fb_hidden_activation,
                                       recurrent_input).to(device)
        self._input = None
        self._sigma = sigma
        self._forward_requires_grad = forward_requires_grad
        self._plots = plots
        self.update_idx = None
        self.device = device
        if plots is not None:
            self.bp_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gn_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gnt_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.bp_activation_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gn_activation_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.reconstruction_loss_init = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.reconstruction_loss = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.td_activation = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gn_activation = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.bp_activation = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.nullspace_relative_norm = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
        # TODO: set the requires_grad attribute of the weight and bias of the
        # linear layer to False.

    def set_layers(self, n_in, n_hidden, n_out, activation, output_activation,
                   bias, forward_requires_grad, size_hidden_fb,
                   initialization, fb_activation, fb_hidden_activation,
                   recurrent_input, device='cuda'):
        n_all = [n_in] + n_hidden + [n_out]
        layers = nn.ModuleList()
        for i in range(1, len(n_all) - 1):
            if i == len(n_all) - 2:
                hidden_fb_layers = None
                recurrent_input_copy = False
                bias_copy = False
            else:
                hidden_fb_layers = size_hidden_fb
                recurrent_input_copy = recurrent_input
                bias_copy = bias
            layers.append(
                DDTPMLPLayer(n_all[i - 1], n_all[i], n_out,
                             bias=bias_copy,
                             forward_requires_grad=forward_requires_grad,
                             forward_activation=activation,
                             feedback_activation=fb_activation,
                             size_hidden_fb=hidden_fb_layers,
                             fb_hidden_activation=fb_hidden_activation,
                             initialization=initialization,
                             recurrent_input=recurrent_input_copy
                             ).to(device)
            )
        layers.append(
            DDTPMLPLayer(n_all[-2], n_all[-1], n_out,
                         bias=bias,
                         forward_requires_grad=forward_requires_grad,
                         forward_activation=output_activation,
                         feedback_activation=output_activation,
                         size_hidden_fb=size_hidden_fb,
                         fb_hidden_activation=fb_hidden_activation,
                         initialization=initialization,
                         is_output=True
                         ).to(device)
        )
        return layers

    def get_feedback_parameter_list(self):
        parameterlist = []
        for layer in self.layers[:-1]:
            parameterlist += [p for p in layer.get_feedback_parameters()]

        return parameterlist

    def compute_output_target(self, loss, target_lr, retain_graph=False):
        """
        Compute the output target for the linear activation of the output
        layer.
        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the linear activation
                of the output layer
            retain_graph: Flag indicating whether the autograd graph should
                be retained.

        Returns: Mini-batch of output targets
        """
        output_activations = self.layers[-1].activations
        gradient = torch.autograd.grad(
            loss, output_activations,
            retain_graph=(self.forward_requires_grad or retain_graph))[
            0].detach()

        output_targets = output_activations - \
                         target_lr * gradient

        if self.layers[-1].forward_activation == 'linear':
            output_targets = output_targets

        elif self.layers[-1].forward_activation == 'sigmoid':
            output_targets = utils.logit(
                output_targets)  # apply inverse sigmoid

        else:
            warnings.warn('Forward activation {} not implemented yet.'.format(
                self.layers[-1].forward_activation))

        return output_targets

    def propagate_backward(self, output_target, i):
        """
        Propagate the linear output target backwards to layer i with the
        direct feedback MLP mapping to provide a target for the nonlinear hidden
        layer activation.
        """

        a_output = self.layers[-1].linearactivations

        h_layer_i = self.layers[i].activations

        h_target_i = self.layers[i].backward(output_target, h_layer_i,
                                             a_output)

        return h_target_i

    def backward_random(self, loss, target_lr, i, save_target=False,
                        norm_ratio=1.):
        """ Compute and propagate the output target to layer i via
        direct feedback MLP connection."""

        self.update_idx = i

        if i != self._depth - 1:
            output_target = self.compute_output_target(loss, target_lr)

            h_target_i = self.propagate_backward(output_target, i)

            if save_target:
                self.layers[i].target = h_target_i

            if i == 0:
                self.layers[i].compute_forward_gradients(h_target_i, self.input,
                                                         norm_ratio=norm_ratio)
            else:
                self.layers[i].compute_forward_gradients(h_target_i,
                                                self.layers[i-1].activations,
                                                         norm_ratio=norm_ratio)

        else:
            output_target = self.compute_dummy_output_target(loss, target_lr)
            h_target_i = output_target
            if save_target:
                self.layers[i].target = h_target_i

            self.layers[i].compute_forward_gradients(h_target_i,
                                                self.layers[i-1].activations,
                                                     norm_ratio=norm_ratio)

    def compute_feedback_gradients(self, i):
        """
        Compute the difference reconstruction loss for layer i of the network
        and compute the gradient of this loss with respect to the feedback
        parameters. The gradients are saved in the .grad attribute of the
        feedback parameter tensors.

        Implementation:
        - get the activation of layer i
        - corrupt the activation of layer i
        - propagate the corrupted activation and noncorrupted activation
          towards the output of the last layer with dummy_forward_linear
        - provide the needed arguments to self.layers[i].compute_feedback_
            gradients
        Args:
            i: the layer index of which the feedback matrices should be updated
        """



        self.reconstruction_loss_index = i

        h_corrupted = self.layers[i].activations + \
                      self.sigma * torch.randn_like(
            self.layers[i].activations)

        output_corrupted = self.dummy_forward_linear_output(h_corrupted, i)
        output_noncorrupted = self.layers[-1].linearactivations


        self.layers[i].compute_feedback_gradients(h_corrupted,
                                                  output_corrupted,
                                                  output_noncorrupted,
                                                  self.sigma)

    def compute_dummy_output_target(self, loss, target_lr, retain_graph=False):
        """ Compute a target for the nonlinear activation of the output layer.
        """
        output_activations = self.layers[-1].activations
        gradient = torch.autograd.grad(
            loss, output_activations,
            retain_graph=(self.forward_requires_grad or retain_graph))[
            0].detach()

        output_targets = output_activations - \
                         target_lr * gradient
        return output_targets

    def dummy_forward_linear_output(self, h, i):
        return DDTPRHLNetwork.dummy_forward_linear_output(self=self,
                                                          h=h,
                                                          i=i)

    def compute_gn_activation_angle(self, output_activation, loss,
                                    damping, i, step,
                                    retain_graph=False,
                                    linear=False):
        return DDTPRHLNetwork.compute_gn_activation_angle(
            self=self,
            output_activation=output_activation,
            loss=loss,
            damping=damping,
            i=i,
            step=step,
            retain_graph=retain_graph,
            linear=linear)

    def compute_bp_activation_angle(self, loss, i, retain_graph=False,
                                    linear=False):
        return DDTPRHLNetwork.compute_bp_activation_angle(self=self,
                                                          loss=loss, i=i,
                                                          retain_graph=retain_graph,
                                                          linear=linear)

    def compute_gnt_angle(self, output_activation, loss, damping,
                          i, step, retain_graph=False, linear=False):
        return DDTPRHLNetwork.compute_gnt_angle(self=self,
                                                output_activation=output_activation,
                                                loss=loss,
                                                damping=damping,
                                                i=i,
                                                step=step,
                                                retain_graph=retain_graph,
                                                linear=linear)

