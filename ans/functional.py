import functools
import math
from math import floor
from typing import Any, Optional, Union

import torch

from ans.autograd import Variable


class Function:

    @classmethod
    def apply(cls, *inputs: Union[Variable, Any], **params: Any) -> Variable:
        tensor_args = [i.data if isinstance(i, Variable) else i for i in inputs]
        output_data, cache = cls.forward(*tensor_args, **params)

        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, ...]:
            dinputs = cls.backward(dout, cache=cache)
            return tuple(dinputs[i] for i, inp in enumerate(inputs) if isinstance(inp, Variable))

        grad_fn.name = f"{cls.__name__}.backward"
        return Variable(
            output_data,
            parents=tuple(i for i in inputs if isinstance(i, Variable)),
            grad_fn=grad_fn
        )

    @staticmethod
    def forward(*inputs: torch.Tensor, **params: Any) -> tuple[torch.Tensor, tuple]:
        raise NotImplementedError

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return str(self)


class Linear(Function):

    @staticmethod
    def forward(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, num_features)
            weight: shape (num_features, num_out)
            bias: shape (num_out,)
        Returns:
            output: shape (num_samples, num_out)
            cache: tuple of intermediate results to use in backward
        """

        ########################################

        output = input @ weight + bias
        cache = (input, weight)

        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_out)
            cache: cache from the forward pass
        Returns:
            tuple of gradient w.r.t. input, weight, bias in this order
        """

        ########################################

        dinput = doutput @ cache[1].T
        dweight = cache[0].T @ doutput
        dbias = torch.sum(doutput, dim=0)

        ########################################

        return dinput, dweight, dbias


class Sigmoid(Function):

    @staticmethod
    def forward(input: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, num_features)
        Returns:
            output: shape (num_samples, num_features)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        output = 1 / (1 + torch.exp(-input))
        cache = (output,)
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_out)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        dinput = doutput * cache[0] * (1 - cache[0])
        ########################################

        return dinput,


class SoftmaxCrossEntropy(Function):

    @staticmethod
    def forward(scores: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            scores: shape (num_samples, num_out)
            targets: shape (num_samples,); dtype torch.int64
        Returns:
            output: shape () (scalar tensor)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        scores = (scores.T - torch.max(scores, dim=1)[0]).T  # norm : sub row max from scores (Reshape for broadcast)
        ln = - scores[torch.arange(scores.shape[0]), targets] + torch.log(torch.sum(torch.exp(scores), dim=1))
        output = torch.mean(ln)
        cache = (scores, targets)
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape ()
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. scores (single-element tuple)
        """

        ########################################
        dscores = torch.exp(cache[0]) / torch.sum(torch.exp(cache[0]), dim=1)[:, None]
        dscores[torch.arange(cache[0].shape[0]), cache[1]] -= 1
        dscores /= cache[0].shape[0]
        ########################################

        return dscores * doutput,


class ReLU(Function):

    @staticmethod
    def forward(input: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, ...)
        Returns:
            output: shape (num_samples, ...)
            cache: tuple of intermediate results to use in backward

        Operation is not inplace, i.e. it does not modify the input.
        """

        ########################################
        output = torch.where(input > 0, input, 0)
        cache = input,
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, ...)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################

        dinput = torch.where(cache[0] > 0, doutput, 0)

        ########################################

        return dinput,


class Tanh(Function):

    @staticmethod
    def forward(input: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, ...)
        Returns:
            output: shape (num_samples, ...)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, ...)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################

        raise NotImplementedError

        ########################################

        return dinput,


class Dropout(Function):

    @staticmethod
    def forward(
            input: torch.Tensor,
            p: float = 0.5,
            training: bool = False,
            seed: Optional[int] = None
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, ...)
            p: probability of element being zeroed out
            training: whether in training mode or eval mode
            seed: enable deterministic behavior (useful for gradient check)
        Returns:
            output: shape (num_samples, ...)
            cache: tuple of intermediate results to use in backward
        """
        # deterministic behavior for gradient check
        if seed is not None:
            torch.manual_seed(seed)

        m = torch.rand(input.shape[0], input.shape[1]) < p
        ########################################
        if training:
            output = torch.where(m, 0, input / (1 - p))
        else:
            output = input.clone()
        cache = (m, p, training)
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, ...)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        m, p, training = cache
        if training:
            dinput = torch.where(m > p, 0, doutput / (1 - p))
        else:
            dinput = doutput.clone()
        ########################################
        return dinput,


class BatchNorm1d(Function):

    @staticmethod
    def forward(
            input: torch.Tensor,
            gamma: Optional[torch.Tensor],
            beta: Optional[torch.Tensor],
            running_mean: Optional[torch.Tensor] = None,
            running_var: Optional[torch.Tensor] = None,
            momentum: float = 0.9,
            eps: float = 1e-05,
            training: bool = False
    ) -> tuple[torch.Tensor, tuple]:
        """

        Args:
            input: shape (num_samples, num_features)
            gamma: shape (num_features,)
            beta: shape (num_features,)
            running_mean: shape (num_features,)
            running_var: shape (num_features,)
            momentum: running average smoothing coefficient
            eps: for numerical stabilization
            training: whether in training mode or eval mode
        Returns:
            output: shape (num_samples, num_features)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        if training:
            # find the params of the batch size
            mi = torch.mean(input, dim=0)
            sigma2 = torch.var(input, dim=0, unbiased=False)

            ## acum running params inplace operation
            n = input.shape[0]
            ## if none ...
            if running_mean is not None or running_var is not None:
                running_mean[:] = (running_mean * momentum) + (1 - momentum) * mi
                running_var[:] = (running_var * momentum) + (1 - momentum) * (n / (n - 1) * sigma2)

            ## get output
            x1 = (input - mi[None, :]) / torch.sqrt(sigma2[None, :] + eps)

        else:  ## Test
            x1 = (input - running_mean[None, :]) / torch.sqrt(running_var[None, :] + eps)
            mean = running_mean
            sigma2 = running_var
        if gamma is None or beta is None:
            output = x1
        else:
            output = (gamma * x1) + beta

        cache = x1, gamma, training, sigma2, eps

        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, ...)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """
        ########################################
        xn, gamma, training, sigma2, eps = cache
        const = gamma / (torch.sqrt(sigma2 + eps))

        if training:
            dinput = const * (
                    doutput - torch.mean(doutput, dim=0) - ((1 / doutput.shape[0]) * xn) * torch.sum(doutput * xn,
                                                                                                     dim=0)
            )
        else:
            dinput = const * doutput

        dgamma = torch.sum(doutput * xn, dim=0)
        dbeta = torch.sum(doutput, dim=0)

        ########################################

        return dinput, dgamma, dbeta


class BatchNorm2d(Function):

    @staticmethod
    def forward(
            input: torch.Tensor,
            gamma: Optional[torch.Tensor],
            beta: Optional[torch.Tensor],
            running_mean: Optional[torch.Tensor] = None,
            running_var: Optional[torch.Tensor] = None,
            momentum: float = 0.9,
            eps: float = 1e-05,
            training: bool = False
    ) -> tuple[torch.Tensor, tuple]:
        """
        Spatial BatchNorm for convolutional networks
        Args:
            input: shape (num_samples, num_channels, height, width)
            gamma: shape (num_channels,)
            beta: shape (num_channels,)
            running_mean: shape (num_channels,)
            running_var: shape (num_channels,)
            momentum: running average smoothing coefficient
            eps: for numerical stabilization
            training: whether in training mode or eval mode
        Returns:
            output: shape (num_samples, num_channels, height, width)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        N, C, M1, M2 = input.shape
        input = input.permute(0, 2, 3, 1) # from N*C*M1*M2 to N*M1*M2*C
        input = input.reshape(N * M1 * M2, C)
        output, cache = BatchNorm1d.forward(input, gamma, beta, running_mean, running_var, momentum, eps, training)
        output = output.reshape(N, M1, M2, C).permute(0, 3, 1, 2)
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_channels, height, width)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        N, C, M1, M2 = doutput.shape
        doutput = doutput.permute(0, 2, 3, 1).reshape(N * M1 * M2, C)
        dinput, dgamma, dbeta = BatchNorm1d.backward(doutput, cache)
        dinput = dinput.reshape(N, M1, M2, C).permute(0, 3, 1, 2)
        ########################################
        return dinput, dgamma, dbeta


class Conv2d(Function):

    @staticmethod
    def forward(
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor],
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, num_channels, height, width)
            weight: shape (num_filters, num_channels, kernel_size[0], kernel_size[1])
            bias: shape (num_filters,)
            stride: convolution step size
            padding: how much should the input be padded on each side by zeroes
            dilation: see torch.nn.functional.conv2d
            groups: see torch.nn.functional.conv2d
        Returns:
            output: shape (num_samples, num_filters, output_height, output_width)
            cache: tuple of intermediate results to use in backward
        """
        ########################################

        output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
        cache = input, weight, bias, stride, padding, dilation, groups
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_filters, output_height, output_width)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input, weight and bias
        """

        ########################################
        input, weight, bias, stride, padding, dilation, groups = cache
        output_padding = (0, 0)
        if stride > 1:
            K = weight.shape[2]
            M = input.shape[2], input.shape[3]  # w/h
            Q = doutput.shape[2], doutput.shape[3]  # q1/q2
            output_padding = tuple([(m - (q - 1) * stride + (2 * padding) - (K - 1) * dilation - 1)
                                    for m, q in zip(M, Q)])

        dinput = torch.nn.functional.conv_transpose2d(input=doutput,
                                                      weight=weight,
                                                      stride=stride,
                                                      padding=padding,
                                                      output_padding=output_padding,
                                                      groups=groups,
                                                      dilation=dilation).to(dtype = input.dtype)

        dbias = doutput.sum(3).sum(2).sum(0)

        dweight = torch.nn.functional.conv2d(input=input.movedim(0, 1),  # switch dim to apply conv2d
                                             weight=doutput.movedim(0, 1),
                                             stride=dilation,
                                             padding=padding,
                                             dilation=stride,
                                             groups=groups)
        #
        dweight = dweight[:, :, 0:weight.shape[2], 0:weight.shape[3]]

        ########################################

        return dinput, dweight.movedim(0, 1), dbias


class MaxPool2d(Function):

    @staticmethod
    def forward(input: torch.Tensor, window_size: int = 2) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, num_channels, height, width)
            window_size: size of pooling window
        Returns:
            output: shape (num_samples, num_channels, height / window_size, width / window_size)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        ## variable prepare
        N, C, M1, M2 = input.shape
        Q = int(M1 / window_size), int(M2 / window_size)
        redundant = M1 % window_size, M2 % window_size
        input = input[:, :, 0:(M1 - redundant[0]), 0:(M2 - redundant[1])]  # odstranění hodnot
        ##MAXPOOL
        output = input.reshape(N, C, Q[0], window_size, Q[1], window_size)  # reshape pro šířku i výšku
        output = output.transpose(3, 4)  # transpozice přidaných pozic na poslední místa

        output = output.reshape(N, C, Q[0], Q[1], window_size * window_size)  # reshape
        output, idx = output.max(dim=4)  # maxpool
        ########################################
        cache = N, C, M1, M2, window_size, idx
        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_channels, height / window_size, width / window_size)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # INIT dat
        N, C, M1, M2, window_size, idx = cache
        Q = int(M1 / window_size), int(M2 / window_size)
        redundant = (M1 % window_size, M2 % window_size)
        dinput = torch.zeros(N, C, M1, M2).to(dtype =doutput.dtype,device=doutput.device)
        # získat matici před .max()
        dinput = dinput[:, :, :M1 - redundant[0], :M2 - redundant[1]]
        dinput = dinput.reshape(N, C, Q[0], window_size, Q[1], window_size).transpose(3, 4)
        dinput = dinput.reshape(N, C, Q[0], Q[1], window_size * window_size)
        # nahradit nuly na indexech maximy
        dinput = dinput.scatter(dim=4, index = idx.unsqueeze(-1),src = doutput.unsqueeze(-1))
        # vrátit do původního tvaru - reverzní postup
        dinput = dinput.reshape(N, C, Q[0], Q[1], window_size, window_size).transpose(3, 4)
        dinput = dinput.reshape(N, C, M1 - redundant[0], M2 - redundant[1])
        # přidat odříznuté hodnoty
        dinput = torch.nn.functional.pad(dinput, (0, M2 - dinput.shape[3], 0, M1 - dinput.shape[2]))

        return dinput,
