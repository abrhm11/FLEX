from math import ceil
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Functional
import torch_dct


########################################################################################################################
# Functional blocks
class Normalizer(nn.Module):
    def __init__(self, numChannels, momentum=0.985, channelNorm=True, batchSize=1):
        super(Normalizer, self).__init__()

        self.momentum = momentum
        self.numChannels = numChannels
        self.channelNorm = channelNorm

        self.register_buffer("movingAverage", torch.zeros(1, numChannels, 1))
        self.register_buffer("movingVariance", torch.ones(1, numChannels, 1))

        self.BatchNormScale = nn.Parameter(torch.ones(1, numChannels, 1))
        self.BatchNormBias = nn.Parameter(torch.zeros(1, numChannels, 1))

    def forward(self, x):
        # Apply channel wise normalization
        if self.channelNorm:
            x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 0.00001)

        # If in training mode, update moving per channel statistics
        if self.training:
            newMean = torch.mean(x, dim=(0, 2), keepdim=True)
            self.movingAverage = ((self.momentum * self.movingAverage) + ((1 - self.momentum) * newMean)).detach()
            x = x - self.movingAverage

            newVariance = torch.mean(torch.pow(x, 2), dim=(0, 2), keepdim=True)
            self.movingVariance = ((self.momentum * self.movingVariance) + ((1 - self.momentum) * newVariance)).detach()
            x = x / (torch.sqrt(self.movingVariance) + 0.00001)
        else:
            x = (x - self.movingAverage) / (torch.sqrt(self.movingVariance) + 0.00001)

        # Apply per channel affine transform
        x = (x * torch.abs(self.BatchNormScale)) + self.BatchNormBias

        return x


class SeperableDenseNetUnit(nn.Module):
    """
    Module that defines a sequence of two convolutional layers with selu activation on both. Channel Normalization
    and stochastic batch normalization with a per channel affine transform is applied before each non-linearity.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernelSize,
        groups=1,
        dilation=1,
        channelNorm=True,
        multiplier=4,
        useNormalizer=True,
        batchSize=1,
    ):
        super(SeperableDenseNetUnit, self).__init__()

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernelSize = kernelSize
        self.groups = groups
        self.dilation = dilation
        self.useNormalizer = useNormalizer

        # Convolutional transforms
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=kernelSize,
            # padding=(kernelSize + ((kernelSize - 1) * (dilation - 1)) - 1) // 2,
            padding=ceil((kernelSize + ((kernelSize - 1) * (dilation - 1)) - 1) / 2),
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            in_channels=in_channels, out_channels=multiplier * out_channels, groups=1, kernel_size=1, padding=0, dilation=1
        )

        self.conv3 = nn.Conv1d(
            in_channels=multiplier * out_channels,
            out_channels=multiplier * out_channels,
            groups=multiplier * out_channels,
            kernel_size=kernelSize,
            padding=(kernelSize + ((kernelSize - 1) * (dilation - 1)) - 1) // 2,
            dilation=dilation,
        )
        self.conv4 = nn.Conv1d(
            in_channels=multiplier * out_channels, out_channels=out_channels, groups=1, kernel_size=1, padding=0, dilation=1
        )

        if self.useNormalizer:
            self.norm1 = Normalizer(numChannels=multiplier * out_channels, channelNorm=channelNorm, batchSize=batchSize)
            self.norm2 = Normalizer(numChannels=out_channels, channelNorm=channelNorm, batchSize=batchSize)

    def forward(self, x):
        # Apply first convolution block
        y = self.conv2(self.conv1(x))
        if self.useNormalizer:
            y = self.norm1(y)
        y = Functional.selu(y)

        # Apply second convolution block
        y = self.conv4(self.conv3(y))
        if self.useNormalizer:
            y = self.norm2(y)
        y = Functional.selu(y)

        # Return densely connected feature map
        return torch.cat((y, x), dim=1)


########################################################################################################################
# Define the Sleep model


class SkipLSTM(nn.Module):
    """
    Module that defines a bidirectional LSTM model with a residual skip connection with transfer shape modulated with a
    mapping 1x1 linear convolution. The output results from a second 1x1 convolution after a tanh nonlinearity,
    critical to prevent divergence during training.
    """

    def __init__(self, in_channels, out_channels=4, hiddenSize=32):
        super(SkipLSTM, self).__init__()

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Bidirectional LSTM to apply temporally across input channels
        self.rnn = nn.LSTM(
            input_size=in_channels, hidden_size=hiddenSize, num_layers=1, batch_first=True, dropout=0.0, bidirectional=True
        )

        # Output convolution to map the LSTM hidden states from forward and backward pass to the output shape
        self.outputConv1 = nn.Conv1d(in_channels=hiddenSize * 2, out_channels=hiddenSize, groups=1, kernel_size=1, padding=0)
        self.outputConv2 = nn.Conv1d(in_channels=hiddenSize, out_channels=out_channels, groups=1, kernel_size=1, padding=0)

        # Residual mapping
        self.identMap1 = nn.Conv1d(in_channels=in_channels, out_channels=hiddenSize, groups=1, kernel_size=1, padding=0)

    def forward(self, x):
        y = x.permute(0, 2, 1)
        y, z = self.rnn(y)
        z = None
        y = y.permute(0, 2, 1)
        y = torch.tanh((self.outputConv1(y) + self.identMap1(x)) / 1.41421)
        y = self.outputConv2(y)
        #y = torch.softmax(y, dim=1)

        return y


class Sleep_model_MultiTarget(nn.Module):
    def addSeperableDenseNetUnit(self, kernelSize=None, groups=1, dilation=1, channelNorm=True, multiplier=4, batchSize=1):
        kernelSize = self.kernelSize if kernelSize is None else kernelSize
        in_channels = self.curChannels
        out_channels = self.numSignals * self.channelMultiplier
        self.curChannels += out_channels
        return SeperableDenseNetUnit(
            in_channels,
            out_channels,
            kernelSize,
            groups,
            dilation,
            channelNorm,
            multiplier=multiplier,
            useNormalizer=self.useNormalizer,
            batchSize=batchSize,
        )

    def __init__(
        self,
        numSignals=12,
        binClasses=[1, 1, 1, 1],
        channelMultiplier=2,
        dilationLayers=[1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1],
        kernelSize=25,
        useSkipLLSTM=True,
        unitMultiplierMod=4,
        unitMultiplierDS=4,
        lstmChannels=64,
        downSampleSteps=[2, 5, 5, 5, 6],
        skipMarginalize=False,
        useNormalizer=False,
        batchSize=1,
    ):
        super(Sleep_model_MultiTarget, self).__init__()
        self.useSkipLLSTM = useSkipLLSTM
        self.channelMultiplier = channelMultiplier
        self.kernelSize = kernelSize
        self.binClasses = binClasses
        self.numSignals = numSignals
        self.curChannels = self.numSignals
        self.dilationLayers = dilationLayers
        self.downSampleSteps = downSampleSteps  # = downsample factor of 1500
        self.skipMarginalize = skipMarginalize
        self.useNormalizer = useNormalizer

        # Set up downsampling densenet blocks
        downsampleKernel = (2 * self.kernelSize) + 1
        for i, _ in enumerate(self.downSampleSteps):
            self.__setattr__(
                "dsMod%i" % i,
                self.addSeperableDenseNetUnit(
                    kernelSize=downsampleKernel, channelNorm=False, multiplier=unitMultiplierDS, batchSize=batchSize
                ),
            )
        for i, dilation in enumerate(self.dilationLayers):
            self.__setattr__(
                "denseMod%i" % i,
                self.addSeperableDenseNetUnit(dilation=dilation, multiplier=unitMultiplierMod, batchSize=batchSize),
            )
        # dsMods = []
        # for i, _ in enumerate(self.downSampleSteps):
        #     dsMods.append(
        #         self.addSeperableDenseNetUnit(kernelSize=downsampleKernel, channelNorm=False, multiplier=unitMultiplierDS),
        #     )
        # self.dsMods = nn.Sequential(*dsMods)

        # denseMods = []
        # for i, dilation in enumerate(self.dilationLayers):
        #     denseMods.append(self.addSeperableDenseNetUnit(dilation=dilation, multiplier=unitMultiplierMod))

        # self.denseMods = nn.Sequential(*denseMods)
        

        if self.useSkipLLSTM:
            out = sum(self.binClasses) if self.skipMarginalize and self.binClasses[0] > 1 else 1
            self.skipLSTM = SkipLSTM(
                self.curChannels, hiddenSize=self.channelMultiplier * lstmChannels, out_channels=out
            )

    def cuda(self, device=None):
        self.skipLSTM.rnn = self.skipLSTM.rnn.cuda(device)

        return super(Sleep_model_MultiTarget, self).cuda(device)

    def forward(self, x_segment_dct, x, segment_idx, segment_length):
        # x = x.detach().contiguous()
        x_segment = torch_dct.idct(x_segment_dct)
        x_segment = x_segment.view(1, 1, segment_length)
        lower = segment_idx * segment_length
        upper = lower + segment_length
        x[:, :, lower:upper] = x_segment
        for i, kernelSize in enumerate(self.downSampleSteps):
            x = self.__getattr__("dsMod%i" % i)(x)
            x = Functional.max_pool1d(x, kernel_size=kernelSize)

        for i, _ in enumerate(self.dilationLayers):
            x = self.__getattr__("denseMod%i" % i)(x)

        # for i, dsMod in enumerate(self.dsMods):
        #     kernelSize = self.downSampleSteps
        #     x = dsMod(x)
        #     x = Functional.max_pool1d(x, kernel_size=kernelSize)

        # for _, denseMod in enumerate(self.denseMods):
        # x = denseMod(x)

        # Bidirectional skip LSTM and convert joint predictions to marginal predictions
        if self.useSkipLLSTM:
            x = self.skipLSTM(x)

        if self.skipMarginalize:
            x = torch.softmax(x, dim=1)
            return x
    

        xses = marginalize(x, self.binClasses)

        if not (self.training):
            for i in range(len(xses)):
                xses[i] = torch.exp(xses[i])

        xses = [torch.softmax(x, dim=1) for x in xses]
        return xses

    def loss(self, output, target, batchSize, classificator):
        weightTensors = torch.IntTensor(classificator["classWeights"])
        l = torch.nn.MultiLabelSoftMarginLoss(reduction="mean", weight=weightTensors).cuda()
        output = output.permute(0, 2, 1).contiguous().squeeze()
        return l(output, target.squeeze()) * classificator["weight"], 0

    def remapOutputsToPrediction(self, outputs):
        pred = []
        for i, classCount in enumerate(self.binClasses):
            pred.append(outputs[i].data.cpu().permute(0, 2, 1).contiguous().view(-1, classCount))
        return pred
    
class CatCrossEntropy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, out, targets):
        targets = targets.argmax(dim=1)
        return self.loss(out, targets)
    
    
'''
class DRCNN(ModelTorchAdapter):
    def initialOptions(self):
        return {
            "channelMultiplier": 2,
            "dilationLayers": [1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1],
            "kernelSize": 25,
            "useSkipLLSTM": True,
            "skipMarginalize": False,
            "unitMultiplierDS": 4,
            "unitMultiplierMod": 4,
            "lstmChannels": 64,
            "downSampleSteps": 64,
            "useNormalizer": False,
        }

    def define(self):
        inputShape = self.inputShape
        numClasses = self.config.numClasses
        batchSize = self.batchSize
        skipMarginalize = self.getOption("skipMarginalize")
        self.model = Sleep_model_MultiTarget(
            numSignals=inputShape[1],
            binClasses=[numClasses],
            dilationLayers=self.getOption("dilationLayers"),
            channelMultiplier=self.getOption("channelMultiplier"),
            kernelSize=self.getOption("kernelSize"),
            useSkipLLSTM=self.getOption("useSkipLLSTM"),
            lstmChannels=self.getOption("lstmChannels"),
            unitMultiplierMod=self.getOption("unitMultiplierMod"),
            unitMultiplierDS=self.getOption("unitMultiplierDS"),
            downSampleSteps=self.getOption("downSampleSteps"),
            skipMarginalize=self.getOption("skipMarginalize"),
            useNormalizer=self.getOption("useNormalizer"),
            batchSize=batchSize,
        )
        self.skipMarginalize = skipMarginalize

        self.weightTensors = None if self.classWeights is None else torch.IntTensor(self.classWeights)
        if self.useGPU:
            self.model.cuda()

    def getLossFunction(self):
        loss = CatCrossEntropy()
        if self.useGPU:
            loss.cuda()
        return loss

    def mapOutput(self, output):
        return output if self.skipMarginalize else output[0]

    def mapOutputForLoss(self, output, mask):
        output = output if self.skipMarginalize else output[0]
        mask = mask.reshape(output.shape[0], output.shape[2])
        return output.permute(0, 2, 1).contiguous().squeeze()[mask]

    def mapOutputForPrediction(self, output):
        output = output if self.skipMarginalize else output[0]
        return output.permute(0, 2, 1)
    
    def loadState(self, state):
        # convert from old model
        if 'dsMods.0.conv1.weight' in state:
            for name in state.copy():
                spl = name.split('.')
                if spl[0] in ['dsMods', 'denseMods']:
                    newName = 'dsMod' if spl[0] == 'dsMods' else 'denseMod'
                    state[newName + '.'.join(spl[1:])] = state[name]
                    del state[name]

        return super().loadState(state)
    def forward(self, x):
        return self.model(x)
    
'''

def marginalize(inputX, outgoingBins: List[int]):
    p_joint = Functional.log_softmax(inputX, dim=1)

    classesX = []
    index = 0
    for counts in outgoingBins:
        xNotInit = True
        x = torch.tensor([])
        for i in range(counts):
            # Compute marginal for sleepstages
            p = p_joint[::, index, ::]
            unsqueezed = p.unsqueeze(1)
            if counts > 1:
                xsub = unsqueezed
                x = xsub if xNotInit else torch.cat((x, xsub), dim=1)
            else:
                xsub = torch.cat((unsqueezed, torch.log(1 - torch.exp(unsqueezed))), dim=1)
                x = xsub
            xNotInit = False
            index += 1
        classesX.append(x)

    return classesX