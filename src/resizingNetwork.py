
import libraryImport as li
# Build my Model

class resBlock(nn.Module):#Resnet Block as defined in original paper.
    def __init__(self, channelSize = 16,negativeSlope = 0.02, kernelSize = 3, stride=1):
        super(resBlock, self).__init__()

      #Define the sequential NN based on the structure listed in our paper
        self.convBlock = nn.Sequential(
        nn.Conv2d(channelSize, kernelSize, stride),
        nn.BatchNorm2d(channelSize),
        nn.LeakyReLU(negativeSlope),
        nn.Conv2d(channelSize, kernelSize, stride),
        nn.BatchNorm2d(channelSize)
      )

    def forward(self, x):#Sum at the end of the resblock

        return x + self.convBlock(x)



class resizingNetwork(nn.Module):
    def __init__(self,channelSize = 16, kernelSize = 7,negativeSlope = 0.02, stride=1, inputSize = (480,640),resizerMode = 'bilinear',numResBlock = 3 ):
      super(resizingNetwork, self).__init__()


        #Block1 as defined in the initial structure
      self.convBlock1 = nn.Sequential(
          nn.Conv2d(channelSize, kernelSize, stride),
          nn.LeakyReLU(negativeSlope),
          nn.Conv2d(channelSize, 1, stride),
          nn.BatchNorm2d(channelSize),
          nn.LeakyReLU(negativeSlope),
      )

      #Bilinear Interpolation
      self.interpolate = F.interpolate(size = (224,224), mode = resizerMode, align_corners=False)

      #ResBlocks.
      self.blockList = [0] * numResBlock
      for r in range(numResBlock):
        self.blockList[r] = resBlock(channelSize, negativeSlope, kernelSize, stride)

      # Block 2 as defined in the structure
      self.convBlock2 = nn.Sequential(
          nn.Conv2d(channelSize, 1, stride),
          nn.BatchNorm2d(channelSize)
      )

      convBlock3 = nn.conv2d(channelSize,kernelSize, stride)



    def forward(self, x):
      #Initial Interpolation
      bilinearOriginal = self.interpolate(x)

      x = self.convBlock1(x)

      #Modified Interpolation
      x = bilinearModified = self.interpolate(x)

      #Apply number of resBlocks
      for resBlock in self.blockList:
        x = resBlock(x)

      x = self.convBlock2(x)
      x += bilinearModified
      x = self.convBlock3(x)


      return x +  bilinearOriginal




#inhereted Resnet 50 Classifier Model, classes to be determined later. Due to paper not specifying class size.

class resnet50Classifier(nn.Module):
    def __init__(self, numClasses =1000, preTrained = False):
        super(resnet50Classifier, self).__init__()

        # Load the un-trained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained= preTrained)

        # Modify the last fully connected layer to match the number of classes
        numFeatures = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(numFeatures, numClasses)

    def forward(self, x):
        return self.resnet50(x)