
import libraryImport as li
import resizingNetwork
import trainFuncs



if torch.cuda.is_available():
  print("GPU is available")
  device = torch.device('cuda')
else:
  print("GPU is not available")
  device = torch.device('cpu')



#Data Loader - Load images into model for training and validation
#Redo to use ImageNette
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


trainDataSet  = torchvision.datasets.MINST('data', train=True, download=True, transform=transform)
testDataSet = torchvision.datasets.MINST('data', train=False, download=True, transform=transform)


trainLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=batchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(testDataSet, batch_size=batchSize, shuffle=False)


#Define Classifiers - Both Learned from Scratch, and

classifierTraditional = resnet50Classifier().to(device)

classifierModified = modifiedClassifier(resizingNetwork(), resnet50Classifier()).to(device)



#Comparison Model - Classify both Traditional Biliinear Resizer, and Learned Resizer to run, and then compare outputs.
optimizer = optim.SGD(classifierModified.parameters(), lr=learningRate, momentum=0.9)
start = time.time()

for epoch in range(1,maxEpoch + 1):
  trainLoss = train(classifierModified, nn.CrossEntropyLoss(), optimizer, trainLoader, epoch)
  testStat = test(classifierModified, nn.CrossEntropyLoss(), testLoader, epoch)

  trainLossTraditional = train(classifierTraditional, nn.CrossEntropyLoss(), optimizer, trainLoader, epoch)
  testStatTraditional = test(classifierTraditional, nn.CrossEntropyLoss(), testLoader, epoch)



end = time.time()
print(f'Finished Training after {end-start} s ')