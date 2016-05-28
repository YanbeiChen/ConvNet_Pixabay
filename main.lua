require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

-------------------------------INITIALIZATION--------------------------
-- torch.setdefaulttensortype('torch.FloatTensor')

-- initializes user preference variables
paths.dofile('opts.lua')
print(opt)

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

-- type: use which GPU
if opt.type == 'cuda' then
  print(sys.COLORS.red ..  '==> switching to CUDA')
  require 'cunn'
  cutorch.setDevice(opt.devid) -- by default, use GPU 1
  print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

-- save models to a new folder
print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)


-------------------------------DATA-----------------------------------
-- load paths of data
-- (1) use class 'trainContainer' to initialize path of dataset
require 'trainContainer'
local trainpath = trainContainer(opt)
require 'testContainer'
local testpath = testContainer(opt)

-- (2) use class 'trainLoader' to load data
-- initialize the dataloader for train set and test set
-- this things are global
require 'trainLoader'
trainData = trainLoader(opt, trainpath)
require 'testLoader'
testData = testLoader(opt, testpath)

------------------------------MODEL-----------------------------------
-- Model + Loss:
-- initialize model
local t = require 'alexnet'
-- this things are global
model = t.model
loss = t.loss

-- CUDify the model
if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

------------------------------MODEL-----------------------------------
-- initialized varibles for train and test
paths.dofile('train.lua')
paths.dofile('test.lua')

-- initialize the test error
-- it is returned by the test function
-- it is used to anneal the learning rate
local test_error = 0


-- initialize the learning rate
-- it is returned by the train function
-- it is used to terminate the training
local learning_rate = 0.1

-- this thing is global
epoch = opt.epochNumber

for i=1,opt.nEpochs do
    -- training and test has interaction:
    -- learning rate is annealed according to the test error
    learning_rate = train(test_error)
    test_error = test()
    
    -- if the learning rate reach the minimal level
    if learning_rate<=opt.minLR then
        break
    end
    
    epoch = epoch + 1
end