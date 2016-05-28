require 'nn'
require 'cunn'

local SpatialConvolution = nn.SpatialConvolution--lib[1]
local SpatialMaxPooling = nn.SpatialMaxPooling--lib[2]

-- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
-- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
local features = nn.Sequential()
features:add(SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
features:add(nn.ReLU(true))
features:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
features:add(SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
features:add(nn.ReLU(true))
features:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
features:add(SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
features:add(nn.ReLU(true))
features:add(SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
features:add(nn.ReLU(true))
features:add(SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
features:add(nn.ReLU(true))
features:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

local classifier = nn.Sequential()
classifier:add(nn.View(256*6*6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*6*6, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))

-- last layer: the number of output needs to be changed from 1000 classes to opt.totalClass
-- classifier:add(nn.Linear(4096, 1000))
classifier:add(nn.Linear(4096, opt.totalClass))

-- classifier:add(nn.LogSoftMax())
-- With multiple labels per sample, you would have nn.Sigmoid as the last layer of your network
classifier:add(nn.LogSoftMax())

local model = nn.Sequential()

model:add(features):add(classifier)

-- Loss: NLL
loss = nn.ClassNLLCriterion()

return {
  model = model,
  loss = loss,
}