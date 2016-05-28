--[[

  This class will load the images and their corresponding labels
  To initialize the class, use input from another class 'trainContainer'
  With the images' paths and labels from the 'trainContainer',
  in each mini-batch, 
  (1)first sample 128 different classes,
  (2)then randomly picl 1 image for each class. 
  There would be 128 images from different class in one batch.
  
  The data preprocess includes:
  Offline preprocessing:
  Basic preprocessing:(1)rescale image size;(2)crop central part of image; (3) normalization
  (4) Data agumentation:
  data augmentation I: randomly flips image horizontally (left<->right)
  data augmentation II: randomly crops a patch of size CropSizeXCropSize
  Note: We only perform data agumentation on the training set but not on the test set.
        For test set, we always crop the central part!
        
  Online preprocessing:
  Basic preprocessing: (5)convert image to a floating point representation 


  To use output of this class:
  example:
  first initialize: trainloader = trainLoader(opt, trainpath)
  then obtain one mini-batch: train_data, train_labels = trainloader:MiniBatch()
  
--]]     

require 'image'
require 'torch'

-- Define a class: trainLoader
-- creat a class "trainLoader"
local trainLoader = torch.class('trainLoader')

-- the initializer
-- self is used to denote object variables
function trainLoader:__init(opt, pathcontainer)
    -- constants variable by user perference
    -- data properties
    self.batchSize = opt.batchSize
    self.totalClass = #pathcontainer.class
    
    -- image preprocessing properties
    self.cropSize = opt.cropSize
    -- cropChoice: center, top left, top right, bottom left and bottom right
    self.cropChoice = {"c", "tl", "tr", "bl", "br"}
    self.flipChoice = {'nflip','flip'}  
  
    -- load paths from pathcontainer
    self.class = pathcontainer.class
    self.imgpath = pathcontainer.imgpath
    self.imgfolder = pathcontainer.imgfolder

    -- initialize data
    self.data = torch.Tensor(self.batchSize,3,self.cropSize,self.cropSize)
    self.labels = torch.Tensor(self.batchSize)
    
    -- initialize a sampler for classes 
    self.sampler = torch.randperm(self.totalClass)
    self.ind = 1     
    -- (1) random sample 128 classes and make sure they are evenly sampled
end

-- method I (used by method II)
-- shuffle the classes by shuffling the index
-- to make sure they are evenly sampled
function trainLoader:Shuffle()
    print('===> Shuffling classes now...')
    -- first, randomly shuffle the indexes of the whole dataset
    self.sampler = torch.randperm(self.totalClass)
end

-- method II 
-- MiniBatch data loading function
function trainLoader:MiniBatch()
    if (self.ind+self.batchSize-1) > self.totalClass then
      self.ind = 1
      self:Shuffle()
    end
    
    -- Define a clock
    -- local timer = torch.Timer()    
    -- reinitialize data
    -- 'random' the index of the sampled class
    -- it is times to train class 'random' in this mini bathch
    self.data = torch.Tensor(self.batchSize,3,self.cropSize,self.cropSize)
    self.labels = torch.Tensor(self.batchSize)
    
    for i = 1, self.batchSize do 
      -- randomly pick one class:
      local random = self.sampler[self.ind]
      self.ind = self.ind+1
      self.labels[i] = random
      
      -- random sample a random index from the random[i]-th class
      local random_index = math.random(1,math.random(1,#self.imgpath[self.class[random]]))
      self.data[i] = image.load(self.imgfolder..self.flipChoice[math.random(1,2)]..'/'..self.cropChoice[math.random(1,5)]..'/'..self.imgpath[self.class[random]][random_index])
    end
    
    -- further online preprocessing:
    -- use a floating point representation
    self.data = self.data:float()

    -- print('===> Create mini batch takes '..timer:time().real..'s')
    -- print(self.ind)

    return self.data, self.labels
end

 

