--[[

  This class will load the images and their corresponding labels for validation set/test set
  To initialize the class, use input from another class 'testContainer'
  With the images' paths and labels' paths from the 'testContainer',
  the class load images and labels for a mini-batch when calling MiniBatch()

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

  When all samples are used in one epoch,
  the data will be automatcally shuffled by calling Shuffle()

  To use output of this class:
  example:
  first initialize: testloader = testLoader(opt, testpath)
  then obtain one mini-batch: test_data, test_labels = testLoader:MiniBatch()
  
--]]   

require 'image'
require 'torch'

-- Define a class: testLoader
-- creat a class "testLoader"
local testLoader = torch.class('testLoader')

-- the initializer
-- self is used to denote object variables
function testLoader:__init(opt, pathcontainer)
   -- constants variable by user perference
   -- data properties
   self.batchSize = opt.batchSize
   self.totalClass = opt.totalClass
   -- image preprocessing properties
   self.cropSize = opt.cropSize
   self.scaleSize = opt.scaleSize
   -- cropChoice: center, top left, top right, bottom left and bottom right
   self.cropChoice = {"c", "tl", "tr", "bl", "br"}
   self.flipChoice = {'nflip','flip'}  
  
   -- load paths and datatype from pathcontainer
   self.imgfolder = pathcontainer.imgfolder
   self.imgpath = pathcontainer.imgpath
   self.labelpath = pathcontainer.labelpath
   self.datatype = pathcontainer.datatype
   
   -- initialize data
   self.dataSize = #self.imgpath
   self.data = torch.Tensor(self.batchSize,3,self.cropSize,self.cropSize)
   self.labels = torch.rand(self.batchSize,self.totalClass):fill(0)
   self.ind = 1
end

-- method I (used by method II)
-- shuffle the data by shuffling the paths
function testLoader:Shuffle()
    print('===> Shuffling paths now...')

    -- first, randomly shuffle the indexes of the whole dataset
    local random = torch.randperm(self.dataSize)

    -- update the paths
    -- (1) use new variables to store new paths for image and its corresponding label
    local new_imgpath = {}
    local new_labelpath = {}
    for i = 1,self.dataSize do
      new_imgpath[i] = self.imgpath[random[i]]
      new_labelpath[i] = self.labelpath[random[i]]
    end

    -- (2) reinitialize the variables for paths
    self.imgpath = new_imgpath
    self.labelpath = new_labelpath
end

-- method II
-- MiniBatch data loading function
function testLoader:MiniBatch()
    -- batch fit?
    if(self.ind+self.batchSize-1) > self.dataSize then
      -- do not fit
      self.ind = 1
      self:Shuffle()
    end

    -- Define a clock
    -- local timer = torch.Timer()

    -- reinitialize data
    self.data = torch.Tensor(self.batchSize,3,self.cropSize,self.cropSize)
    self.labels = torch.rand(self.batchSize,self.totalClass):fill(0)

    -- initialize varible for file operation
    local fh,err,line
    
    for i = 1, self.batchSize do
        -- (1) load an preprocessed image with its path directly
        self.data[i] = image.load(self.imgfolder..self.flipChoice[1]..'/'..self.cropChoice[1]..'/'..self.imgpath[i+self.ind-1])
        -- (2) load an image's label by opening the corresonding file
        fh,err = io.open(self.labelpath[i+self.ind-1])
        if err then print("OOps, not open:"..self.labelpath[i+self.ind-1]); return; end
        line = fh:read()
        for token in string.gmatch(line, "[^%s]+") do
              self.labels[i][token]=1
        end
        fh:close()
    end
    
    -- update the index
    -- (next mini-batch will start from new index)
    self.ind = self.ind+self.batchSize

    -- further online preprocessing:
    -- use a floating point representation
    self.data = self.data:float()
    
    -- print('===> Create mini batch takes '..timer:time().real..'s\n')

    return self.data, self.labels
end

