--[[
  This class will load the paths for either for validation set or said test set
  The class returns a path container: (imgpath, labelpath),
  which stores the sample paths both for image and image's label

  To use the output of this class to load data:
  use imgpath and image.load(filename) to get image:
      filename: imgfolder..option1..'/'..option2..'/'..imgpath[i]
      where option1 is flip option, option2 is random cropping option
  use labelpath path and io.open(filename) to get label:
      filename: labelpath[i]
--]]


require 'torch'

-- Define a class: testContainer
-- creat a class "testContainer"
local testContainer = torch.class('testContainer')


-- the initializer
-- self is used to denote object variables
function testContainer:__init(opt)
  -- Read the paths of validation set
  local indexPath = '/home/yanbei/visualmodel2/data/test.txt'

  print('===>loading paths for validation samples now... and storing paths in a path container: (imgpath, labelpath)')
  
  -- Define a clock
  local timer = torch.Timer()

  -- Define paths of folders for images and their labels
  local ImgFolderPath = '/storage/data/Pixabay/jitterimgs/'
  local LabelFolderPath = '/home/yanbei/visualmodel2/data/testData/'
  
  -- Initialize variables
  local appendzero = {'00000000','0000000','000000','00000','0000','000'}
  local ImgPath = {}
  local LabelPath = {}
  local ind = 1
  
  -- Open index txt file and initialize path variable
  local fh,err = io.open(indexPath)
  local line
  if err then print("OOps, not open:"..indexPath); return;
  else print("Open file :"..indexPath); end
  while true do
      line = fh:read()
      if line == nil then break end
      digits = string.len(line)
      -- both image and label's index start from 0
      ImgPath[ind] = appendzero[digits]..line..'.jpg'
      LabelPath[ind] = LabelFolderPath..appendzero[digits]..line..'.txt'
      ind = ind+1
  end
  fh:close()
  
  -- Count time and print
  print('===>loading paths for validation samples takes '..timer:time().real..'s\n')
  
  -- initialize the container: (imgpath, labelpath)
  self.imgfolder = ImgFolderPath
  self.imgpath = ImgPath
  self.labelpath = LabelPath
  
end
