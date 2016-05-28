--[[
  This class will load the paths for either for train set.
  The class returns a path container for training data,
  which stores the list of labels and their corresponding samples' paths
  
  To use output of this class:
  example:
  first initialize: trainpath = trainContainer(opt)
  
--]]

-- To use the output of this class to load data:
-- filename: imgfolder..option1..'/'..option2..'/'..imgpath[class[j]][i] 
-- where option1 is flip option, option2 is random cropping option

require 'torch'

-- Define a class: trainContainer
-- creat a class "trainContainer"
local trainContainer = torch.class('trainContainer')

-- the initializer
-- self is used to denote object variables
function trainContainer:__init(opt)
    print('===>loading paths for training samples now... and storing paths in a path container:')
    -- Define a clock
    local timer = torch.Timer()
    
    -- Read the labels of dataset
    local labelPath = opt.labelPath
    local fh,err
    local line
    fh,err = io.open(labelPath)
    if err then print("OOps, not open:"..labelPath); return;
    else print("Open file :"..labelPath); end
    local class = {}
    local index = 1
    while true do
        line = fh:read()
        if line == nil then break end
        class[index] = line
        index = index+1
    end
    fh:close()
    print('Finish reading indexes of training set.')

    -- initialize variables
    -- Define paths of folders for images and labels
    local ImgFolderPath = '/storage/data/Pixabay/jitterimgs/'
    local LabelFolderPath = '/home/yanbei/visualmodel2/data/trainData/'
    local appendzero = {'00000000','0000000','000000','00000','0000','000'}
    local digits
    local labelcontainer = {}
    local labelindex = {}

    -- read from documents and store the paths
    for i = 1,#class do
        index = 1
        labelindex = {}
        local path = LabelFolderPath..class[i]..'.txt'
        fh,err = io.open(path)
        if err then print("OOps, not open:"..path); return; end
        while true do
            line = fh:read()
            if line == nil then break end
            digits = string.len(line)
            labelindex[index] = appendzero[digits]..line..'.jpg'
            index = index+1
        end
        fh:close()
        labelcontainer[class[i]] = labelindex
    end
    
    -- Count time and print
    print('===>loading paths for training samples takes '..timer:time().real..'s\n')
    
    -- initialize the container: (imgpath, labelpath)
    self.class = class
    self.imgpath = labelcontainer
    self.imgfolder = ImgFolderPath
end
