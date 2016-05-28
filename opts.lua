---------------------OPTS---------------------------------------------
--[[
  This script initializes user preference variables
  and use torch.CmdLine() to extract command line arguments
--]]

local cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training AlexNet on Pixabay')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Training Options')
cmd:option('-LR',                 0.01,                      'learning rate')
cmd:option('-minLR',              1e-6,                     'minimal learning rate')
cmd:option('-LRDecay',            0,                        'learning rate decay (in # samples)')
cmd:option('-weightDecay',        5e-4,                     'L2 penalty on the weights')
cmd:option('-momentum',           0.9,                      'momentum')
cmd:option('-batchSize',          128,                      'batch size')
cmd:option('-epochSize',          10000,                    'Number of batches per epoch')
cmd:option('-epochNumber',        1,                        'Manual epoch number (useful on restarts)')
cmd:option('-seed',               123,                      'torch manual random number generator seed')
cmd:option('-nEpochs',            50000,                       'Number of total epochs to run')

cmd:text('===>Data Options')
cmd:option('-totalClass',         2907,                     'total amount of classes')
cmd:option('-cropSize',           224,                      'image crop size')
cmd:option('-scaleSize',          256,                      'image rescale size')
-- cmd:option('-indexPath',          '/Users/apple/Desktop/visualmodel1index/',
cmd:option('-indexPath',          '/home/yanbei/visualmodel1/data/visualmodel1index/',
'Path stores indexes for dataset')
-- cmd:option('-labelPath',          '/Users/apple/Documents/model2label/Data/label.txt',
cmd:option('-labelPath',          '/home/yanbei/visualmodel2/data/label.txt',
'Path stores indexes for dataset')
cmd:text('===>Optimization Options')
cmd:option('-optimState',         'none',                   'provide path to an optimState to reload from')
cmd:option('-optimization',       'sgd',                    'optimization method')
cmd:option('-threads',            8,                        'number of threads')
cmd:option('-type',               'cuda',                   'float or cuda')
cmd:option('-devid',               3,                        'device ID (if using CUDA)')
--cmd:option('-nGPU',               1,                        'num of gpu devices used')

cmd:text('===>Save/Load Options')
--cmd:option('-load',               '',                       'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''),   'save directory')
--cmd:option('-optState',           false,                    'Save optimization state every epoch')
--cmd:option('-checkpoint',         0,                        'Save a weight check point every n samples. 0 for off')

--cmd:text('===>Other Options')

opt = cmd:parse(arg or {})
