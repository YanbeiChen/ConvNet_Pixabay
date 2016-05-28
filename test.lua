
--[[
   1. Create loggers.
   2. Preallocate some varibles and parameters
   3. test - this function handles the high-level test loop,
              i.e. load data, test model, and return the validation error in each epoch
]]--
----------------------------------------------------------------------
-- 1. Create loggers.
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))


----------------------------------------------------------------------
-- 2. Preallocate some varibles and parameters

-- initialize related parameters used for record in each epoch
local batchNumber -- count how many batches in one epoch
local top1_center, loss_epoch -- averaged error average samples and loss in each epoch

-- predefine clocks
local timer = torch.Timer()

-- GPU inputs (preallocate)
-- initializae data
local inputs = torch.CudaTensor()
local targets = torch.CudaTensor()


----------------------------------------------------------------------
-- 3. test - this function handles the high-level test loop,
--              i.e. load data, test model, and return the validation error in each epoch
function test()
    print('==> doing epoch on validation data:')
    print("==> online epoch # " .. epoch)

    -- reset time to count time for training in one epoch
    timer:reset()

    -- refresh related parameters at the start of each epoch 
    -- include: batchNumber,loss_epoch,top1_epoch
    batchNumber = 0
    top1_center = 0
    -- loss_epoch = 0

    -- set the dropouts to evaluate mode
    model:evaluate()

    ------------------------LOOP for Test Batch---------------------------
    -- how many datapoints in the test set: testData.dataSize
    for i = 1, testData.dataSize/opt.batchSize do
       collectgarbage()

       -- load training data
       inputsCPU, targetsCPU = testData:MiniBatch()
       -- transfer over to GPU
       inputs:resize(inputsCPU:size()):copy(inputsCPU)
       targets:resize(targetsCPU:size()):copy(targetsCPU)
       
       -- forward pass
       local outputs = model:forward(inputs)
       -- local err = loss:forward(outputs, targets)

       -- update related parameter obtained in one batch:
       -- include: batchNumber,loss_epoch,top1_epoch
       batchNumber = batchNumber+1
       -- loss_epoch = loss_epoch + err
       -- compute top-1 error on test data as the validation error
       do
          local _,prediction_sorted = outputs:float():sort(2, true) -- descending
          for i = 1,opt.batchSize do
            if targetsCPU[i][prediction_sorted[i][1]]==1 then
                top1_center = top1_center + 1;
            end
          end
       end
       
       -- print out the test progress
        if batchNumber % 1024 == 0 then
          print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, testData.dataSize))
        end
      
    end

    ------------------------Parameters for one epoch----------------------

    -- how many datapoints are used: (batchNumber*opt.batchSize)
    top1_center = top1_center * 100 / (batchNumber*opt.batchSize)
    -- loss_epoch = loss_epoch / batchNumber -- because loss is calculated per batch

    -- record averaged top 1 error and loss of test data in each epoch
    testLogger:add{
      ['% top1 accuracy (test set) (center crop)'] = top1_center
    }
    print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t ',
                       epoch, timer:time().real, top1_center))
    print('\n')

    return top1_center
end -- of test()
-----------------------------------------------------------------------------
