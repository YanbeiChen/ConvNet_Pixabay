require 'optim'

--[[
   1. Setup SGD optimization state
   2. Create loggers
   3. Preallocate some varibles and parameters
   4. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
              and return the learning rate
]]--

----------------------------------------------------------------------
-- 1. Setup SGD optimization state
-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}
--in case there is a path we can reload from for an optimState
if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end


----------------------------------------------------------------------
-- 2. Create loggers
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))


----------------------------------------------------------------------
-- 3. Preallocate some varibles and parameters

-- initialize related parameters used for record in each epoch
local batchNumber -- count how many batches in one epoch
local top1_epoch -- averaged error average samples and loss in each epoch
local loss_epoch = 0.00

-- initialize variables used for annealing learning rate
local pre_test_error = 0.00
local epoch_LR = 0
local pre_loss = 1.00 -- previous training loss
local loss_same = 0

-- predefine clocks
local tm = torch.Timer() -- count time for each epoch
local timer = torch.Timer() -- count time for a single batch: data loading and training 

-- GPU inputs (preallocate)
-- initializae data
local inputs = torch.CudaTensor()
local targets = torch.CudaTensor()

-- initialize weight and gradient
local parameters, gradParameters = model:getParameters()


----------------------------------------------------------------------
-- 4. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
--            and return the learning rate
function train(test_error)
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)

    -- (1) anneal learing rate based on test error
    -- if the last learning rate has been used for more than 10 times
    -- and if the test_error has been improved
    -- then change the learning rate
    if test_error > pre_test_error and epoch_LR >=10 then
      print('==> Anneal learing rate')
      -- refresh the epoch_LR counter
      epoch_LR = 1
      -- divide learning rate with 2
      optimState.learningRate = optimState.learningRate/2
    else
      epoch_LR = epoch_LR+1
    end
    -- record the test error in last epoch: pre_test_error
    pre_test_error = test_error

    -- reset time to count time for training in one epoch
    tm:reset()

    -- refresh related parameters at the start of each epoch 
    -- include: batchNumber,loss_epoch,top1_epoch
    batchNumber = 0
    top1_epoch = 0 
    loss_epoch = 0

    -- set the dropouts to training mode
    model:training()

    ------------------------LOOP for TrainBatch---------------------------
    -- this part is the training loop:
    -- In each iteration, load mini-batch data and train on a single batch
    
    for i=1,opt.epochSize do  -- each epoch has opt.epochSize amount of mini-batch
        collectgarbage()
        
        -- reset time for data loading
        timer:reset()
        -- load training data
        inputsCPU, targetsCPU = trainData:MiniBatch()
        -- transfer over to GPU
        inputs:resize(inputsCPU:size()):copy(inputsCPU)
        targets:resize(targetsCPU:size()):copy(targetsCPU)

        -- count time for data loading and reset time
        local dataLoadingTime = timer:time().real
        timer:reset()
        
        -- define a closure
        local err, outputs
        feval = function(x)
          -- initialize gradient: important!
          model:zeroGradParameters()
          -- forward pass
          outputs = model:forward(inputs)
          err = loss:forward(outputs, targets)
          -- backward pass
          local gradOutputs = loss:backward(outputs, targets)
          model:backward(inputs, gradOutputs)
          return err, gradParameters
        end
        -- finally, update weight
        optim.sgd(feval, parameters, optimState)

        -- update related parameter obtained in one batch:
        -- include: batchNumber,loss_epoch,top1_epoch
        batchNumber = batchNumber + 1
        loss_epoch = loss_epoch + err
        -- compute top-1 error on training data as the prediction error 
        local top1 = 0
        do
          local _,prediction_sorted = outputs:float():sort(2, true) -- descending
          for i = 1,opt.batchSize do
            if prediction_sorted[i][1]==targetsCPU[i] then
                top1_epoch = top1_epoch + 1;
                top1 = top1 + 1
            end
          end
          top1 = top1 * 100 / opt.batchSize;
        end

        print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.0e DataLoadingTime %.3f'):format(
               epoch, batchNumber, opt.epochSize, timer:time().real, err, top1,
               optimState.learningRate, dataLoadingTime))
    end
  
    ----------------------------------------------------------------------
    -- top 1 error for one epoch: the averaged error of training samples
    top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
    -- loss for one epoch: the average of all mini-batch training loss
    loss_epoch = loss_epoch / opt.epochSize

    -- record averaged top 1 error and loss of training data in each epoch
    trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['avg loss (train set)'] = loss_epoch
    }
    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t',
                          epoch, tm:time().real, loss_epoch, top1_epoch))
    -- save model
    collectgarbage()
    -- clear the intermediate states in the model before saving to disk
    -- this saves lots of disk space
    torch.save(paths.concat(opt.save, 'alexnet_' .. epoch .. '.t7'), model)

    -- save optimState
    torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)

    -- return the learning rate
    return optimState.learningRate
end -- of train()
