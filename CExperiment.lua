require 'nn'
require 'sys'
require './SparseLinearX.lua'
local trainerPool = require('./deposTrainerPool.lua')

CExperiment = torch.class("CExperiment")

function CExperiment:__init(oDataLoader)
  self.oDataLoader = oDataLoader
end

function CExperiment:buildArch()
  self.taMetaInfo = self.oDataLoader:loadSparseMetaInfo()

  -- 1) Build the layer0
  local nUnitWidthLayer0 = 1
  local mLayer0 = nn.ParallelTable()
  local nParallels = 0
  for key, taFileInfo in pairs(self.taMetaInfo) do
    mLayer0:add(nn.SparseLinearX(taFileInfo.nWidth, nUnitWidthLayer0 ))
    nParallels = nParallels + 1
  end

  -- 2) Build the rest of the FNN:
  self.mNet = nn.Sequential()
  self.mNet:add(mLayer0)
  self.mNet:add(nn.JoinTable(2))
  self.mNet:add(nn.Linear(nParallels, 1))
end

function CExperiment:train(nIteration, isKeepData)
  local nIteration = nIteration or 20

  -- 1) load input
  local taInput = {}
  for key, taFileInfo in pairs(self.taMetaInfo) do
    local taOneInput = self.oDataLoader:loadSparseInputSingleV2(taFileInfo.strFilename)
    table.insert(taInput, taOneInput)
  end

  -- 2) Load the Target
  local teTarget = self.oDataLoader:loadTarget()

  -- 3) Train
  sys.tic()
  local dTrainErr = trainerPool.trainSparseInputNet(self.mNet, taInput, teTarget, nIteration)
  print("\ntraining error:" .. dTrainErr) 
  print("training elapsed time(s):" .. sys.toc())

  -- 4) isKeepData
  if isKeepData then
    self.taInput = taInput
  end

end

function CExperiment:save(strFilePath)
  torch.save(strFilePath, self)
end

function CExperiment.loadFromFile(strFilePath)
  local oExperiment = torch.load(strFilePath)

  return oExperiment
end

function CExperiment:predict()

  local nId = 0
  for key, taFileInfo in pairs(self.taMetaInfo) do
    nId = nId + 1
    print(nId)
  end

end


