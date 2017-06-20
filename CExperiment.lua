--[[ Description:
      Coordinates details related to training and prediction of DeepPep.]]

require 'nn'
require 'sys'
require 'sparsenn'

local trainerLib = require('./trainerLib.lua')
CExperiment = torch.class("CExperiment")

-- Input(oData): data object (CData).
-- Input(fuArchBuilder): a function that generates the nn architecture.
function CExperiment:__init(oData, fuArchBuilder)
  self.oData = oData
  self.fuArchBuilder = fuArchBuilder
end

function CExperiment:buildArch(taArchParams)
  self.taMetaInfo = self.oData:loadSparseMetaInfo()
  self.fuArchBuilder(self, taArchParams)
end

-- Description: After loading the data, performs a single forward and backward pass on nn.
function CExperiment:roundTrip()
  -- 1) load input
  local taInput = self.oData:loadSparseBlockInput(self.taMetaInfo)

  -- 2) Load the Target
  local teTarget = self.oData:loadTarget()

  -- ToDo: now doing simple single forward, backward for until fully implemented
  sys.tic()
  local teOutput = self.mNet:forward(taInput)
  print("forward elapsed time(s):" .. sys.toc())

  sys.tic()
  local taGradInput = self.mNet:backward(taInput, teOutput, 0)
  print("backward elapsed time(s):" .. sys.toc())
end
 
function CExperiment:train(taTrainParam, nIteration, strOptimMethod, isEarlyStop, dStopError)
  nIteration = nIteration or 20
  strOptimMethod = strOptimMethod or "SGD"
  isEarlyStop = isEarlyStop or false
  dStopError = dStopError or 0.0001

  -- 1) load input
  local taInput = self.oData:loadSparseBlockInput(self.taMetaInfo)

  -- 2) Load the Target
  local teTarget = self.oData:loadTarget()

  -- 3) Train
  self.mNet:training()
  sys.tic()
  local dTrainErr = trainerLib.trainSparseInputNet(self.mNet, taInput, teTarget, nIteration, strOptimMethod, isEarlyStop, dStopError, taTrainParam)
  print("\ntraining error:" .. dTrainErr) 
  print("training elapsed time(s):" .. sys.toc())
end

-- Input(strFilePath): filepath to save this CExperiment object.
function CExperiment:save(strFilePath)
  torch.save(strFilePath, self)
end

-- Input(strFilePath): filepath to load a CExperiment from.
function CExperiment.loadFromFile(strFilePath)
  local oExperiment = torch.load(strFilePath)

  return oExperiment
end

-- Input(teOutputAll): precalucated output of nn for all proteins.
-- Input(taOutputFirst): precalculated output of first part of nn (before joining proteins).
-- Input(taInput): taInput for the given protein
-- Input(nProtId): proein id
-- Return: calculated confidence from deepPep for the given protein.
function CExperiment:pri_getConfidenceOneVFast(teOutputAll, taOutputFirst, taInput, nProtId)
  -- 1) calculate the columns contribution in in final prediction
  self.mRest:pub_setColIds(nProtId) -- ensure calculates are done only for the given protein.
  local teOutputResidual = self.mRest:forward(taOutputFirst):clone():abs():squeeze()
  
  -- 2) calculate prot_pepdide confidences
  local dSum = self:pri_getNormalizedResidualSum(taInput, teOutputResidual)
  return dSum /teOutputAll:size(1)
end

-- Input(taInput): protein input
-- Input(teOutputResidual): protein's contribution to predicted probabilities
-- Return: normalized residual sum of preotein's contribution to predicted probabilities.
function CExperiment:pri_getNormalizedResidualSum(taInput, teOutputResidual)
  local dSum = 0
  local nMatchingPeptides = taInput.teRowIdx:size(1)
  for i=1, nMatchingPeptides do
    local n_ji = taInput.teValue[i]:sum()
    local nIdx = taInput.teRowIdx[i]:squeeze()
    local c_ij = teOutputResidual[nIdx]/n_ji
    dSum = dSum + c_ij
  end

  return dSum
end

-- Return: table containing calculated confidence levels for each protein.
function CExperiment:getConfidenceAll()
  sys.tic()
  self.mNet:evaluate()
  
  local nEnd = #self.taMetaInfo
  local taInput = self.oData:loadSparseBlockInput(self.taMetaInfo)
  
  local teOutputAll = self.mNet:forward(taInput):clone()

  local taOutputFirst = self.mFirst:forward(taInput) 

  local taProtInfo = {}
  for i=1, nEnd do
    local dConf = self:pri_getConfidenceOneVFast(teOutputAll, taOutputFirst, taInput.taData[i], i)
    local strProtFilename = self.taMetaInfo[i].strFilename
    local strProtName = strProtFilename:sub(1, strProtFilename:len() -4)  -- remove the ".txt" from the end
    local taRow = { strProtName, dConf }
    table.insert(taProtInfo, taRow)
  end

  print("confidence total elapsed time(s):" .. sys.toc())
  return taProtInfo
end

-- Input(taProtInfo): table containing protein names and their predicted confidences.
function CExperiment:saveResult(taProtInfo)
  self.oData:saveProtInfo(taProtInfo)
  self.oData:saveModelParams(self.mNet:getParameters())

  if self.strArchDescription ~= nil then
    self.oData:saveDescription(self.strArchDescription)
  end
end

