require 'nn'
require 'sys'
require './SparseLinearX.lua'
local trainerPool = require('./deposTrainerPool.lua')

CExperiment = torch.class("CExperiment")

function CExperiment:__init(oDataLoader)
  self.oDataLoader = oDataLoader
end

function CExperiment:buildArch(dDropoutRate)
  self.taMetaInfo = self.oDataLoader:loadSparseMetaInfo()
  local dDropoutRate = dDropoutRate or 0.7

  -- 1) Build the layer0
  local nUnitWidthLayer0 = 1
  local mLayer0 = nn.ParallelTable()
  local nParallels = 0
  for key, taFileInfo in pairs(self.taMetaInfo) do
    local mSeq = nn.Sequential()
      mSeq:add(nn.SparseLinearX(taFileInfo.nWidth, nUnitWidthLayer0 ))
      mSeq:add(nn.Dropout(dDropoutRate))
    
    mLayer0:add(mSeq)
    nParallels = nParallels + 1
  end

  -- 2) Build the rest of the FNN:
  self.mNet = nn.Sequential()
  self.mNet:add(mLayer0)
  self.mNet:add(nn.JoinTable(2))

  local mSeq = nn.Sequential()
    mSeq:add(nn.Linear(nParallels*nUnitWidthLayer0, 1))
    mSeq:add(nn.Sigmoid())
--    mSeq:add(nn.Dropout(0.70))
  self.mNet:add(mSeq)
--  self.mNet:add(nn.Linear(nParallels, 1))
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
	self.mNet:training()
  sys.tic()
  local dTrainErr = trainerPool.trainSparseInputNet(self.mNet, taInput, teTarget, nIteration, "CG", true)
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

function CExperiment:pri_getLayers() -- ToDo: this would change for other architectures
  local mFirst = nn.Sequential()
  mFirst:add(self.mNet.modules[1])
  mFirst:add(self.mNet.modules[2])

  local mRest = self.mNet.modules[3]

  return {mFirst = mFirst, mRest = mRest}
end

function CExperiment:pri_getEmptyInput(nRows)
  return { nBatchSize = nRows, teOnes = torch.LongTensor() }
end

function CExperiment:getConfidenceOne(teOutputAll, teOutputFirst, taMNetLayers, nProtId, taProtData)
  -- 1) save the nProtId column we are about the replace
  local teProtOrig = teOutputFirst:narrow(2, nProtId, 1):clone()

  -- 2) replace the protein column with predicted value for empty info  for nProtId
  local mProt = taMNetLayers.mFirst.modules[1].modules[nProtId]
  local teEmpty = self:pri_getEmptyInput(teOutputFirst:size(1))
  local teProt = mProt:forward(teEmpty):clone()
  teOutputFirst:narrow(2, nProtId, 1):copy(teProt)

  -- 3) calculate the final prediction
  local teOutputAllNew = taMNetLayers.mRest:forward(teOutputFirst):clone()

  -- 4) calculate the difference
  local teOutputResidual = torch.add(torch.mul(teOutputAllNew, -1),
                                     teOutputAll):abs():squeeze()
                                      
  -- 5) replace the orig column
  teOutputFirst:narrow(2, nProtId, 1):copy(teProtOrig)

  -- 6) calculate prot_pepdide confidences

  -- 6.1) calculate the counts
  local taProtPepCount = {}
  local nRows = taProtData.teOnes:size(1)
  for i=1, nRows do
    local nIdx = taProtData.teOnes[i][1]
    local nCount = taProtData.teOnes[i][3]

    local nExistingCount = taProtPepCount[nIdx] or 0
    taProtPepCount[nIdx] = nExistingCount + nCount
  end

  -- 6.2) calculate the confidences
  local dSum = 0
  for key, value in pairs(taProtPepCount) do
    local dConfCurr = teOutputResidual[key]/value
    dSum = dSum + dConfCurr
  end

  return dSum/taProtData.nBatchSize
end

function CExperiment:getConfidenceRange(nStart, nEnd)
  sys.tic()

  local nStart = nStart or 1
  local nEnd = nEnd or #self.taMetaInfo

--	self.mNet:evaluate()
  local teOutputAll = self.mNet:forward(self.taInput):clone()
  local taMNetLayers = self:pri_getLayers()
  local teOutputFirst = taMNetLayers.mFirst:forward(self.taInput):clone()

  local taProtInfo = {}
  for i=1, nEnd do
    local taFileInfo = self.taMetaInfo[i]
    local dConf = self:getConfidenceOne(teOutputAll, teOutputFirst, taMNetLayers, i, self.taInput[i])

    local strProtFilename = taFileInfo.strFilename
    local strProtName = strProtFilename:sub(1, strProtFilename:len() -4)  -- remove the ".txt" from the end
    local taRow = { strProtName, dConf }
    table.insert(taProtInfo, taRow)
  end

  print("confidence total elapsed time(s):" .. sys.toc())
  return taProtInfo
end

function CExperiment:saveResult(taProtInfo)
  local taRef = self.oDataLoader:loadProtRef()

  for key, value in pairs(taProtInfo) do
    local strProtName = value[1]

    if taRef[strProtName] == nil then
      table.insert(value, 0)
    else
      table.insert(value, 1)
    end
  end

  self.oDataLoader:saveProtInfo(taProtInfo)
end

