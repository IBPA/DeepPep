--[[ Description:
      Enables file operations including loading data in the 
        SparseBlock format to be used for SpaseBlock architectures.]]

local csv = csv or require("csv")
CData = torch.class("CData")

function CData:__init(exprSettings)
  self.exprSettings = exprSettings
end

function CData:pri_insertLineInfo(taIdx, strLine)
  local taSplit1 = strLine:split(':')
  local nRowId = tonumber(taSplit1[1])

  local taSplit2 = taSplit1[2]:split('|')
  for key, value in pairs(taSplit2) do
    local taSplit3 = value:split(',')
    local nStartId = tonumber(taSplit3[1])
    local nLength = tonumber(taSplit3[2])
    local taRecord = {nRowId + 1, nStartId + 1, nLength } -- adding 1 since indexes are 0 based
    table.insert(taIdx, taRecord)
  end
end

function CData:pri_loadSparseInputSingle(strFilename)
  local strFilename = string.format("%s/%s", self.exprSettings.strBaseDir, strFilename)
  local taRes = { nBatchSize = self.exprSettings.nRows }

  local file = io.open(strFilename, "r")
  local taIdx = {}

  for strLine in file:lines() do
    self:pri_insertLineInfo(taIdx, strLine)
  end
  file:close()
  
  taRes.teOnes = torch.LongTensor(taIdx)

  return taRes
end

function pri_getBlockRowIdx(taSparseInput)
    -- capture the unique ids
    local taUnique = {}
    for i=1, taSparseInput.teOnes:size(1) do
      taUnique[taSparseInput.teOnes[i][1]] = true
    end

    -- create block ids
    local taBlockRowIdx = {}
    for key, value in pairs(taUnique) do
      table.insert(taBlockRowIdx, key)
    end
    local teBlockRowIdx = torch.LongTensor(taBlockRowIdx)

    -- create reverse map (input to block)
    local taBlockRowReverseMap = {}
    for i=1, teBlockRowIdx:size(1) do
        taBlockRowReverseMap[teBlockRowIdx[i]] = i
    end

    teBlockRowIdx:resize(teBlockRowIdx:size(1), 1) -- resize to add second dimension (necessary in SparseBlockToDenseLinear)
    return teBlockRowIdx, taBlockRowReverseMap
end

function CData:pri_sparseToSparseBlock(taSparseInput, nWidth)

  local taRes = { teRowIdx = nil,
                  teValue = nil}

  local taBlockRowReverseMap = nil
  taRes.teRowIdx, taBlockRowReverseMap = pri_getBlockRowIdx(taSparseInput)
  local nBlocks = taRes.teRowIdx:size(1)
  taRes.teValue =  torch.Tensor(nBlocks, nWidth, 1):fill(0)

  local teOnes = taSparseInput.teOnes
  for i=1, teOnes:size(1) do
      local nRowId = taBlockRowReverseMap[teOnes[i][1]]
      local nStartId = teOnes[i][2]
      local nLength = teOnes[i][3]
      taRes.teValue[nRowId]:narrow(1, nStartId, nLength):fill(1)
  end

  return taRes
end

function CData:loadSparseBlockInput()

  self.taMetaInfo = self:loadSparseMetaInfo()
  local taInput = {nBatchSize = self.exprSettings.nRows,
                   taData = {}}
  for key, taFileInfo in pairs(self.taMetaInfo) do
    local taSparseInput = self:pri_loadSparseInputSingle(taFileInfo.strFilename)
    local taSparseBlockInput = self:pri_sparseToSparseBlock(taSparseInput, taFileInfo.nWidth)

    table.insert(taInput.taData, taSparseBlockInput)
  end

  return taInput

end

function CData:loadSparseMetaInfo()
  local strFilename = self.exprSettings.strFilenameMetaInfo
  local taLoadParams = {header=false, separator=","}
  local f = csv.open(strFilename, taLoadParams)

  local taMetaInfo= {}
  for fields in f:lines() do
    local taRow = { strFilename = fields[1], nWidth = fields[2] }
    table.insert(taMetaInfo, taRow)
  end

  return taMetaInfo
end

function CData:loadTarget()
  local strFilename = self.exprSettings.strFilenameTarget
  local taLoadParams = {header=false, separator=","}
  local f = csv.open(strFilename, taLoadParams)

  local taRecords = {}
  for fields in f:lines() do
    table.insert(taRecords, fields[1])
  end

  local teRecords = torch.Tensor(taRecords)
  local teResult = torch.Tensor(teRecords:size(1), 1)
  teResult:select(2, 1):copy(teRecords)

  return teResult
end

function CData:saveProtInfo(taProtInfo)
  local file = io.open(self.exprSettings.strFilenameProtInfo, "w")
  for key, value in pairs(taProtInfo) do
    file:write(string.format("%s,%.12f\n", value[1], value[2]))
  end

  file:close()
end

function CData:saveModelParams(teParams)
  local file = io.open(self.exprSettings.strFilenameExprParams , "w")

  for i=1, teParams:size(1) do
    file:write(string.format("%f\n", teParams[i]))
  end

  file:close()
end

function CData:saveDescription(strDescription)
  local file = io.open(self.exprSettings.strFilenameExprDescription, "w")
  file:write(strDescription)
  file:close()
end
