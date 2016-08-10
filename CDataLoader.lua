local csv = csv or require("csv")

CDataLoader = torch.class("CDataLoader")

function CDataLoader:__init(exprSettings)
  self.exprSettings = exprSettings
end

function CDataLoader:loadSparseInput()
  local strFilename = self.exprSettings.strFilenameInputSparse
  local taLoadParams = {header=false, separator=","}
  local f = csv.open(strFilename, taLoadParams)

  local dValueNonsparse = 1
  local taRecords = {}
  for fields in f:lines() do

    local teIdx = torch.Tensor(fields)
    local teCurr = torch.Tensor(teIdx:size(1), 2)
    teCurr:select(2, 1):copy(teIdx)
    teCurr:select(2, 2):fill(dValueNonsparse)
    table.insert(taRecords, teCurr)
  end

  return taRecords
end

function CDataLoader:pri_getLineInfo(file)
  local lineId = file:read("*number") 
  if lineId == nil then
    return nil
  end
  lineId = lineId + 1 -- since it's zero based!

  file:seek("cur", 1) -- skip ":"
  local strLine = file:read("*line")
  local taIdx = strLine:split(',')
  local teIdx = torch.Tensor(table.getn(taIdx), 2)
  teIdx:select(2, 1):copy(torch.add(torch.Tensor(taIdx), 1)) -- add 1 to indexes since it's 0 based
  teIdx:select(2, 2):fill(1)

  return lineId, teIdx:clone()
end

function CDataLoader:pri_insertLineInfo(taIdx, strLine)
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

function CDataLoader:loadSparseInputSingleV2(strFilename)
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

function CDataLoader:loadSparseInputSingle(strFilename)
  local strFilename = string.format("%s/%s", self.exprSettings.strBaseDir, strFilename)

  local taRows = {}
  local file = io.open(strFilename, "r")

  local id = -1; local teIdx = nil
  taRows[id] = teIdx
  while id ~= nil do
    id, teIdx = self:pri_getLineInfo(file)
    if id ~= nil then
      taRows[id] = teIdx
    end
  end
  file:close()


  -- Fill in the empty lines
--  --[[
  local taRes = {}
  for i=1,self.exprSettings.nRows do
    if taRows[i] == nil then
      --todo: undo this
      taRes[i] = torch.Tensor({{1, 0}})
    else
      taRes[i] = taRows[i]
    end
  end
  --]]


  return taRes --taRows --taRes
end

function CDataLoader:loadSparseMetaInfo()
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

function CDataLoader:loadTarget()
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

function CDataLoader:loadProtRef()
  local strFilename = self.exprSettings.strFilenameProtRef
  local taLoadParams = {header=false, separator=","}
  local f = csv.open(strFilename, taLoadParams)

  local taRecords = {}
  for fields in f:lines() do
    taRecords[fields[1]] = 1
  end

  return taRecords
end


