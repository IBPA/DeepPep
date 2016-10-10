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

		return teBlockRowIdx, taBlockRowReverseMap
end

function CDataLoader:pri_sparseToBlockSparse(taSparseInput, nWidth)

  local taRes = { 
									dDefault = torch.Tensor({1, 1, 1}):fill(0),
									teRowIdx = nil,
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

function CDataLoader:loadBlockSparseInput()

  self.taMetaInfo = self:loadSparseMetaInfo()
	local taInput = {nBatchSize = self.exprSettings.nRows,
									 taData = {}}
  for key, taFileInfo in pairs(self.taMetaInfo) do
    local taSparseInput = self:loadSparseInputSingleV2(taFileInfo.strFilename)
		local taBlockSparseInput = self:pri_sparseToBlockSparse(taSparseInput, taFileInfo.nWidth)

    table.insert(taInput.taData, taBlockSparseInput)
  end

	return taInput

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

function CDataLoader:saveProtInfo(taProtInfo)
  local file = io.open(self.exprSettings.strFilenameProtInfo, "w")
  for key, value in pairs(taProtInfo) do
    file:write(string.format("%s,%.12f,%d\n", value[1], value[2], value[3]))
  end

  file:close()
end

