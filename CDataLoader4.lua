require('./CDataLoader.lua')
local csv = csv or require("csv")

CDataLoader4, CDataLoader4_parent = torch.class("CDataLoader4", "CDataLoader")

function CDataLoader4:__init(exprSettings, isUseDetectabilities)
  CDataLoader4_parent.__init(self, exprSettings)
	self.isUseDetectabilities = isUseDetectabilities or false
end

function CDataLoader4:pri_insertLineInfo(taIdx, strLine)
  local taSplit1 = strLine:split(':')
  local nRowId = tonumber(taSplit1[1])

  local taSplit2 = taSplit1[2]:split('|')
  for key, value in pairs(taSplit2) do
    local taSplit3 = value:split(',')
    local nLeft = tonumber(taSplit3[1])
    local nRight = tonumber(taSplit3[2])
    local taRecord = {nRowId + 1, nLeft + 1, nRight + 1 + 1 } -- adding 1 since indexes are 0 based, adding another "1" to nRight since it's the next position that it is cut.
    table.insert(taIdx, taRecord)
  end
end

function CDataLoader4:loadSparseInputSingleV2(strFilename, nWidth)
  local strFilename = string.format("%s/%s", self.exprSettings.strBaseDir, strFilename)
  local taRes = { nBatchSize = self.exprSettings.nRows, nWidth = nWidth }

  local file = io.open(strFilename, "r")
  local taIdx = {}

  for strLine in file:lines() do
    self:pri_insertLineInfo(taIdx, strLine)
  end
  file:close()
  
  taRes.teIdx = torch.LongTensor(taIdx)

  return taRes
end

function CDataLoader4:loadSparseBlockInput()

  self.taMetaInfo = self:loadSparseMetaInfo()
	local taInput = {nBatchSize = self.exprSettings.nRows,
									 taData = {}}
  for key, taFileInfo in pairs(self.taMetaInfo) do
--		local key = 33
--		local taFileInfo = self.taMetaInfo[key]
    local taSparseInput = self:loadSparseInputSingleV2(taFileInfo.strFilename, taFileInfo.nWidth)


    table.insert(taInput.taData, taSparseInput)
  end

	return taInput

end

----[[
function CDataLoader4:loadSparseMetaInfo()
  local strFilename = self.exprSettings.strFilenameMetaInfo
  local taLoadParams = {header=false, separator=","}
  local f = csv.open(strFilename, taLoadParams)

  local taMetaInfo= {}
  for fields in f:lines() do
    local taRow = { strFilename = fields[1], nWidth = fields[2] + 1 } --adding one, due to allow most right cleavage site
    table.insert(taMetaInfo, taRow)
  end

  return taMetaInfo
end
--]]


function CDataLoader4:loadTarget()

  local strFilename = self.exprSettings.strFilenameTarget
  local taLoadParams = {header=false, separator=","}
  local f = csv.open(strFilename, taLoadParams)

  local taRecords = {}
  for fields in f:lines() do
		if self.isUseDetectabilities then
	    table.insert(taRecords, fields[1]/fields[2])
		else
	    table.insert(taRecords, fields[1])
    end
  end

  local teRecords = torch.Tensor(taRecords)
  local teResult = torch.Tensor(teRecords:size(1), 1)
  teResult:select(2, 1):copy(teRecords)

  return teResult
end

function CDataLoader4:loadDetectabilities()
  local strFilename = self.exprSettings.strFilenameTarget
  local taLoadParams = {header=false, separator=","}
  local f = csv.open(strFilename, taLoadParams)

  local taRecords = {}
  for fields in f:lines() do
    table.insert(taRecords, fields[2])
  end

  local teRecords = torch.Tensor(taRecords)
  local teResult = torch.Tensor(teRecords:size(1), 1)
  teResult:select(2, 1):copy(teRecords)

  return teResult

end

