require('./CDataLoader.lua')
CDataLoader4, CDataLoader4_parent = torch.class("CDataLoader4", "CDataLoader")

function CDataLoader4:__init(exprSettings)
  CDataLoader4_parent.__init(self, exprSettings)
end

function CDataLoader4:pri_insertLineInfo(taIdx, strLine)
  local taSplit1 = strLine:split(':')
  local nRowId = tonumber(taSplit1[1])

  local taSplit2 = taSplit1[2]:split('|')
  for key, value in pairs(taSplit2) do
    local taSplit3 = value:split(',')
    local nLeft = tonumber(taSplit3[1])
    local nRight = tonumber(taSplit3[2])
    local taRecord = {nRowId + 1, nLeft + 1, nRight + 1 } -- adding 1 since indexes are 0 based
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

function CDataLoader:loadSparseBlockInput()

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

