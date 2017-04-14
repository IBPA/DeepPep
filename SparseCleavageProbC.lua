-- input: SparseInput: format is still different from SparseBlocks: 
-- 	first column in teIdx is the row id
--	second column is the cleavage start
--	third column is the cleavage end
local SparseCleavageProbC, parent_SparseCleavageProbC = torch.class('nn.SparseCleavageProbC', 'nn.Module')

function SparseCleavageProbC:__init()

end

function SparseCleavageProbC:pri_ensureWeight(input)
	if self.weight ~= nil then
		return
	end

	local nColumns = table.getn(input.taData)
	self.weightMeta = torch.LongTensor(nColumns)

	-- find the size:
	local nTotalWeightSize = 0
	for i=1, nColumns do
		local taInputCurr = input.taData[i]
		local nWidth = taInputCurr.nWidth
		self.weightMeta[i] = nTotalWeightSize + 1
		nTotalWeightSize = nTotalWeightSize + nWidth
	end

	-- allocate:
	self.weight = torch.Tensor(nTotalWeightSize):fill(0.4)
--	self.weight = torch.rand(nTotalWeightSize)
	self.gradWeight = torch.zeros(nTotalWeightSize)

	self:reset()
end

function SparseCleavageProbC:pri_getSubW(i, teW)
	local nStart = self.weightMeta[i]
	local nLenght = -1  
	if i < self.weightMeta:size(1) then
		nLenght = self.weightMeta[i+1] - nStart
	else -- the last one, is different
		nLenght = teW:size(1) - nStart + 1
	end

	return teW:narrow(1, nStart, nLenght)
end

function SparseCleavageProbC:pri_getSubWeight(i)
	return self:pri_getSubW(i, self.weight)
end

function SparseCleavageProbC:pri_getSubGradWeight(i)
	return self:pri_getSubW(i, self.gradWeight)
end

function SparseCleavageProbC:pri_getSparseBlockRowIdx(teIdx)
	local taIdx = {}
	local idxLast = -1

	for i=1, teIdx:size(1) do
		local idxCurr = teIdx[i][1]
		if  idxCurr ~= idxLast then -- this is useful when one peptide has multiple matches (as in Yeast)
			table.insert(taIdx, idxCurr)
		end

		idxLast = idxCurr
	end

	local teRowIdx = torch.LongTensor(taIdx)
	return teRowIdx:resize(teRowIdx:size(1), 1)
end

function SparseCleavageProbC:pri_ensureOutput(input)
	if self.output ~= nil then
		return
	end

	self.output = { nBatchSize = input.nBatchSize, taData = {} }

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]

		local teRowIdx = self:pri_getSparseBlockRowIdx(taInputCurr.teIdx)

		taOutputCurr = { teValue = torch.zeros(teRowIdx:size(1), 1),
										 teRowIdx = teRowIdx }

		table.insert(self.output.taData, taOutputCurr)
	end

end

function SparseCleavageProbC:pri_g(dX)
	if dX < 0 then 
		return 0
	end

	if dX >1 then 
		return 1
	end
	
	return dX
end

function SparseCleavageProbC:pri_dg(dX)
	if dX < 0 or dX > 1 then 
		return 0
	end

	return 1
end


function SparseCleavageProbC:pri_updateOutput_column_row(teWeight, idxL, idxR, taOutput, iOutput)

	local dMul = 1 - self:pri_g(teWeight[idxL]) * self:pri_g(teWeight[idxR])

	--[[
	for i=idxL+1, idxR-1 do
		dMul = dMul * self:pri_g((1- teWeight[i]))
	end
	--]]

	local dOutputValue = taOutput.teValue[iOutput][1]
	taOutput.teValue[iOutput][1] = dOutputValue * dMul
end

function SparseCleavageProbC:pri_updateOutput_column(taInput, taOutput, teWeight)
	taOutput.teValue:fill(1)
	local nRows = taInput.teIdx:size(1)
	local iInputPrev = -1
	local iOutput = 1
	for i=1, nRows do
		if iInputPrev> 0 and taInput.teIdx[i][1] > taInput.teIdx[iInputPrev][1] then
			iOutput = iOutput + 1
		end

		local idxL = taInput.teIdx[i][2]
		local idxR = taInput.teIdx[i][3]
		
		self:pri_updateOutput_column_row(teWeight, idxL, idxR, taOutput, iOutput)

		iInputPrev = i
	end

end

function SparseCleavageProbC:updateOutput(input)
	self:pri_ensureWeight(input)
	self:pri_ensureOutput(input)

	local nColumns = table.getn(self.output.taData)

	for i=1, nColumns do
		self:pri_updateOutput_column(input.taData[i],
																 self.output.taData[i],
																 self:pri_getSubWeight(i))

	end

	return self.output
end

function SparseCleavageProbC:pri_getGradWeight_Edge(teF, teDf, idx)
	local dMul = -1
	local nWidth = teF:size(1)

		if idx == 1 then
			dMul = dMul * teDf[1] * teF[nWidth]
		else
			dMul = dMul * teDf[nWidth] * teF[1]
		end

	--[[
	for i=1, nWidth do
		if i == idx then
			dMul = dMul * teDf[i]
		else
			dMul = dMul * teF[i]
		end
	end
	--]]

	return dMul
end

function SparseCleavageProbC:pri_accGradWeight_column_row(teWeight, teGradWeight, idxL, idxR, dGradOutput, scale)
	local nWidth = idxR - idxL + 1
	local teF = torch.Tensor(nWidth)
	local teDf = torch.Tensor(nWidth)

	-- idxL
	teF[1] = self:pri_g(teWeight[idxL])
	teDf[1] = self:pri_dg(teWeight[idxL])

	-- idxR
	teF[nWidth] = self:pri_g(teWeight[idxR])
	teDf[nWidth] = self:pri_dg(teWeight[idxR])

	-- the rest
	--[[
	for i=idxL+1, idxR-1 do
		teF[idxR - i + 1] = 1 - self:pri_g(teWeight[i])
		teDf[idxR - i + 1] =  (-1) * self:pri_dg(teWeight[i]) 
	end
	--]]

--	for i=1, nWidth do
		local i = 1
		teGradWeight[idxL + i - 1] = teGradWeight[idxL + i - 1] + self:pri_getGradWeight_Edge(teF, teDf, i) * dGradOutput * scale
		i = nWidth
		teGradWeight[idxL + i - 1] = teGradWeight[idxL + i - 1] + self:pri_getGradWeight_Edge(teF, teDf, i) * dGradOutput * scale
--	end

	--]]

end

function SparseCleavageProbC:pri_accGradWeight_column(taInput, taGradOutput, teWeight, teGradWeight, scale)
	teGradWeight:fill(0)

	local nRows = taInput.teIdx:size(1)
	local iInputPrev = -1
	local iOutput = 1
	for i=1, nRows do
		if iInputPrev> 0 and taInput.teIdx[i][1] > taInput.teIdx[iInputPrev][1] then
			iOutput = iOutput + 1
		end

		local idxL = taInput.teIdx[i][2]
		local idxR = taInput.teIdx[i][3]
		local dCurrGradOutput = taGradOutput.teValue[iOutput][1]
		
		self:pri_accGradWeight_column_row(teWeight, teGradWeight, idxL, idxR, dCurrGradOutput, scale)

		iInputPrev = i
	end

end

function SparseCleavageProbC:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
	local nColumns = table.getn(input.taData)

	for i=1, nColumns do
		self:pri_accGradWeight_column(input.taData[i],
																	gradOutput.taData[i],
																	self:pri_getSubWeight(i),
																	self:pri_getSubGradWeight(i),
																	scale)
	end

end
