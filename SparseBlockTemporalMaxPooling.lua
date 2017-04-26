local SparseBlockTemporalMaxPooling, parent = torch.class('nn.SparseBlockTemporalMaxPooling', 'nn.Module')

function SparseBlockTemporalMaxPooling:__init(kW, dW, isRelax, isBP)
   dW = dW or kW

   self.kW = kW
   self.dW = dW
	 self.isRelax = isRelax or false
	 self.isBP = isBP or true
end

function SparseBlockTemporalMaxPooling:pri_ensureOutput(input)
	if self.output ~= nil then
		return
	end

	self.output = { nBatchSize = input.nBatchSize, taData = {} }

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]

		taOutputCurr = { teValue = torch.Tensor(),
										 teRowIdx = taInputCurr.teRowIdx }

		table.insert(self.output.taData, taOutputCurr)
	end
end

function SparseBlockTemporalMaxPooling:pri_ensureTaIndices(input)
	if self.taIndices ~= nil then
		return
	end

	self.taIndices = {}

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]
		table.insert(self.taIndices, torch.Tensor()) -- Important Note: due to a bug in torch+nn, this is reverted back (instead of torch.LongTensor()). For this to work, need to be using torch @ commit: 1e5a315d03c91286d859512574d3b0b25e12d512, and nn @ commit: 1443cd7c2becd793b3d954144dcf4a1bf9947771
	end
end

function SparseBlockTemporalMaxPooling:pri_updateOutput_column(taInput, taOutput, teIndices)
	local input = taInput.teValue
	local output = taOutput.teValue

	local kW = self.kW
	if self.isRelax  then
		kW = math.min(input:size(2), kW)
	end

   input.THNN.TemporalMaxPooling_updateOutput(
       input:cdata(), output:cdata(),
       teIndices:cdata(), kW, self.dW
   )

end

function SparseBlockTemporalMaxPooling:updateOutput(input)
	self:pri_ensureTaIndices(input)
	self:pri_ensureOutput(input)

	local nColumns = table.getn(self.output.taData)
	for i=1, nColumns do
		self:pri_updateOutput_column(input.taData[i], 
																 self.output.taData[i],
																 self.taIndices[i])
	end

	return self.output
end

function SparseBlockTemporalMaxPooling:pri_ensureGradInput(input)
	if self.gradInput ~= nil then
		return
	end

	self.gradInput = { nBatchSize = input.nBatchSize, taData = {} }

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]

		taGradInputCurr = { teValue = torch.Tensor(),
												teRowIdx = taInputCurr.teRowIdx }
		table.insert(self.gradInput.taData, taGradInputCurr)
	end
end

function SparseBlockTemporalMaxPooling:pri_updateGradInput_column(taInput, taGradOutput, taGradInput, teIndices)
	local input = taInput.teValue
	local gradOutput = taGradOutput.teValue
	local gradInput = taGradInput.teValue

	local kW = self.kW
	if self.isRelax  then
		kW = math.min(input:size(2), kW)
	end


	input.THNN.TemporalMaxPooling_updateGradInput(
	    input:cdata(), gradOutput:cdata(),
	    gradInput:cdata(), teIndices:cdata(),
	    kW, self.dW
	)

end


function SparseBlockTemporalMaxPooling:updateGradInput(input, gradOutput)
	if not self.isBP then
		return nil
	end

	self:pri_ensureGradInput(input)

	local nColumns = table.getn(self.gradInput.taData)
	for i=1, nColumns do
		self:pri_updateGradInput_column(input.taData[i],
																		gradOutput.taData[i],
																		self.gradInput.taData[i],
																		self.taIndices[i])
	end
	
	return self.gradInput

end

function SparseBlockTemporalMaxPooling:empty()
   assert(false, "empty not implemented!")
end

