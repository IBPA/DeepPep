local SparseBlockDropout, parent = torch.class('nn.SparseBlockDropout', 'nn.Module')

function SparseBlockDropout:__init(p)
   self.p = p or 0.5
   self.train = true

   if self.p >= 1 or self.p < 0 then
      error('<Dropout> illegal percentage, must be 0 <= p < 1')
   end
end

function SparseBlockDropout:pri_ensureOutput(input)
	if self.output ~= nil then
		return
	end

	self.output = { nBatchSize = input.nBatchSize, taData = {} }
	 self.taNoise = {}

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]

		taOutputCurr = { teValue = torch.zeros(taInputCurr.teValue:size()),
										 teRowIdx = taInputCurr.teRowIdx,
									 	 teDefault = taInputCurr.teDefault }

		table.insert(self.output.taData, taOutputCurr)
		table.insert(self.taNoise, torch.zeros(taInputCurr.teValue:size()))
	end
end

function SparseBlockDropout:pri_updateOutput_column(taInput, taOutput, teNoise)
	local input = taInput.teValue
	local output = taOutput.teValue
	output:copy(input)

	if self.train then -- only mimicing "v2" of the Dropout
		teNoise:bernoulli(1- self.p)
		teNoise:div(1-self.p)
		output:cmul(teNoise)
	end

	taOutput.teDefault = taInput.teDefault

end

function SparseBlockDropout:updateOutput(input)
	assert(self.p>0, "only accept p>0")

	self:pri_ensureOutput(input)

	local nColumns = table.getn(self.output.taData)
	for i=1, nColumns do
		self:pri_updateOutput_column(input.taData[i], 
																 self.output.taData[i],
																 self.taNoise[i])
	end
	
	return self.output
end

function SparseBlockDropout:pri_ensureGradInput(input)
	if self.gradInput ~= nil then
		return
	end

	self.gradInput = { nBatchSize = input.nBatchSize, taData = {} }

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]

		taGradInputCurr = { teValue = torch.Tensor(taInputCurr.teValue:size()),
												teRowIdx = taInputCurr.teRowIdx }
		table.insert(self.gradInput.taData, taGradInputCurr)
	end
end

function SparseBlockDropout:pri_updateGradInput_column(taInput, taGradOutput, taGradInput, teNoise)
	local input = taInput.teValue
	local gradOutput = taGradOutput.teValue
	local gradInput = taGradInput.teValue

	gradInput:copy(gradOutput)

  if self.train then
		gradInput:cmul(teNoise)
	end

	taGradInput.teGradOutputSum = taGradOutput.teGradOutputSum
end

function SparseBlockDropout:updateGradInput(input, gradOutput)
	self:pri_ensureGradInput(input)

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		self:pri_updateGradInput_column(input.taData[i], 
																	  gradOutput.taData[i],
																		self.gradInput.taData[i],
																		self.taNoise[i])
	end
	
	return self.gradInput
end
