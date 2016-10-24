local SparseBlockTemporalConvolution, parent = torch.class('nn.SparseBlockTemporalConvolution', 'nn.Module')

-- Module description:
-- strict limitted support assuming: a) parallel inputs with shared default value, b) fullbatch d) default input is allways zero  which enables for sparse backpropagaion. e) no bias in this layer
--ToDo: see input.THNN.TemporalConvolution_updateOutput

function SparseBlockTemporalConvolution:__init(inputFrameSize, outputFrameSize, kW, dW)
   dW = dW or 1

   self.inputFrameSize = inputFrameSize
   self.outputFrameSize = outputFrameSize
   self.kW = kW
   self.dW = dW

   self.weight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.gradWeight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
	 self.dummyGradBias = torch.zeros(outputFrameSize)
   
   self:reset()
end

function SparseBlockTemporalConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.inputFrameSize)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
   end
end

function SparseBlockTemporalConvolution:pri_ensureOutput(input)
	if self.output ~= nil then
		return
	end


	self.output = { nBatchSize = input.nBatchSize, taData = {} }

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]
		local nRows = taInputCurr.teValue:size(1)
  	local nWidth =  (taInputCurr.teValue:size(2) - self.kW) / self.dW + 1;

		taOutputCurr = { teValue = torch.zeros(nRows, nWidth, self.outputFrameSize),
													 teRowIdx = taInputCurr.teRowIdx }

		table.insert(self.output.taData, taOutputCurr)
	end

end

function SparseBlockTemporalConvolution:pri_ensureGradInput(input)
	if self.gradInput ~= nil then
		return
	end

	self.gradInput = { nBatchSize = input.nBatchSize, taData = {} }

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]

		taGradInputCurr = { teValue = torch.zeros( taInputCurr.teValue:size() ),
												teRowIdx = taInputCurr.teRowIdx }
		table.insert(self.gradInput.taData, taGradInputCurr)
	end
end

function SparseBlockTemporalConvolution:pri_updateOutput_column(taInput, taOutput)
	local nOutputWidth =  taOutput.teValue:size(2)
	local nInputWidth = taInput.teValue:size(2)
	local nRows = taOutput.teValue:size(1)

	-- ouch
	for k=1, nOutputWidth do
		local teOutputWindow = taOutput.teValue:select(2, k)

		local teInput2d = taInput.teValue:view(nRows, self.inputFrameSize * nInputWidth)
		local nInputOffset = (self.dW * (k-1) + 1) * self.inputFrameSize -1
		local teInputWindow = teInput2d:narrow(2, nInputOffset, self.inputFrameSize * self.kW)

		teOutputWindow:addmm(teInputWindow, self.weight:t())
	end
	
end

function SparseBlockTemporalConvolution:updateOutput(input)
	-- ensure output created
	self:pri_ensureOutput(input)

	-- update data
	local nColumns = table.getn(self.output.taData)
	for i=1, nColumns do
		self:pri_updateOutput_column(input.taData[i], 
																 self.output.taData[i])
	end
	
	return self.output
end

function SparseBlockTemporalConvolution:pri_updateGradInput_column(taInput, taGradOutput, taGradInput)
	-- Note: mimicing the logic of nDimension==3 in "TemporalConvolution_updateGradInput"
	local input = taInput.teValue
	local gradOutput = taGradOutput.teValue
	local gradInput = taGradInput.teValue

  input.THNN.TemporalConvolution_updateGradInput(
	  input:cdata(), gradOutput:cdata(),
	  gradInput:cdata(), self.weight:cdata(),
	  self.kW, self.dW
    )
end

function SparseBlockTemporalConvolution:updateGradInput(input, gradOutput)
	-- ensure GradInput created
	self:pri_ensureGradInput(input)

	-- update data
	local nColumns = table.getn(self.gradInput.taData)
	for i=1, nColumns do
		self:pri_updateGradInput_column(input.taData[i],
																		gradOutput.taData[i],
																		self.gradInput.taData[i])
	end
	
	return self.gradInput
end

function SparseBlockTemporalConvolution:pri_accGradParameters_column(taInput, taGradOutput, scale)
	local input = taInput.teValue
	local gradOutput = taGradOutput.teValue

  input.THNN.TemporalConvolution_accGradParameters(
       input:cdata(), gradOutput:cdata(),
       self.gradWeight:cdata(), self.dummyGradBias:cdata(),
       self.kW, self.dW, scale
   )

end

function SparseBlockTemporalConvolution:accGradParameters(input, gradOutput, scale)
	local scale = scale or 1

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		self:pri_accGradParameters_column(input.taData[i],
																			gradOutput.taData[i],
																			scale)
	end

end

