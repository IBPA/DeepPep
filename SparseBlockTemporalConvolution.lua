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

function SparseBlockTemporalConvolution:updateOutput(input)
	-- ensure output created
	self:pri_ensureOutput(input)

	-- update data
	local nColumns = table.getn(self.output.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]
		local taOutputCurr = self.output.taData[i]

	end
	
	return self.output
end

function SparseBlockTemporalConvolution:updateGradInput(input, gradOutput)
	assert(false, "not implemented")
end

function SparseBlockTemporalConvolution:accGradParameters(input, gradOutput, scale)
	assert(false, "not implemented")
	local scale = scale or 1
end

