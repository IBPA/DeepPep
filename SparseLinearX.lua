local SparseLinearX, parent = torch.class('nn.SparseLinearX', 'nn.Module')

function SparseLinearX:__init(inputSize, outputSize)
   parent.__init(self)

   -- Todo: remove artifitial initial weights and implement reset with random
   self.weight = torch.Tensor(outputSize, inputSize):fill(0)
   self.bias = torch.Tensor(outputSize):fill(0)
   self.gradWeight = torch.Tensor(outputSize, inputSize):fill(0)
   self.gradBias = torch.Tensor(outputSize):fill(0)

   self:reset()
end

function SparseLinearX:reset(stdv)
  if stdv then
    stdv = stdv * math.sqrt(3)
  else
    stdv = 1./math.sqrt(self.weight:size(2))
  end
  if nn.oldSeed then
    for i=1,self.weight:size(1) do
      self.weight:select(1, i):apply(function()
        return torch.uniform(-stdv, stdv)
      end)
    end
    if self.bias then
      for i=1,self.bias:nElement() do
        self.bias[i] = torch.uniform(-stdv, stdv)
      end
    end
  else
    self.weight:uniform(-stdv, stdv)
    if self.bias then self.bias:uniform(-stdv, stdv) end
  end
  return self
end


local function updateAddBuffer(self, nframe)
  self.addBuffer = self.addBuffer or torch.Tensor()
  if self.addBuffer:nElement() ~= nframe then
    self.addBuffer:resize(nframe):fill(1)
  end
end

function SparseLinearX:updateOutput(input)
  self.output = torch.zeros(input.nBatchSize, self.bias:size(1)) -- ToDo: just using dimention applicable to our particular problem

  if input.teOnes:nElement() ~= 0 then

    local nRecords = input.teOnes:size(1)
    for i=1, nRecords do
      local nRowId = input.teOnes[i][1]
      local nStartId = input.teOnes[i][2]
      local nLength = input.teOnes[i][3]

      self.output[nRowId]:add(self.weight:narrow(2, nStartId, nLength):sum(2))
    end

  end

--  self.output:add(self.bias:expandAs(self.output))
   updateAddBuffer(self, input.nBatchSize)
   self.output:addr(1, self.addBuffer, self.bias)

  return self.output
end

function SparseLinearX:updateGradInput(input, gradOutput)
  return nil
  -- maybe we dont need this for the first layer
  --[[
  if self.gradInput then

    local nElement = self.gradInput:nElement()
    self.gradInput:resize(input.nBatchSize, self.weight:size(2))
    if self.gradInput:nElement() ~= nElement then
      self.gradInput:zero()
    end
    
    self.gradInput:addmm(0, 1, gradOutput, self.weight)

    return self.gradInput
  end
  --]]
end

function SparseLinearX:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
--       self.gradWeight:addmm(scale, gradOutput:t(), input)
  local nRecords = input.teOnes:size(1)

  for i=1, nRecords do
    local nRowId = input.teOnes[i][1]
    local nStartId = input.teOnes[i][2]
    local nLength = input.teOnes[i][3]

    self.gradWeight:narrow(2, nStartId, nLength):add(
                                                     torch.mul(
                                                              gradOutput:narrow(1, nRowId, 1), scale):t():expand(self.gradWeight:size(1), nLength
                                                              )
                                                    )
  end

  updateAddBuffer(self, input.nBatchSize)
  self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)

end

function SparseLinearX:clearState()
  if self.addBuffer then self.addBuffer:set() end
  return parent.clearState(self)
end

