require 'nn'
require 'sys'
require '../SparseLinearX.lua'

local sparseLinearX_test = {}

function wrapUT(fuUT, strName)
  fuUT()
  print("PASS " .. strName)
end

function getTestInputCSparse()
  return { nBatchSize = 10,
                          --rowid, startId, length
           teOnes = torch.LongTensor({{1, 3, 5},
                                      {2, 1, 3},
                                      {2, 7, 3},
                                      {8, 8, 2}} )}
end

function getTestInput()
  local teRes = torch.zeros(10, 10)
  teRes:narrow(1, 1, 1):narrow(2, 3, 5):fill(1)
  teRes:narrow(1, 2, 1):narrow(2, 1, 3):fill(1)
  teRes:narrow(1, 2, 1):narrow(2, 7, 3):fill(1)
  teRes:narrow(1, 8, 1):narrow(2, 8, 2):fill(1)

  return teRes
end

function sparseLinearX_test.updateOutput()
  
  local nInputWidth = 10

  -- sparse
  local taInputCSparse = getTestInputCSparse()
  torch.manualSeed(1)
  local mSparseLinearX = nn.SparseLinearX(nInputWidth, 1)

  sys.tic()
  local teOutput = mSparseLinearX:forward(taInputCSparse)
  print("elapsed:" .. sys.toc())

  -- non-sparse
  local teInput = getTestInput()
  torch.manualSeed(1)
  local mLinear = nn.Linear(nInputWidth, 1)

  sys.tic()
  local teOutput2 = mLinear:forward(teInput)
  print("elapsed:" .. sys.toc())

  -- compare
  local dDiff = (teOutput2-teOutput):sum()
  assert(dDiff == 0, "not matching!")


end

function sparseLinearX_test.updateGradInput()
  local nInputWidth = 10
  local nOutputWidth = 1
  local gradOutput = torch.Tensor({{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}})

  -- sparse
  local taInputCSparse = getTestInputCSparse()
  torch.manualSeed(1)
  local mSparseLinearX = nn.SparseLinearX(nInputWidth, nOutputWidth)
  local teGradInput = mSparseLinearX:updateGradInput(taInputCSparse, gradOutput)

  -- non-sparse
  local teInput = getTestInput()
  torch.manualSeed(1)
  local mLinear = nn.Linear(nInputWidth, 1)
  local teGradInput2 = mLinear:updateGradInput(teInput, gradOutput)

  -- compare
  local dDiff = (teGradInput2-teGradInput):sum()
  assert(dDiff == 0, "not matching!")
end

function sparseLinearX_test.accGradParameters()
  local nInputWidth = 10
  local nOutputWidth = 1
  local gradOutput = torch.Tensor({{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}})

  -- sparse
  local taInputCSparse = getTestInputCSparse()
  torch.manualSeed(1)
  local mSparseLinearX = nn.SparseLinearX(nInputWidth, nOutputWidth)
  local teOutput = mSparseLinearX:forward(taInputCSparse)
  mSparseLinearX:accGradParameters(taInputCSparse, gradOutput)

  -- non sparse
  local teInput = getTestInput()
  torch.manualSeed(1)
  local mLinear = nn.Linear(nInputWidth, 1)
  local teOutput2 = mLinear:forward(teInput)
  mLinear:accGradParameters(teInput, gradOutput)

  -- compare
  local dDiff = (mLinear.gradWeight - mSparseLinearX.gradWeight):sum()
  assert(dDiff == 0, "gradWeight not matching!")

  dDiff = (mLinear.gradBias - mSparseLinearX.gradBias):sum()
  assert(dDiff == 0, "gradBias not matching!")
end


wrapUT(sparseLinearX_test.updateOutput, "updateOutput")
wrapUT(sparseLinearX_test.updateGradInput, "updateGradInput")
wrapUT(sparseLinearX_test.accGradParameters, "accGradParameters")
