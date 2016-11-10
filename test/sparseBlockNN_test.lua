--[[
Plan:
	1) load sparse data into sparseBlockTensor
	2) implement ReLU sparseBlockTensor in, sparseBlockTensor out
	3) implement TemporalConv sparseBlockTensor in, sparseBlockTensor out
	4) implement TemporalConv with shared weight + async queue

--]]

require 'nn'
require('../SparseBlockReLU.lua')
require '../SparseBlockTemporalConvolution.lua'
require '../SparseBlockTemporalMaxPooling.lua'
require '../SparseBlockFlattenDim3.lua'
require '../SparseBlockLinear.lua'
require '../SparseBlockToDenseLinear.lua'
require '../SparseBlockDropout.lua'

local deposUtil = deposUtil or require('../deposUtil.lua')

local sparseBlockTensor_test = {}

local taInput1 = { teDefault = torch.Tensor(1, 1, 1):fill(0),
									 nBatchSize = 4,
									 taData = {
										 { teRowIdx = torch.LongTensor({{2}, {4}}),
										 	 teValue = torch.Tensor({{{0, 0, 1, 1, 1, 1, 0},
											 												{1, 1, 1, -10, 0, 0, 0}}}) },
										 { teRowIdx = torch.LongTensor({2}),
										 	 teValue = torch.Tensor({{{ 0, -0.1, 1 }}}) }
										}
									}

local taInput2 = { nBatchSize = 4,
									 taData = {
										 { teRowIdx = torch.LongTensor({{2}, {4}}),
										 	 teValue = torch.Tensor({{{1}, {1}, {1}, {1}, {1}, {1}, {1}},
											 												{{1}, {1}, {1}, {2}, {0}, {0}, {0}}}) },
										 { teRowIdx = torch.LongTensor({{2}}),
										 	 teValue = torch.Tensor({ {{0}, {-1}, {1} }}) }
										}
									}

local taInput3 = { nBatchSize = 4,
									 taData = {
										 { teRowIdx = torch.LongTensor({{2}, {4}}),
										 	 teValue = torch.Tensor({{{1, 10}, {1, 10}, {1, 10}, {1, 10}, {1, 10}, {1, 10}, {1, 10}},
											 												{{1, 10}, {1, 10}, {1, 10}, {2, 20}, {0, 0}, {0, 0}, {0, 0}}}) },
										 { teRowIdx = torch.LongTensor({{2}}),
										 	 teValue = torch.Tensor({ {{0, 0}, {-1, -10}, {1, 10} }}) }
										}
									}

local taInput4 = { nBatchSize = 8,
									 taData = {
										 { teRowIdx = torch.LongTensor({{2}, {4}}),
										 	 teValue = torch.Tensor({{{1, 10}, {1, 10}, {1, 10}, {1, 10}, {1, 10}, {1, 10}, {1, 10}},
											 												{{1, 10}, {1, 10}, {1, 10}, {2, 20}, {0, 0}, {0, 0}, {0, 0}}}) },
										 { teRowIdx = torch.LongTensor({{8}}),
										 	 teValue = torch.Tensor({ {{0, 0}, {-1, -10}, {1, 10} }}) }
										}
									}

local taInput5 = { nBatchSize = 3,
									 taData = 
									{ 
										 {	teRowIdx = torch.LongTensor{{1}, {3}},
									 			teValue = torch.Tensor({{{1}}, {{3}}}) }
								 	},
								}

local taInput6 = { nBatchSize = 1,
									 taData = 
									{ 
										 {	teRowIdx = torch.LongTensor{{1}},
									 			teValue = torch.Tensor({{{100}}}) }
								 	},
								}


function sparseBlockTensor_test.ReLU_test1()
	local mNet = nn.SparseBlockReLU()
	local taOutput = mNet:forward(taInput2)
	deposUtil.printSparseBlockInput(taOutput)
end

-- description: TemporalConvolution
-- case: inputFrameSize=1, outputFrameSize=1
function sparseBlockTensor_test.TemporalConvolution_test1()
	local taInput = taInput2
	local mNet = nn.SparseBlockTemporalConvolution(1, 1, 3, 1)
	mNet.weight:fill(1)
	print("=========== weight ====:")
	print(mNet.weight)
	
	print("============ input ======:")
	deposUtil.printSparseBlockInput(taInput)

	local taOutput = mNet:forward(taInput)
	print("============ output ======:")
	deposUtil.printSparseBlockInput(taOutput)
end

-- description: TemporalConvolution
-- case: inputFrameSize=2, outputFrameSize=1
function sparseBlockTensor_test.TemporalConvolution_test2()
	local taInput = taInput3
	local mNet = nn.Sequential()
	local mConv = nn.SparseBlockTemporalConvolution(2, 2, 3, 2)
	mNet:add(mConv)
	mNet:add(nn.SparseBlockReLU())

	mConv.weight[1] = mConv.weight[1]:fill(1):cumsum()
	mConv.weight[2] = mConv.weight[2]:fill(2):cumsum()
	print("=========== weight ====:")
	print(mConv.weight)
	
	print("============ input ======:")
	deposUtil.printSparseBlockInput(taInput)

	local taOutput = mNet:forward(taInput)
	print("============ output ======:")
	deposUtil.printSparseBlockInput(taOutput)


	print("---------- TemporalConvolution -------")
	mNet = nn.Sequential()
	mConv = nn.TemporalConvolution(2, 2, 3, 2)
	mNet:add(mConv)
	mNet:add(nn.ReLU())
	mConv.weight[1] = mConv.weight[1]:fill(1):cumsum()
	mConv.weight[2] = mConv.weight[2]:fill(2):cumsum()
	mConv.bias:fill(0)
	print("=========== weight ====:")
	print(mConv.weight)
	local teOutput = mNet:forward(taInput.taData[1].teValue)
	print("============ output ======:")
	print(teOutput)


end

function sparseBlockTensor_test.TemporalConvolution_test3()
	print("======= mConv =======")
	local taInput = taInput3
	local mConv = nn.SparseBlockTemporalConvolution(2, 2, 3, 2)
	local taOutput = mConv:forward(taInput)

--	print("--output--")
--	deposUtil.printSparseBlockInput(taOutput)
	local gradInput = mConv:updateGradInput(taInput, taOutput)
	print("--GradInput--")
	deposUtil.printSparseBlockInput(gradInput)

	print("======= mConvMain =======")
	local mConvMain = nn.TemporalConvolution(2, 2, 3, 2)
	mConvMain.weight:copy(mConv.weight)
	mConvMain.bias:fill(0)
	local teOutput = mConvMain:forward(taInput.taData[1].teValue)
--	print("--output--")
--	print(teOutput)
	gradInput = mConvMain:updateGradInput(taInput.taData[1].teValue, teOutput)
	print("--GradInput--")
	print(gradInput)
end

function sparseBlockTensor_test.TemporalConvolution_test4()
	local scale = 1
	print("======= mConv =======")
	local taInput = taInput3
	local mConv = nn.SparseBlockTemporalConvolution(2, 2, 3, 2)
	local taOutput = mConv:forward(taInput)

--	print("--output--")
--	deposUtil.printSparseBlockInput(taOutput)
	mConv:accGradParameters(taInput, taOutput, scale)
	print("--gradWeight--")
	print(mConv.gradWeight)

	print("======= mConvMain =======")
	local mConvMain = nn.TemporalConvolution(2, 2, 3, 2)
	mConvMain.weight:copy(mConv.weight)
	mConvMain.bias:fill(0)

	--1
	local teOutput = mConvMain:forward(taInput.taData[1].teValue)
	mConvMain:accGradParameters(taInput.taData[1].teValue, teOutput, scale)

	--2
	teOutput = mConvMain:forward(taInput.taData[2].teValue)
	mConvMain:accGradParameters(taInput.taData[2].teValue, teOutput, scale)

	print("--gradWeight--")
	print(mConvMain.gradWeight)

end

function sparseBlockTensor_test.TemporalConvolution_test5()
	local scale = 1
	print("======= mConv =======")
	local taInput = taInput3
	local mConv = nn.SparseBlockTemporalConvolution(2, 2, 4, 2)
	local taOutput = mConv:forward(taInput)
end


function sparseBlockTensor_test.TemporalMaxPooling_test1()
	local taInput = taInput3
	print("======= mMaxPool =======")
	local mMaxPool = nn.SparseBlockTemporalMaxPooling(2)
	local taOutput = mMaxPool:forward(taInput)
	deposUtil.printSparseBlockInput(taOutput)

	print("======= mMaxPoolMain =======")
	local mMaxPoolMain = nn.TemporalMaxPooling(2)
	local teOutput = mMaxPoolMain:forward(taInput.taData[1].teValue)
	print(teOutput)
end

function sparseBlockTensor_test.TemporalMaxPooling_test2()
	local taInput = taInput3
	print("======= mMaxPool =======")
	local mMaxPool = nn.SparseBlockTemporalMaxPooling(2)
	local taOutput = mMaxPool:forward(taInput)
	local taGradInput = mMaxPool:updateGradInput(taInput, taOutput)
	deposUtil.printSparseBlockInput(taGradInput)

	print("======= mMaxPoolMain =======")
	local mMaxPoolMain = nn.TemporalMaxPooling(2)
	local teOutput = mMaxPoolMain:forward(taInput.taData[1].teValue)
	local teGradInput = mMaxPoolMain:updateGradInput(taInput.taData[1].teValue, teOutput)
	print(teGradInput)
end

function sparseBlockTensor_test.ConvMaxPool_test1()
	local scale = 1
	print("======= mConv =======")
	local taInput = taInput3
	local mSeq = nn.Sequential()
	local mConv1 = nn.SparseBlockTemporalConvolution(2, 4, 2, 1)
	mSeq:add(mConv1)
	mSeq:add(nn.SparseBlockTemporalMaxPooling(2))

	local taOutput = mSeq:forward(taInput)
	deposUtil.printSparseBlockInput(taOutput)
end

function sparseBlockTensor_test.SparseBlockFlattenDim3_test1()
	local taInput = taInput3

	print("======= mFlattenDim3 =======")
	local mSeq = nn.Sequential()
	local mConv = nn.SparseBlockTemporalConvolution(2, 2, 3, 2)
	mSeq:add(mConv)
	mSeq:add(nn.SparseBlockFlattenDim3())

	local taOutput = mSeq:forward(taInput)
	deposUtil.printSparseBlockInput(taOutput)

	print("===== mSeqMain ====")
	local mSeqMain = nn.Sequential()
	local mConvMain = nn.TemporalConvolution(2, 2, 3, 2)
	mConvMain.bias:fill(0)
	mConvMain.weight:copy(mConv.weight)
	mSeqMain:add(mConvMain)
	mSeqMain:add(nn.View(2, -1))

	local teOutput = mSeqMain:forward(taInput.taData[1].teValue)
	print(teOutput)

end

function sparseBlockTensor_test.SparseBlockFlattenDim3_test2()
	local taInput = taInput3

	print("======= mFlattenDim3 =======")
	local mSeq = nn.Sequential()
	local mConv = nn.SparseBlockTemporalConvolution(2, 2, 3, 2)
	mSeq:add(mConv)
	mSeq:add(nn.SparseBlockFlattenDim3())

	local taOutput = mSeq:forward(taInput)
	local taGradInput = mSeq:updateGradInput(taInput, taOutput)
	deposUtil.printSparseBlockInput(taGradInput)

	print("===== mSeqMain ====")
	local mSeqMain = nn.Sequential()
	local mConvMain = nn.TemporalConvolution(2, 2, 3, 2)
	mConvMain.bias:fill(0)
	mConvMain.weight:copy(mConv.weight)
	mSeqMain:add(mConvMain)
	mSeqMain:add(nn.View(2, -1))

	local teOutput = mSeqMain:forward(taInput.taData[1].teValue)
	local teGradInput = mSeqMain:forward(taInput.taData[1].teValue, teOutput)
	print(teGradInput)
end

function sparseBlockTensor_test.SparseBlockLinear_test1()
	local taInput = taInput3
	local mLinear = nn.SparseBlockLinear(2)
	local mSeq = nn.Sequential()
	mSeq:add(nn.SparseBlockFlattenDim3())
	mSeq:add(mLinear)
	local taOutput = mSeq:forward(taInput)
	deposUtil.printSparseBlockInput(taOutput)

end

function sparseBlockTensor_test.SparseBlockLinear_test5_bias()
	local taInput = taInput3
	print("---- input ---")
	deposUtil.printSparseBlockInput(taInput)

	local mLinear = nn.SparseBlockLinear(2, true)

	local mSeq = nn.Sequential()
	mSeq:add(nn.SparseBlockFlattenDim3())
	mSeq:add(mLinear)
	local taOutput = mSeq:forward(taInput) -- warmup
	mLinear.weight:fill(1)
	mLinear.bias:fill(0.1)
	local taOutput = mSeq:forward(taInput)
	print("---- output ---")
	deposUtil.printSparseBlockInput(taOutput)

	print("##### bias #####")
	print(mLinear.bias)
end

function sparseBlockTensor_test.SparseBlockLinear_test6_bias()
	local taInput = taInput3
	print("---- input ---")
	deposUtil.printSparseBlockInput(taInput)

	local mLinear = nn.SparseBlockLinear(2, true)

	local mSeq = nn.Sequential()
	mSeq:add(nn.SparseBlockFlattenDim3())
	mSeq:add(mLinear)
	local taOutput = mSeq:forward(taInput) -- warmup
	mLinear.weight:fill(0)
	mLinear.bias:fill(0.1)
	local taOutput = mSeq:forward(taInput)
	taOutput.taData[1].teGradOutputSum = torch.Tensor({{1, 2}})
	taOutput.taData[2].teGradOutputSum = torch.Tensor(1, 2):fill(1)

	mSeq:accGradParameters(taInput, taOutput, 1)
	mSeq:accGradParameters(taInput, taOutput, 1)
	print("***** gradWeight, gradBias:")
	print(mLinear.gradWeight)
	print(mLinear.gradBias)
	print("*****")

	print("---- output ---")
	deposUtil.printSparseBlockInput(taOutput)

	print("##### bias #####")
	print(mLinear.bias)
end


function sparseBlockTensor_test.SparseBlockLinear_test2()
	local taInput = taInput3
	print("===== mSeq ====")
	local mLinear = nn.SparseBlockLinear(2)
	local mSeq = nn.Sequential()
	mSeq:add(nn.SparseBlockFlattenDim3())
	mSeq:add(mLinear)
	local taOutput = mSeq:forward(taInput)
--	deposUtil.printSparseBlockInput(taOutput)
	local taGradInput = mSeq:updateGradInput(taInput, taOutput)
	deposUtil.printSparseBlockInput(taGradInput)

	print("===== mSeqMain ====")

	local mLinearMain = nn.Linear(14, 2)
	mLinearMain.bias:fill(0)
	mLinearMain.weight:copy(mLinear:pri_getSubWeight(1):t())
	local mSeqMain = nn.Sequential()
	mSeqMain:add(nn.View(2, -1))
	mSeqMain:add(mLinearMain)

	local teOutput = mSeqMain:forward(taInput.taData[1].teValue)
--	print(teOutput)
	local teGradInput = mSeqMain:updateGradInput(taInput.taData[1].teValue, teOutput)
	print(teGradInput)

end


function sparseBlockTensor_test.SparseBlockLinear_test3()
	local scale = 1
	local taInput = taInput3
	print("===== mSeq ====")
	local mLinear = nn.SparseBlockLinear(2)
	local mSeq = nn.Sequential()
	mSeq:add(nn.SparseBlockFlattenDim3())
	mSeq:add(mLinear)
	local taOutput = mSeq:forward(taInput)
	mSeq:accGradParameters(taInput, taOutput, scale)
	print(mLinear:pri_getSubGradWeight(1):t())

	print("===== mSeqMain ====")

	local mLinearMain = nn.Linear(14, 2)
	mLinearMain.bias:fill(0)
	mLinearMain.weight:copy(mLinear:pri_getSubWeight(1):t())
	local mSeqMain = nn.Sequential()
	mSeqMain:add(nn.View(2, -1))
	mSeqMain:add(mLinearMain)

	local teOutput = mSeqMain:forward(taInput.taData[1].teValue)
	mSeqMain:accGradParameters(taInput.taData[1].teValue, teOutput, scale)
	print(mLinearMain.gradWeight)

end

function sparseBlockTensor_test.SparseBlockLinear_test4()
	local scale = 1
	local taInput = taInput3
	print("===== mSeq ====")
	local mLinear = nn.SparseBlockLinear(1)
	local mSeq = nn.Sequential()
	mSeq:add(nn.SparseBlockFlattenDim3())
	mSeq:add(mLinear)
	local taOutput = mSeq:forward(taInput)
	deposUtil.printSparseBlockInput(taOutput)
end

function sparseBlockTensor_test.SparseBlockToDenseLinear_test1()
	local scale = 1
	local taInput = taInput4

	print("===== mSeq ====")
	local mDenseToLinear = nn.SparseBlockToDenseLinear(2)
	local mSeq = nn.Sequential()
	mSeq:add(nn.SparseBlockFlattenDim3())
	mSeq:add(mDenseToLinear)
	local teOutput = mSeq:forward(taInput)
	print(teOutput)

	print("===== mSeqMain ====")
	local mLinearMain = nn.Linear(14, 2)
	mLinearMain.bias:fill(0)
	mLinearMain.weight:copy(mDenseToLinear:pri_getSubWeight(1):t())
	local mSeqMain = nn.Sequential()
	mSeqMain:add(nn.View(2, -1))
	mSeqMain:add(mLinearMain)
	local teOutput = mSeqMain:forward(taInput.taData[1].teValue)
	print(teOutput)

	print("===== mSeqMain2 ====")
	local mLinearMain2 = nn.Linear(6, 2)
	mLinearMain2.bias:fill(0)
	mLinearMain2.weight:copy(mDenseToLinear:pri_getSubWeight(2):t())
	local mSeqMain2 = nn.Sequential()
	mSeqMain2:add(nn.View(1, -1))
	mSeqMain2:add(mLinearMain2)
	local teOutput2 = mSeqMain2:forward(taInput.taData[2].teValue)
	print(teOutput2)

end

function sparseBlockTensor_test.SparseBlockToDenseLinear_test2()
	local scale = 1
	local taInput = taInput4

	print("===== mSeq ====")
	local mDenseToLinear = nn.SparseBlockToDenseLinear(2)
	local mSeq = nn.Sequential()
	mSeq:add(nn.SparseBlockFlattenDim3())
	mSeq:add(mDenseToLinear)
	local teOutput = mSeq:forward(taInput)
	local taGradInput = mSeq:updateGradInput(taInput, teOutput)
	deposUtil.printSparseBlockInput(taGradInput)

	print("===== mSeqMain ====")
	local mLinearMain = nn.Linear(14, 2)
	mLinearMain.bias:fill(0)
	mLinearMain.weight:copy(mDenseToLinear:pri_getSubWeight(1):t())
	local mSeqMain = nn.Sequential()
	mSeqMain:add(nn.View(2, -1))
	mSeqMain:add(mLinearMain)
	local teOutput = mSeqMain:forward(taInput.taData[1].teValue)
	local teGradInput = mSeqMain:updateGradInput(taInput.taData[1].teValue, teOutput)
	print(teGradInput)
end

function sparseBlockTensor_test.SparseBlockToDenseLinear_test3()
	torch.manualSeed(1)
	local scale = 0.1
	local taInput = taInput4

	print("===== mSeq ====")
	local mDenseToLinear = nn.SparseBlockToDenseLinear(2)
	local mSeq = nn.Sequential()
	mSeq:add(nn.SparseBlockFlattenDim3())
	mSeq:add(mDenseToLinear)
	teOutput = mSeq:forward(taInput)
	mSeq:accGradParameters(taInput, teOutput, scale)
	teOutput = mSeq:forward(taInput)

	mSeq:accGradParameters(taInput, teOutput, scale)
	print(mDenseToLinear:pri_getSubGradWeight(1))
--	print(mDenseToLinear:pri_getSubWeight(1))


	print("===== mSeqMain ====")
	local mLinearMain = nn.Linear(14, 2)
	mLinearMain.bias:fill(0)
	mLinearMain.weight:copy(mDenseToLinear:pri_getSubWeight(1):t())
	local mSeqMain = nn.Sequential()
	mSeqMain:add(nn.View(2, -1))
	mSeqMain:add(mLinearMain)
	local teOutput = mSeqMain:forward(taInput.taData[1].teValue)
	mSeqMain:accGradParameters(taInput.taData[1].teValue, teOutput, scale)
	teOutput = mSeqMain:forward(taInput.taData[1].teValue)

	mSeqMain:accGradParameters(taInput.taData[1].teValue, teOutput, scale)
	print(mLinearMain.gradWeight:t())
--	print(mLinearMain.weight:t())

end


function sparseBlockTensor_test.SparseBlockToDenseLinear_test4_teDefault()
	torch.manualSeed(2)
	local taInput = taInput3
	local mLinear = nn.SparseBlockLinear(2, true)
	local mDenseToLinear = nn.SparseBlockToDenseLinear(2, true)
	local mSeq = nn.Sequential()
	local mFlatten = nn.SparseBlockFlattenDim3()
	mSeq:add(mFlatten)
	mSeq:add(mLinear)
	mSeq:add(mDenseToLinear)

-- warmup:
	local taOutput = mSeq:forward(taInput)

	-- set weights:
	mLinear.weight:fill(-100)
	mLinear.bias:fill(10)

	mDenseToLinear.weight:fill(2)
	mDenseToLinear.bias[1][1] = -0.1--:fill(-0.1)
	mDenseToLinear.bias[1][2] = -0.2--:fill(-0.1)

	print("======= mSeq =======")
	local taOutput = mSeq:forward(taInput)

	print("#### mFlatten output:")
	deposUtil.printSparseBlockInput(mFlatten.output)

	print("#### mLinear output:")
	deposUtil.printSparseBlockInput(mLinear.output)

	print("#### mDenseToLinear output:")
	print(taOutput)
--

end

function sparseBlockTensor_test.SparseBlockToDenseLinear_test5_teDefault()
	torch.manualSeed(2)
	local taInput = taInput3
	local mLinear = nn.SparseBlockLinear(2, true)
	local mDenseToLinear = nn.SparseBlockToDenseLinear(2, true)
	local mSeq = nn.Sequential()
	local mFlatten = nn.SparseBlockFlattenDim3()
	mSeq:add(mFlatten)
	mSeq:add(mLinear)
	mSeq:add(mDenseToLinear)

-- warmup:
	local taOutput = mSeq:forward(taInput)

	-- set weights:
	mLinear.weight:fill(-100)
	mLinear.bias:fill(10)

	mDenseToLinear.weight:fill(2)
	mDenseToLinear.bias[1][1] = -0.1--:fill(-0.1)
	mDenseToLinear.bias[1][2] = -0.2--:fill(-0.1)

	print("======= mSeq =======")
	local taOutput = mSeq:forward(taInput)
	local taGradInput = mSeq:updateGradInput(taInput, taOutput)
	print(mDenseToLinear.gradInput.taData[2].teGradOutputSum)

end
function sparseBlockTensor_test.SparseBlockToDenseLinear_test6_teDefault()
	torch.manualSeed(2)
	local taInput = taInput3
	local mLinear = nn.SparseBlockLinear(2, true)
	local mDenseToLinear = nn.SparseBlockToDenseLinear(2, true)
	local mSeq = nn.Sequential()
	local mFlatten = nn.SparseBlockFlattenDim3()
	mSeq:add(mFlatten)
	mSeq:add(mLinear)
	mSeq:add(mDenseToLinear)

-- warmup:
	local taOutput = mSeq:forward(taInput)

	-- set weights:
	mLinear.weight:fill(-100)
	mLinear.bias:fill(10)

	mDenseToLinear.weight:fill(2)
	mDenseToLinear.bias[1][1] = -0.1--:fill(-0.1)
	mDenseToLinear.bias[1][2] = -0.2--:fill(-0.1)

	print("======= mSeq =======")
	local teOutput = mSeq:forward(taInput)
	local taGradInput = mSeq:updateGradInput(taInput, teOutput)
	print("$$$$ before $$$")
	print(mDenseToLinear.gradBias)
	mSeq:accGradParameters(taInput, teOutput)
	print("$$$$ after $$$")
	print(mDenseToLinear.gradBias)
end

function printParams(m)
	print("weight:")
	print(m.weight)
	print("bias:")
	print(m.bias)
end

function printGradParams(m)
	print("gradWeight:")
	print(m.gradWeight)
	print("gradBias:")
	print(m.gradBias)
end

function matchGradParams(m1, m2)
	local isGradWeightMatch = torch.all(torch.eq(m1.gradWeight, m2.gradWeight))
	assert(isGradWeightMatch, "****** gradWeight don't match!")

	local isGradBiasMatch = torch.all(torch.eq(m1.gradBias, m2.gradBias))
	assert(isGradBiasMatch, "****** gradBias don't match!")
end

function matchTensors(teA, teB)
	local isMatch = torch.all(torch.eq(teA, teB))
	assert(isMatch, "--------- tensors don't match")
end

function sparseBlockTensor_test.SparseBlockToDenseLinear_test7_validate()
	torch.manualSeed(1)
	local taInput = taInput6

	local teGradOutput = torch.ones(taInput.nBatchSize, 1)--:uniform(-1, 1)
	local scale = 1

	print("===== mSeq ====")
	-- a) create arch
	local mDenseToLinear = nn.SparseBlockToDenseLinear(1, true)
	local mBlockLinear = nn.SparseBlockLinear(1, true)
	local mSeq = nn.Sequential()
	mSeq:add(nn.SparseBlockFlattenDim3())
	mSeq:add(mBlockLinear)
	mSeq:add(mDenseToLinear)

	-- b) warmap
	local teOutput = mSeq:forward(taInput)
	mSeq:updateGradInput(taInput, teOutput:zero())

	-- c) assign parameters
	mBlockLinear.weight:fill(1)
	mBlockLinear.bias:fill(1)
	mDenseToLinear.weight:fill(1)
	mDenseToLinear.bias:fill(1)

	-- d) output
	local teOutput = mSeq:forward(taInput)
	local taGradInput = mSeq:backward(taInput, teGradOutput, scale)
	--print(teOutput)
	printGradParams(mDenseToLinear)

	print("===== mSeqMain ====")
	-- A) create arch
	local mLinear = nn.Linear(1, 1)
	local mLinearLast = nn.Linear(1, 1)
	local mSeqMain = nn.Sequential()
	mSeqMain:add(mLinear)
	mSeqMain:add(mLinearLast)

	-- B) prepare input
	local nRows = teOutput:size(1)
	local teInput = torch.Tensor(nRows, 1):fill(100)
--	teInput[1][1] = 1
--	teInput[3][1] = 3

	-- C) assign parameters
	mLinear.weight:copy(mBlockLinear.weight)
	mLinear.bias:copy(mBlockLinear.bias)
	mLinearLast.weight:copy(mDenseToLinear.weight)
	mLinearLast.bias:copy(mDenseToLinear.bias)

	local teOutputMain = mSeqMain:forward(teInput)
	local taGradInput = mSeqMain:backward(teInput, teGradOutput, scale)
--	print(teOutputMain)
	printGradParams(mLinearLast)

	matchTensors(teOutput, teOutputMain)
--	matchGradParams(mBlockLinear, mLinear)
	matchGradParams(mDenseToLinear, mLinearLast)
end



function sparseBlockTensor_test.SparseBlockDropout_test1()
	torch.manualSeed(2)
	local taInput = taInput4
	local mDropout = nn.SparseBlockDropout(0.91)
	local taOutput = mDropout:forward(taInput)
	print("======= taOutput ======")
	deposUtil.printSparseBlockInput(taOutput)
	local taGradInput = mDropout:updateGradInput(taInput, taOutput)
	print("======= taGradInput ======")
	deposUtil.printSparseBlockInput(taGradInput)

end

--sparseBlockTensor_test.ReLU_test1()
--sparseBlockTensor_test.TemporalConvolution_test1()
--sparseBlockTensor_test.TemporalConvolution_test2()
--sparseBlockTensor_test.TemporalConvolution_test3()
--sparseBlockTensor_test.TemporalConvolution_test4()

--sparseBlockTensor_test.TemporalMaxPooling_test1()
--sparseBlockTensor_test.TemporalMaxPooling_test2()
--sparseBlockTensor_test.ConvMaxPool_test1()
--sparseBlockTensor_test.SparseBlockFlattenDim3_test1()
--sparseBlockTensor_test.SparseBlockLinear_test1()
--sparseBlockTensor_test.SparseBlockLinear_test2()
--sparseBlockTensor_test.SparseBlockLinear_test3()
--sparseBlockTensor_test.SparseBlockLinear_test4()
--sparseBlockTensor_test.SparseBlockLinear_test5_bias()
--sparseBlockTensor_test.SparseBlockLinear_test6_bias()
--sparseBlockTensor_test.SparseBlockToDenseLinear_test1()
--sparseBlockTensor_test.SparseBlockToDenseLinear_test2()
--sparseBlockTensor_test.SparseBlockToDenseLinear_test3()
--sparseBlockTensor_test.SparseBlockToDenseLinear_test4_teDefault()
--sparseBlockTensor_test.SparseBlockToDenseLinear_test5_teDefault()
--sparseBlockTensor_test.SparseBlockToDenseLinear_test6_teDefault()
--sparseBlockTensor_test.SparseBlockToDenseLinear_test7_validate()
--sparseBlockTensor_test.SparseBlockDropout_test1()

sparseBlockTensor_test.TemporalConvolution_test5()
