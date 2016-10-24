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
local deposUtil = deposUtil or require('../deposUtil.lua')

local sparseBlockTensor_test = {}

local taInput1 = { teDefault = torch.Tensor(1, 1, 1):fill(0),
									 nBatchSize = 4,
									 taData = {
										 { teRowIdx = torch.LongTensor({2, 4}),
										 	 teValue = torch.Tensor({{{0, 0, 1, 1, 1, 1, 0},
											 												{1, 1, 1, -10, 0, 0, 0}}}) },
										 { teRowIdx = torch.LongTensor({2}),
										 	 teValue = torch.Tensor({{{ 0, -0.1, 1 }}}) }
										}
									}

local taInput2 = { nBatchSize = 4,
									 taData = {
										 { teRowIdx = torch.LongTensor({2, 4}),
										 	 teValue = torch.Tensor({{{1}, {1}, {1}, {1}, {1}, {1}, {1}},
											 												{{1}, {1}, {1}, {2}, {0}, {0}, {0}}}) },
										 { teRowIdx = torch.LongTensor({2}),
										 	 teValue = torch.Tensor({ {{0}, {-1}, {1} }}) }
										}
									}

local taInput3 = { nBatchSize = 4,
									 taData = {
										 { teRowIdx = torch.LongTensor({2, 4}),
										 	 teValue = torch.Tensor({{{1, 10}, {1, 10}, {1, 10}, {1, 10}, {1, 10}, {1, 10}, {1, 10}},
											 												{{1, 10}, {1, 10}, {1, 10}, {2, 20}, {0, 0}, {0, 0}, {0, 0}}}) },
										 { teRowIdx = torch.LongTensor({2}),
										 	 teValue = torch.Tensor({ {{0, 0}, {-1, -10}, {1, 10} }}) }
										}
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


--sparseBlockTensor_test.ReLU_test1()
--sparseBlockTensor_test.TemporalConvolution_test1()
--sparseBlockTensor_test.TemporalConvolution_test2()
--sparseBlockTensor_test.TemporalConvolution_test3()
--sparseBlockTensor_test.TemporalConvolution_test4()
--sparseBlockTensor_test.TemporalMaxPooling_test1()
--sparseBlockTensor_test.TemporalMaxPooling_test2()
sparseBlockTensor_test.SparseBlockFlattenDim3_test1()
