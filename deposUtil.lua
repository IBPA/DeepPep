do
	local deposUtil = {}

	function deposUtil.printSparseBlockInput(taInput)
		local taData = taInput.taData
		local nDim = 3 

		if taInput.teDefault ~= nil then
			nDim = taInput.teDefault:dim()
			print("teDefault:" .. tostring(taInput.teDefault:squeeze(3)))
		end

		for i=1, #taData do
			
			nDim = taData[i].teValue:dim()
			io.write(string.format("#%d: ", i))
			print(taData[i].teValue:squeeze(nDim))

		end
	end

	function deposUtil.mulSparseBlockInput(taInput, dValue)
		taInput.teDefault:mul(dValue)

		local taData = taInput.taData
		for i=1, #taData do
			taData[i].teValue:mul(dValue)
		end
	end

	function deposUtil.getCopyRandomizedBlocks(taInput)
		local taRes = { teDefault = torch.rand(taInput.teDefault:size()),
										taData = {}}

		local taData = taInput.taData
		for i=1, #taData do
			local taNew = { teValue = torch.rand(taData[i].teValue:size()),
											teRowIdx = taData[i].teRowIdx:clone()}
			table.insert(taRes.taData, taNew)
		end


		return taRes
	end

	return deposUtil
end
