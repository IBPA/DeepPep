
local nSize = tonumber(arg[1] )
local teValue = torch.Tensor(nSize):fill(1)

	print("waiting for 20 seconds ...")
	os.execute("sleep 20" )
	print("done")

print(teValue:sum())
