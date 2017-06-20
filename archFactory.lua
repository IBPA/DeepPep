--[[ Description:
       Each function in this file generates an nn architecture (not all of these are used in DeepPep)]]
do
  local archFactory = {}

  archFactory.taBuilders = {
    function(self) -- 1
      self.strArchDescription = "MaxPooling(nPoolingWindow=4), Linear(nNodes=2)"

      self.mFirst = nn.Sequential()
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(4))

        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(2, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 2
      self.strArchDescription = "MaxPooling(nPoolingWindow=8), Linear(nNodes=2)"

      self.mFirst = nn.Sequential()
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(8))

        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(2, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 3
      self.strArchDescription = "MaxPooling(nPoolingWindow=12), Linear(nNodes=1)"

      self.mFirst = nn.Sequential()
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(12))

        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(2, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 4
      self.strArchDescription = "MaxPooling(nPoolingWindow=16), Linear(nNodes=2)"

      self.mFirst = nn.Sequential()
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(16))

        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(2, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 5
      self.strArchDescription = "MaxPooling(nPoolingWindow=21), Linear(nNodes=1)"

      self.mFirst = nn.Sequential()
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(21))

        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(1, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 6
      self.strArchDescription = "MaxPooling(nPoolingWindow=24), Linear(nNodes=2)"

      self.mFirst = nn.Sequential()
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(24))

        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(2, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 7
      self.strArchDescription = "MaxPooling(nPoolingWindow=28), Linear(nNodes=1)"

      self.mFirst = nn.Sequential()
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(28))

        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(2, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 8
      self.strArchDescription = "Linear(nNodes=1)"

      self.mFirst = nn.Sequential()

        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(1, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 9
      self.strArchDescription = "Linear(nNodes=2)"

      self.mFirst = nn.Sequential()

        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(2, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 10
      self.strArchDescription = "Linear(nNodes=3)"

      self.mFirst = nn.Sequential()

        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(3, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 11
      self.strArchDescription = "Conv(in=1, out=5, kW=8), ReLU, MaxPool(w=4), Conv(in=5, out=10, kW=8), ReLU, MaxPool(w=3), Linear(nNodes=2)"

      self.mFirst = nn.Sequential()

        self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 5, 8))
        self.mFirst:add(nn.SparseBlockReLU())
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(4))

        self.mFirst:add(nn.SparseBlockTemporalConvolution(5, 10, 8))
        self.mFirst:add(nn.SparseBlockReLU())
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(3))


        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(2, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 12
      self.strArchDescription = "Conv(in=1, out=2, kW=8), ReLU, MaxPool(w=4), Conv(in=2, out=4 kW=8), ReLU, MaxPool(w=3), Linear(nNodes=2)"

      self.mFirst = nn.Sequential()

        self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 2, 8))
        self.mFirst:add(nn.SparseBlockReLU())
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(4))

        self.mFirst:add(nn.SparseBlockTemporalConvolution(2, 4, 8))
        self.mFirst:add(nn.SparseBlockReLU())
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(3))


        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(2, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 13
      self.strArchDescription = "Conv(in=1, out=1, kW=8), ReLU, MaxPool(w=4), Conv(in=1, out=1 kW=8), ReLU, MaxPool(w=3), Linear(nNodes=2)"

      self.mFirst = nn.Sequential()

        self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 1, 8))
        self.mFirst:add(nn.SparseBlockReLU())
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(4))

        self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 1, 8))
        self.mFirst:add(nn.SparseBlockReLU())
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(3))


        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(2, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 14
      self.strArchDescription = "Conv(in=1, out=1, kW=4), ReLU, MaxPool(w=4), Conv(in=1, out=1 kW=4), ReLU, MaxPool(w=3), Linear(nNodes=1)"

      self.mFirst = nn.Sequential()

        self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 1, 4))
        self.mFirst:add(nn.SparseBlockReLU())
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(4))

        self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 1, 4))
        self.mFirst:add(nn.SparseBlockReLU())
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(3))


        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(2, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 15 (used for 18mix, yeast)
      self.strArchDescription = "MaxPooling(nPoolingWindow=12)"

      self.mFirst = nn.Sequential()
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(12))

        self.mFirst:add(nn.SparseBlockFlattenDim3())

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 16 (used for sigma49)
      self.strArchDescription = "MaxPooling(nPoolingWindow=21)"

      self.mFirst = nn.Sequential()
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(21))

        self.mFirst:add(nn.SparseBlockFlattenDim3())

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 17
      self.strArchDescription = "just final layer"

      self.mFirst = nn.Sequential()

        self.mFirst:add(nn.SparseBlockFlattenDim3())

      self.mRest = nn.SparseBlockToDenseLinear(1, false, 0)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 18
      self.strArchDescription = "MaxPooling(nPoolingWindow=25)"

      self.mFirst = nn.Sequential()
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(25, 25, false, false))

        self.mFirst:add(nn.SparseBlockFlattenDim3())

      self.mRest = nn.SparseBlockToDenseLinear(1, false, 0)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 19
      self.strArchDescription = "Simply Count in first layer"

      self.mFirst = nn.Sequential()
        self.mFirst:add(nn.SparseBlockSum())

        self.mFirst:add(nn.SparseBlockFlattenDim3())

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,

    function(self) --20
      self.strArchDescription = "Layer1: Cleavage porbability, LayerFinal:  DenseSum "

      self.mFirst = nn.Sequential()
        self.mFirst:add(nn.SparseCleavageProb())

        --self.mRest = nn.SparseBlockToDenseLinear(1, false, 1) --change this to sum 
        self.mRest = nn.SparseBlockToDenseSum() 

        self.mNet = nn.Sequential()
          self.mNet:add(self.mFirst)
          self.mNet:add(self.mRest)
    end,

    function(self) --21
      self.strArchDescription = "Layer1: MaxPooling(nPoolingWindow=4), Linear, Sum"

      self.mFirst = nn.Sequential()

        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(4))
        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(1, false))

        self.mRest = nn.SparseBlockToDenseSum() 

        self.mNet = nn.Sequential()
          self.mNet:add(self.mFirst)
          self.mNet:add(self.mRest)
    end,

    function(self) --22
      self.strArchDescription = "Layer1: Cleavage porbability, LayerFinal:  DenseMul"

      self.mFirst = nn.Sequential()
        self.mFirst:add(nn.SparseCleavageProbC())

        self.mRest = nn.SparseBlockToDenseMul(-1, 1) 

        self.mNet = nn.Sequential()
          self.mNet:add(self.mFirst)
          self.mNet:add(self.mRest)
    end,

    function(self) --23
      self.strArchDescription = "Layer1: Linear (change to LinearNonNegativeWeights)"

      self.mFirst = nn.Sequential()
        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinearNonNegativeW(1))

        self.mRest = nn.SparseBlockToDenseSum() 

        self.mNet = nn.Sequential()
          self.mNet:add(self.mFirst)
          self.mNet:add(self.mRest)
    end,
    function(self) -- 24
      self.strArchDescription = "experimenting ...." -- For DeepPep

      self.mFirst = nn.Sequential()

        self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 8, 8))
        self.mFirst:add(nn.SparseBlockReLU())
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(8))

--        self.mFirst:add(nn.SparseBlockTemporalConvolution(4, 8, 4))
--        self.mFirst:add(nn.SparseBlockReLU())
--        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(4))

        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(2, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self, taArchParams) -- 25
      self.strArchDescription = "Arch25, Conv Configurable" 

      local nOutputFrameConv1 = taArchParams.nOutputFrameConv1 or 8
      local nWindowSizeConv1 = taArchParams.nWindowSizeConv1 or 8
      local nWindowSizeMaxPool1 = taArchParams.nWindowSizeMaxPool1 or 8

      self.mFirst = nn.Sequential()

        self.mFirst:add(nn.SparseBlockTemporalConvolution(1, nOutputFrameConv1, nWindowSizeConv1))
        self.mFirst:add(nn.SparseBlockReLU())
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(nWindowSizeMaxPool1))



        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(2, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self) -- 26 Linear
      self.strArchDescription = "Linear ...." 

      self.mFirst = nn.Sequential()

        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(1, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self, taArchParams) -- 27

      local nOutputPerColumn = taArchParams and taArchParams.nOutputPerColumn or 10
      local nFirstLayers = taArchParams and taArchParams.nFirstLayers or 0
      self.strArchDescription = string.format("Lin27, Deep Linear Configurable, nOutputPerColumn:%d, nFirstLayers:%d",  nOutputPerColumn, nFirstLayers)

      self.mFirst = nn.Sequential()

        self.mFirst:add(nn.SparseBlockFlattenDim3())
        
        for i=1, nFirstLayers do
          self.mFirst:add(nn.SparseBlockLinear(nOutputPerColumn, false))
          self.mFirst:add(nn.SparseBlockReLU(0, 1))
        end

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,
    function(self, taArchParams) -- 28
      self.strArchDescription = "Arch28, Conv Configurable" 

      local nWindowSizeConv1 = taArchParams and taArchParams.nWindowSizeConv1 or 8
      local nWindowSizeMaxPool1 = taArchParams and taArchParams.nWindowSizeMaxPool1 or 3

      local dW = 3

      self.mFirst = nn.Sequential()

        self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 5, nWindowSizeConv1, dW, true))
        self.mFirst:add(nn.SparseBlockReLU())
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(nWindowSizeMaxPool1, nWindowSizeMaxPool1, true))
        self.mFirst:add(nn.SparseBlockDropout(0.2))

        self.mFirst:add(nn.SparseBlockTemporalConvolution(5, 10, nWindowSizeConv1, dW, true))
        self.mFirst:add(nn.SparseBlockReLU())
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(nWindowSizeMaxPool1, nWindowSizeMaxPool1, true))
        self.mFirst:add(nn.SparseBlockDropout(0.2))


        self.mFirst:add(nn.SparseBlockTemporalConvolution(10, 15, nWindowSizeConv1, dW, true))
        self.mFirst:add(nn.SparseBlockReLU())
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(nWindowSizeMaxPool1, nWindowSizeMaxPool1, true))
        self.mFirst:add(nn.SparseBlockDropout(0.2))


        self.mFirst:add(nn.SparseBlockTemporalConvolution(15, 20, nWindowSizeConv1, dW, true))
        self.mFirst:add(nn.SparseBlockReLU())
        self.mFirst:add(nn.SparseBlockTemporalMaxPooling(nWindowSizeMaxPool1, nWindowSizeMaxPool1, true))
        self.mFirst:add(nn.SparseBlockDropout(0.2))


        self.mFirst:add(nn.SparseBlockFlattenDim3())
        self.mFirst:add(nn.SparseBlockLinear(2, false))

      self.mRest = nn.SparseBlockToDenseLinear(1, false)

      self.mNet = nn.Sequential()
        self.mNet:add(self.mFirst)
        self.mNet:add(self.mRest)
    end,

  }

  function archFactory.getArchBuilder(nArchId)
    return archFactory.taBuilders[nArchId]
  end

  return archFactory
end
