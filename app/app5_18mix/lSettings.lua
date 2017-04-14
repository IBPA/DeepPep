local lSettings = {}

do
  lSettings.strBaseDir = "/Users/ameen/mygithub/depos/app/app5_18mix/sparseData2"
  lSettings.strFilenameTarget = string.format("%s/target.csv", lSettings.strBaseDir)
  lSettings.strFilenameMetaInfo = string.format("%s/metaInfo.csv", lSettings.strBaseDir)
  lSettings.strFilenameProtRef = string.format("%s/18mix_reference.csv", lSettings.strBaseDir)
  lSettings.strFilenameProtInfo = string.format("%s/protInfo.csv", lSettings.strBaseDir)
  lSettings.strFilenameExperiment1Obj = string.format("./model/experiment_1.obj" )
  lSettings.nRows=759 -- ToDo: find the right number

	

	function lSettings.setExprId(nExprId)
		lSettings.nExprId = nExprId
 	  lSettings.strFilenameProtInfo = string.format("%s/protInfo_expr_%d.csv", lSettings.strBaseDir, nExprId)
 	  lSettings.strFilenameExprDescription = lSettings.strFilenameProtInfo .. ".desc"	
 	  lSettings.strFilenameExprParams = lSettings.strFilenameProtInfo .. ".params"	
	end

  return lSettings
end

