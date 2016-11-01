local lSettings = {}

do
  lSettings.strBaseDir = "/Users/Ameen/mygithub/depos/app/testApp11_18mix/sparseData2"
  lSettings.strFilenameTarget = string.format("%s/target.csv", lSettings.strBaseDir)
  lSettings.strFilenameMetaInfo = string.format("%s/metaInfo.csv", lSettings.strBaseDir)
  lSettings.strFilenameProtRef = string.format("%s/18mix_reference.csv", lSettings.strBaseDir)
  lSettings.strFilenameProtInfo = string.format("%s/protInfo.csv", lSettings.strBaseDir)
  lSettings.strFilenameExperiment1Obj = string.format("./model/experiment_1.obj" )
  lSettings.strFilenameExperiment2_LinearObj = string.format("./model/experiment_2_Linear.obj" )
  lSettings.nRows=759 -- ToDo: find the right number

  return lSettings
end

