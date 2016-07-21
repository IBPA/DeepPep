local lSettings = {}

do
  local strBaseDir = "/home/ameen/mygithub/depos/DeepPep/"
  lSettings.strFilenameInputSparse = string.format("%s/input_sparseList.csv", strBaseDir)
  lSettings.strFilenameTarget = string.format("%s/target.csv", strBaseDir)
  lSettings.nInputWidth=50104

  return lSettings
end

