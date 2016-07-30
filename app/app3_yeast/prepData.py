import xml.etree.ElementTree as ET


xmlPath = '/home/user/eetemame/data/protein/yeast/all/interact.pep.xml'
tree = ET.parse(xmlPath)
root = tree.getroot()
print('begin')
print(root.tag)
baseXmlAddrs="{http://regis-web.systemsbiology.net/pepXML}"
#element = root.find(baseXmlAddrs + "msms_run_summary/" + baseXmlAddrs + "spectrum_query/" + baseXmlAddrs + "search_hit")
#print(element.tag)


#for child in root.findall(baseXmlAddrs + "msms_run_summary/" + baseXmlAddrs + "spectrum_query/" + baseXmlAddrs + "search_hit"):
counter = 0
dicAll = {}
for eSearchHit in root.findall(".//" +  baseXmlAddrs + "search_hit"):
    counter+=1
    strPeptide = eSearchHit.get('peptide')

    ePeptideProphetRes = eSearchHit.find(".//" + baseXmlAddrs + "peptideprophet_result")
    dProbability = ePeptideProphetRes.get('probability')

    if strPeptide not in dicAll:
        dicAll[strPeptide] = list()

    dicAll[strPeptide].append(dProbability)

#    if counter > 10:
#        break

print(counter)
print(len(dicAll))
'''
for xElementHit in root.findall('"{http://regis-web.systemsbiology.net/pepXML}analysis_summary'):
    print('#')
#    strPeptide = xElementHit.get('peptide')
#    print(strPeptide)

    counter+=1
    if counter > 10:
        break
'''
