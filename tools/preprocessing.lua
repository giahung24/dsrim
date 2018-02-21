require 'tools/toolkit'

local PreProcess = torch.class("PreProcess")

---- Make a relevance dict for each qNum:{list of docNum}
function PreProcess:makeRelevanceDict(qRelPath, docIDListPath)
    if not fileExists(qRelPath) or not fileExists(docIDListPath) then 
        print("Error! Input file not found")
        return nil;
    end
    --process docIdList
    local docIdDict = {}
    local count = 1
    for line in io.lines(docIDListPath) do 
        docIdDict[line] = (count)
        count=count + 1
    end
    
    -- process qRel
    local qNum = 0
    local lastQid = ""
    local qRelevantDict = {}
    local totalPair=0
    for line in io.lines(qRelPath) do 
        local fields = split(line, " ")
        local qId = fields[1]
        local visitId = fields[3]
        local thisDocIsRelevant = tonumber(fields[4])>0
        
        if qId ~= lastQid then
        	qNum = qNum + 1           
        	lastQid = qId
        end
        
        if thisDocIsRelevant then
            local docNum = docIdDict[visitId]
            local relDocs = qRelevantDict[qNum]
            if relDocs == nil then
                relDocs = {}
            end
            table.insert(relDocs,docNum)
            qRelevantDict[qNum] = relDocs
            totalPair = totalPair + 1
            ---- qNum - docNum : indexId + 1
            
        end 

    end
--    print(totalPair)
    return qRelevantDict
end

---- Read termVector csv file, return tensor(rowNum,colNum)
function PreProcess:readDataSet(filePath, rowNum, colNum)
    -- Read data from CSV to tensor
    print('Reading '..filePath)
    local csvFile = io.open(filePath, 'r')  
    --local header = csvFile:read()
    
    local data = torch.IntTensor(rowNum, colNum)
    
    i = 0  
    for line in csvFile:lines('*l') do  
      i = i + 1
      data[i] = torch.IntTensor{split(line,',')}
      -- for key, val in ipairs(l) do
      --   data[i][key] = val
      -- end
      -- print('line '..i)
    end
    csvFile:close()
    return data
end

