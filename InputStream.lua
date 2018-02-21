require 'torch'
require 'tools/toolkit'
require( "SemanticVectorLoader" )  -- unavailable
require( "Doc2VecLoader" )  -- unavailable


local InputStream = torch.class("InputStream")
--[[
    After each training epoch, should call self:shuffle() to shuffle training set
]]--

function InputStream:__init(scenario, preparedPath, dataset, min_max, k_cluster)
    --[[
        Init a training scenario (a fold of k) 
        @param scenario: int from 0-9 (k-fold)
        @param preparedPath: directory of "prepared data"
    ]]--
    self.__indexTrain = 1
    self.__indexDev = 1
    self.__indexTest = 1

    self.dataset = dataset
    if scenario < 0 or scenario > 9 then
        error("Invalid scenario! Must be between 0 and 9")
    end
    if not string.endWith(preparedPath, '/') then
        preparedPath = preparedPath.."/"
    end
    self.preparedPath = preparedPath
    local scenarioPath = self.preparedPath  
    -------------- load scenario file ---------------------
    scenarioPath = scenarioPath .. "k_fold_query_id/" .. scenario .. ".train.dev.test.txt"
    -- a scenario file contains 3 lines of query ids
    scenarioFile = io.open (scenarioPath, 'r')
    if not scenarioFile then
        error( "Invalid Pretrait Path => scenario file not found!" )
    end
    local trainString = scenarioFile:read()
    local devString = scenarioFile:read()
    local testString = scenarioFile:read()
    scenarioFile:close()
    -- remove first '[' and last ']'
    trainString = trainString:sub(2,trainString:len()-1)
    devString = devString:sub(2,devString:len()-1)
    testString = testString:sub(2,testString:len()-1)
    -- split into list of query id
    self.trainSetQuery = split(trainString, ", ")
    self.devSetQuery = split(devString, ", ")
    self.testSetQuery = split(testString, ", ")
    shuffleTable(self.trainSetQuery) 
    ----------------------------------------------------
    ------------- Get query documents pairs ------------
    self.trainPairs = {}
    self:loadTrainPairs()
    self.devPairs = {}
    self:loadDevPairs()
    self.testPairs = {}
    self:loadTestPairs()
    ----------------------------------------------------
    ------------- Init vector loaders ------------------
    --[[ 
        this corresponds to 2 input parts [section 3.3.1 in the paper]
        I don't put the source code of these 2 classes because I can't share the huge prepared vectors data
        You must generate the ParagraphVector (use gensim.doc2vec) also the SemanticVector (section 3.2)
        Given a doc_id/query_id, the 'loader' return its prepared vector.
    --]]
    self.d2vLoader = Doc2VecLoader(self.preparedPath) 
    self.semVecLoader = SemanticVectorLoader(self.preparedPath, dataset, min_max, k_cluster)

end


function InputStream:hasNextTrainPair()
    return self.__indexTrain <= #self.trainPairs
end


function InputStream:hasNextDevPair()
    return self.__indexDev <= #self.devPairs
end


function InputStream:hasNextTestPair()
    return self.__indexTest <= #self.testPairs
end


function InputStream:nextTrainPair()
    -- return table of tensor: {Q,D+,D-,D-,D-,D-}
    local out = self.trainPairs[self.__indexTrain]

    self.__indexTrain = self.__indexTrain + 1
    local idList = split(out, " ")
    -- Q vectors
    local queryTxtVec = self.d2vLoader:getQueryTextVector(idList[1])
    local querySemVec = self.semVecLoader:getQueryVector(idList[1])
    if queryTxtVec == nil then
        return nil
    end
    if querySemVec == nil then
        return nil
    end
    local queryVector = torch.cat(queryTxtVec,querySemVec)

    -- D+ vectors
    local posDocTxtVec = self.d2vLoader:getDocTextVector(idList[2],self.dataset)
    local posDocSemVec = self.semVecLoader:getDocVector(idList[2],self.dataset)
    if posDocTxtVec == nil then
        return nil
    end
    if posDocSemVec == nil then
        return nil
    end
    local posDocVector = torch.cat(posDocTxtVec,posDocSemVec)

    -- D- vectors
    local negDocTxtVec1 = self.d2vLoader:getDocTextVector(idList[3],self.dataset)
    local negDocSemVec1 = self.semVecLoader:getDocVector(idList[3],self.dataset)

    local negDocTxtVec2 = self.d2vLoader:getDocTextVector(idList[4],self.dataset)
    local negDocSemVec2 = self.semVecLoader:getDocVector(idList[4],self.dataset)

    local negDocTxtVec3 = self.d2vLoader:getDocTextVector(idList[5],self.dataset)
    local negDocSemVec3 = self.semVecLoader:getDocVector(idList[5],self.dataset)

    local negDocTxtVec4 = self.d2vLoader:getDocTextVector(idList[6],self.dataset)
    local negDocSemVec4 = self.semVecLoader:getDocVector(idList[6],self.dataset)

    local not_nil = 0
    if negDocTxtVec1 ~= nil then
        not_nil = negDocTxtVec1
    elseif negDocTxtVec2 ~= nil then
        not_nil = negDocTxtVec2
    elseif negDocTxtVec3 ~= nil then
        not_nil = negDocTxtVec3
    elseif negDocTxtVec4 ~= nil then
        not_nil = negDocTxtVec4
    end
    if not_nil == 0 then
        return nil
    end

    if negDocTxtVec1 == nil then
         negDocTxtVec1 = not_nil
    end
    if negDocTxtVec2 == nil then
         negDocTxtVec2 = not_nil
    end
    if negDocTxtVec3 == nil then
         negDocTxtVec3 = not_nil
    end
    if negDocTxtVec4 == nil then
         negDocTxtVec4 = not_nil
    end

    not_nil = 0
    if negDocSemVec1 ~= nil then
        not_nil = negDocSemVec1
    elseif negDocSemVec2 ~= nil then
        not_nil = negDocSemVec2
    elseif negDocSemVec3 ~= nil then
        not_nil = negDocSemVec3
    elseif negDocSemVec4 ~= nil then
        not_nil = negDocSemVec4
    end
    if not_nil == 0 then
        return nil
    end

    if negDocSemVec1 == nil then
         negDocSemVec1 = not_nil
    end
    if negDocSemVec2 == nil then
         negDocSemVec2 = not_nil
    end
    if negDocSemVec3 == nil then
         negDocSemVec3 = not_nil
    end
    if negDocSemVec4 == nil then
         negDocSemVec4 = not_nil
    end

    if negDocSemVec1 == nil then
        return nil
    end
    local negDocVector1 = torch.cat(negDocTxtVec1,negDocSemVec1)
    local negDocVector2 = torch.cat(negDocTxtVec2,negDocSemVec2)
    local negDocVector3 =  torch.cat(negDocTxtVec3,negDocSemVec3)
    local negDocVector4 = torch.cat(negDocTxtVec4,negDocSemVec4)
    local tableOut = {}
    tableOut['Q'] = queryVector
    tableOut['D+'] = posDocVector
    -- print(self:checkZeroTensor(posDocSemVec))
    tableOut['D1-'] = negDocVector1
    tableOut['D2-'] = negDocVector2
    tableOut['D3-'] = negDocVector3
    tableOut['D4-'] = negDocVector4
    return tableOut
end



function InputStream:nextDevPair()
    
    local out = self.devPairs[self.__indexDev]
    
    self.__indexDev = self.__indexDev + 1
    local idList = split(out, " ")
     -- Q vectors
    local queryTxtVec = self.d2vLoader:getQueryTextVector(idList[1])
    local querySemVec = self.semVecLoader:getQueryVector(idList[1])
    if queryTxtVec == nil then
        return nil
    end
    if querySemVec == nil then
        return nil
    end
    local queryVector = torch.cat(queryTxtVec,querySemVec)

    -- D+ vectors
    local posDocTxtVec = self.d2vLoader:getDocTextVector(idList[2],self.dataset)
    local posDocSemVec = self.semVecLoader:getDocVector(idList[2],self.dataset)
    if posDocTxtVec == nil then
        return nil
    end
    if posDocSemVec == nil then
        return nil
    end
    local posDocVector = torch.cat(posDocTxtVec,posDocSemVec)

    -- D- vectors
    local negDocTxtVec1 = self.d2vLoader:getDocTextVector(idList[3],self.dataset)
    local negDocSemVec1 = self.semVecLoader:getDocVector(idList[3],self.dataset)

    local negDocTxtVec2 = self.d2vLoader:getDocTextVector(idList[4],self.dataset)
    local negDocSemVec2 = self.semVecLoader:getDocVector(idList[4],self.dataset)

    local negDocTxtVec3 = self.d2vLoader:getDocTextVector(idList[5],self.dataset)
    local negDocSemVec3 = self.semVecLoader:getDocVector(idList[5],self.dataset)

    local negDocTxtVec4 = self.d2vLoader:getDocTextVector(idList[6],self.dataset)
    local negDocSemVec4 = self.semVecLoader:getDocVector(idList[6],self.dataset)

    local not_nil = 0
    if negDocTxtVec1 ~= nil then
        not_nil = negDocTxtVec1
    elseif negDocTxtVec2 ~= nil then
        not_nil = negDocTxtVec2
    elseif negDocTxtVec3 ~= nil then
        not_nil = negDocTxtVec3
    elseif negDocTxtVec4 ~= nil then
        not_nil = negDocTxtVec4
    end
    if not_nil == 0 then
        return nil
    end

    if negDocTxtVec1 == nil then
         negDocTxtVec1 = not_nil
    end
    if negDocTxtVec2 == nil then
         negDocTxtVec2 = not_nil
    end
    if negDocTxtVec3 == nil then
         negDocTxtVec3 = not_nil
    end
    if negDocTxtVec4 == nil then
         negDocTxtVec4 = not_nil
    end

    not_nil = 0
    if negDocSemVec1 ~= nil then
        not_nil = negDocSemVec1
    elseif negDocSemVec2 ~= nil then
        not_nil = negDocSemVec2
    elseif negDocSemVec3 ~= nil then
        not_nil = negDocSemVec3
    elseif negDocSemVec4 ~= nil then
        not_nil = negDocSemVec4
    end
    if not_nil == 0 then
        return nil
    end

    if negDocSemVec1 == nil then
         negDocSemVec1 = not_nil
    end
    if negDocSemVec2 == nil then
         negDocSemVec2 = not_nil
    end
    if negDocSemVec3 == nil then
         negDocSemVec3 = not_nil
    end
    if negDocSemVec4 == nil then
         negDocSemVec4 = not_nil
    end

    if negDocSemVec1 == nil then
        return nil
    end
    local negDocVector1 = torch.cat(negDocTxtVec1,negDocSemVec1)
    local negDocVector2 = torch.cat(negDocTxtVec2,negDocSemVec2)
    local negDocVector3 =  torch.cat(negDocTxtVec3,negDocSemVec3)
    local negDocVector4 = torch.cat(negDocTxtVec4,negDocSemVec4)

    local tableOut = {}
    tableOut['Q'] = queryVector
    tableOut['D+'] = posDocVector
    tableOut['D1-'] = negDocVector1
    tableOut['D2-'] = negDocVector2
    tableOut['D3-'] = negDocVector3
    tableOut['D4-'] = negDocVector4
    return tableOut
end


function InputStream:nextTestPair()

    local out = self.testPairs[self.__indexTest]

    local idList = split(out, " ")
    -- Q vectors
    local queryTxtVec = self.d2vLoader:getQueryTextVector(idList[1])
    local querySemVec = self.semVecLoader:getQueryVector(idList[1])
     if queryTxtVec == nil then
        return nil
    end
    if querySemVec == nil then
        return nil
    end
    local queryVector =  torch.cat(queryTxtVec,querySemVec)

    -- D+ vectors
    local posDocTxtVec = self.d2vLoader:getDocTextVector(idList[2],self.dataset)
    local posDocSemVec = self.semVecLoader:getDocVector(idList[2],self.dataset)
    if posDocTxtVec == nil then
        return nil
    end
    if posDocSemVec == nil then
        return nil
    end
    local posDocVector = torch.cat(posDocTxtVec,posDocSemVec)
    
    local tableOut = {}
    tableOut['Q'] = queryVector
    tableOut['D+'] = posDocVector
    self.__indexTest = self.__indexTest + 1
    return tableOut
end


function InputStream:shuffle()
    shuffleTable(self.trainPairs)
end

function InputStream:resetTrain()
    self.__indexTrain = 1
    self:shuffle()
end

function InputStream:resetDev()
    self.__indexDev = 1
end


-- =============================================================
-- utils 
function InputStream:getSemanticVectorSize()
    -- body
    return self.semVecLoader:getVectorSize(self.dataset)
    
end

-- =============================================================

-- ======== SHOULD NOT USE THESE FUNCTIONS OUTSIDE =============---
-- ================= a.k.a. PRIVATE METHODS ==================== --


function InputStream:loadTrainPairs()
    local collection_Q_D_path = self.preparedPath.."collection_Q_D/"
    for _, qid in pairs( self.trainSetQuery ) do
        --[[
            qrelsFile for a query id=Q1 contains lines:
            Q1 positiveDocId1 negativeDocId11 negativeDocId12 negativeDocId13 negativeDocId14 negativeDocId15
            Q1 positiveDocId2 negativeDocId21 negativeDocId22 negativeDocId23 negativeDocId24 negativeDocId25
            Q1 positiveDocId3 negativeDocId31 negativeDocId32 negativeDocId33 negativeDocId34 negativeDocId35
            ...
        --]]
        local qrelsFile = collection_Q_D_path..qid..".qrels.neg.csv"

        for line in io.lines( qrelsFile ) do
            table.insert( self.trainPairs, line )
        end
    end
end

function InputStream:loadDevPairs()
    local collection_Q_D_path = self.preparedPath.."collection_Q_D/"
    for _, qid in pairs( self.devSetQuery ) do
        local qrelsFile = collection_Q_D_path..qid..".qrels.neg.csv"
        for line in io.lines( qrelsFile ) do
            table.insert( self.devPairs, line )
        end
    end
end

function InputStream:loadTestPairs()
    local collection_Q_D_path = self.preparedPath.."collection_Q_D/"
    for _, qid in pairs( self.testSetQuery ) do
        local qrelsFile = collection_Q_D_path..qid..".qrels.neg.csv"
        for line in io.lines( qrelsFile ) do
            table.insert( self.testPairs, line )
        end
    end
end

function InputStream:checkZeroTensor(t)
    for i = 1, #t:storage() do
        if t[i] ~= 0 then
            return false
        end
    end
    return true
end

