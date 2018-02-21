local Doc2VecLoader = torch.class("Doc2VecLoader")

--[[    self.normalizer = nn.Normalize(1)

    
]]--
function Doc2VecLoader:__init(pretraitPath)
    --[[ pretraitPath where we find doc_in_qrels,query,doc_desc,query_desc ]]--
    if not string.endWith(pretraitPath, '/') then
        pretraitPath = pretraitPath.."/"
    end
    self.pretraitPath = pretraitPath
end

function Doc2VecLoader:zNormalize(vector)
    if vector:std() == 0 then
        return (vector - vector)
    end
    return (vector - vector:mean()):div(vector:std())
end

function Doc2VecLoader:getDocTextVector(docId , dataset)
    local vecTable = {}
    local d2vVecFile = ""
    if dataset == 'gov' then
        d2vVecFile = self:makePathGov(self.pretraitPath.."vectors/doc2vec/doc_text", docId, '.d2v.vec')
    else -- 'pmc'
        d2vVecFile = self:makePathPmc(self.pretraitPath.."vectors/doc2vec/doc_text", docId, '.d2v.vec')
    end
    -- if doc vector not exist
    if not fileExists(d2vVecFile) then 
        -- local output = torch.Tensor(100):random(100)
        -- output = output:div(torch.norm(output))
        -- return output
        return nil
    end
    for line in io.lines(d2vVecFile) do
        table.insert( vecTable, line)
    end
    local output = self:zNormalize(torch.Tensor(vecTable))
    return output
end

function Doc2VecLoader:getDocDescVector(docId , dataset)
    local vecTable = {}
    local d2vVecFile = ""
    if dataset == 'gov' then
        d2vVecFile = self:makePathGov(self.pretraitPath.."vectors/doc2vec/doc_desc", docId, '.d2v.vec')
    else -- 'pmc'
        d2vVecFile = self:makePathPmc(self.pretraitPath.."vectors/doc2vec/doc_desc", docId, '.d2v.vec')
    end
    -- if doc vector not exist
    if not fileExists(d2vVecFile) then 
        -- local output = torch.Tensor(100):random(100)
        -- output = output:div(torch.norm(output))
        -- return output
        return nil
    end
    for line in io.lines(d2vVecFile) do
        table.insert( vecTable, line)
    end
    local output = self:zNormalize(torch.Tensor(vecTable))
    return output
end

-- QUERY

function Doc2VecLoader:getQueryTextVector(queryId )
    local vecTable = {}
    local d2vVecFile = self.pretraitPath .. "vectors/doc2vec/query_text/" .. queryId .. '.d2v.vec'
    -- if doc vector not exist
    if not fileExists(d2vVecFile) then 
        -- local output = torch.Tensor(100):random(100)
        -- output = output:div(torch.norm(output))
        -- return output
        return nil
    end
    for line in io.lines(d2vVecFile) do
        table.insert( vecTable, line)
    end
    return self:zNormalize(torch.Tensor(vecTable))
end

function Doc2VecLoader:getQueryDescVector(queryId)
    local vecTable = {}
    local d2vVecFile = self.pretraitPath .. "vectors/doc2vec/query_desc/" .. queryId .. '.d2v.vec'
    -- if doc vector not exist
    if not fileExists(d2vVecFile) then 
        -- local output = torch.Tensor(100):random(100)
        -- output = output:div(torch.norm(output))
        -- return output
        return nil
    end
    for line in io.lines(d2vVecFile) do
        table.insert( vecTable, line)
    end
    return self:zNormalize(torch.Tensor(vecTable))
end



function Doc2VecLoader:makePathGov( root_dir, doc_id, file_ext )
    if not string.endWith(root_dir, '/') then
        root_dir = root_dir.."/"
    end
    local path = root_dir .. doc_id:sub(1,5) .. "/" .. doc_id:sub(7,8) .. "/" .. doc_id .. file_ext
    return path
end


function Doc2VecLoader:makePathPmc( root_dir, doc_id, file_ext )
    if not string.endWith(root_dir, '/') then
        root_dir = root_dir.."/"
    end

    local first = tostring(self:javaHashCode(doc_id)):sub(-4,-3)
    local last = tostring(self:javaHashCode(doc_id)):sub(-2)
    if string.startWith(first, '0') then
        first = first:sub(2)
    end
    if string.startWith(last, '0') then
        last = last:sub(2)
    end
    local path = root_dir .. first .. "/" .. last .. "/" .. doc_id .. file_ext
    return path
end


function Doc2VecLoader:javaHashCode( s )
    local h = 0
    for c in s:gmatch"." do
        h = bit.band((31 * h + c:byte()) , 0xFFFFFFFF)
    end
    return (bit.band((h + 0x80000000) , 0xFFFFFFFF) - 0x80000000)
end