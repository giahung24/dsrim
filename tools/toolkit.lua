
function has_value (tab, val) -- [Tool] check if val is in tab
    for index, value in ipairs (tab) do
        if value == val then
            return true
        end
    end
    return false
end

function fileExists(file)
    local f = io.open(file, "rb")
    if f then f:close() end
    return f ~= nil
end

function split(inputstr, sep) --only char as sep
    if sep == nil then
        sep = "%s"
    end
    local t={} ; local i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end

function splitByString(pString, pPattern) -- str as sep
   local Table = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pPattern
   local last_end = 1
   local s, e, cap = pString:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
     table.insert(Table,cap)
      end
      last_end = e+1
      s, e, cap = pString:find(fpat, last_end)
   end
   if last_end <= #pString then
      cap = pString:sub(last_end)
      table.insert(Table, cap)
   end
   return Table
end

-- string.split = function(delimiter, text)
--   local list = {}
--   local pos = 1
--   if string.find("", delimiter, 1) then -- this would result in endless loops
--     error("delimiter matches empty string!")
--   end
--   while 1 do
--     local first, last = string.find(text, delimiter, pos)
--     if first then -- found?
--       table.insert(list, string.sub(text, pos, first-1))
--       pos = last+1
--     else
--       table.insert(list, string.sub(text, pos))
--       break
--     end
--   end
--   return list
-- end

function accuracyTensors(x, y)
    local s = x - y
    local count = 0
    local num = torch.numel(x) 
    for i=1, num do
        if s[i] == 0 then
            count = count + 1
        end
    end
    return count/num
end

function string.startWith(String,Start)
   return string.sub(String,1,string.len(Start))==Start
end

function string.endWith(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end

function shuffleTable(table)
    --[[ Shuffle train pool after every epoch ]]--
    math.randomseed( os.time() )
    local rand = math.random 
    local iterations = #table
    local j    
    for i = iterations, 2, -1 do
        j = rand(i)
        table[i], table[j] = table[j], table[i]
    end
end
