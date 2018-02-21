--[[
@edit by Jean
Copy the MarginRankingCriterion, adapted to max(0, margin - y*(4*x1 - x2 - x3 - x4 - x5))
--]]

local MarginQuadRankingCriterion, parent = torch.class('nn.MarginQuadRankingCriterion', 'nn.Criterion')

function MarginQuadRankingCriterion:__init(margin)
   parent.__init(self)
   margin=margin or 1
   self.margin = margin
   self.gradInput = {torch.Tensor(1), torch.Tensor(1), torch.Tensor(1), torch.Tensor(1), torch.Tensor(1)}
   self.sizeAverage = true
end

function MarginQuadRankingCriterion:updateOutput(input, y)
    if torch.type(y) == 'number' then -- non-batch mode
      self.output = math.max(0, -y * (4 * input[1] - input[2] - input[3] - input[4] - input[5]) + self.margin)
   else
      self._output = self._output or input[1]:clone()
      self._output:resizeAs(input[1])
      self._output:copy(input[1]:mul(4)) -- 4*x[1]

      self._output:add(-1, input[2])  -- 4*x[1] - x[2]
      self._output:add(-1, input[3]) 
      self._output:add(-1, input[4]) 
      self._output:add(-1, input[5])
      -- print( self._output )
      -- print(y)
      self._output:mul(-1):cmul(y) -- -y* (4x1 - x[2] ... - x[5])
      self._output:add(self.margin) -- + margin

      self._output:cmax(0)

      self.output = self._output:sum()

      if self.sizeAverage then
         self.output = self.output/y:size(1)
      end
   end

   return self.output
end

function MarginQuadRankingCriterion:updateGradInput(input, y)
    if torch.type(y) == 'number' then -- non-batch mode
      local dist = -y * (4 * input[1] - input[2] - input[3] - input[4] - input[5]) + self.margin
      self.gradInput = input:clone():zero()
      if dist >= 0 then
         self.gradInput[1] = -4 * y
         self.gradInput[2] = y
         self.gradInput[3] = y
         self.gradInput[4] = y
         self.gradInput[5] = y

      end
   else
      self.dist = self.dist or input[1].new()
      self.dist = self.dist:resizeAs(input[1]):copy(input[1]:mul(4))
      local dist = self.dist

      dist:add(-1, input[2])
      dist:add(-1, input[3])
      dist:add(-1, input[4])
      dist:add(-1, input[5])

      dist:mul(-1):cmul(y)
      dist:add(self.margin)

      self.mask = self.mask or input[1].new()
      self.mask = self.mask:resizeAs(input[1]):copy(dist)
      local mask = self.mask

      mask:ge(dist, 0)

      self.gradInput[1]:resize(dist:size())
      self.gradInput[2]:resize(dist:size())
      self.gradInput[3]:resize(dist:size())
      self.gradInput[4]:resize(dist:size())
      self.gradInput[5]:resize(dist:size())

      self.gradInput[1]:copy(mask)
      self.gradInput[1]:mul(-4):cmul(y)
      self.gradInput[2]:copy(mask)
      self.gradInput[2]:cmul(y)
      self.gradInput[3]:copy(mask)
      self.gradInput[3]:cmul(y)
      self.gradInput[4]:copy(mask)
      self.gradInput[4]:cmul(y)
      self.gradInput[5]:copy(mask)
      self.gradInput[5]:cmul(y)

      if self.sizeAverage then
         self.gradInput[1]:div(y:size(1))
         self.gradInput[2]:div(y:size(1))
         self.gradInput[3]:div(y:size(1))
         self.gradInput[4]:div(y:size(1))
         self.gradInput[5]:div(y:size(1))
      end

   end
   return self.gradInput
end
