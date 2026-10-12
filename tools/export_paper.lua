function RawInline(el)
  if el.format == 'html' then
    local id = el.text:match('<a id="([^"]+)"')
    if id then return pandoc.RawInline('latex', '\\phantomsection\\label{' .. id .. '}') end
    if el.text == '</a>' then return {} end
  end
end
function Image(el)
  el.attributes.width = '6.8in'
  return el
end
function Header(el)
  if el.level == 2 or el.level == 3 then
    return {pandoc.RawBlock('latex', '\\FloatBarrier\\Needspace{8\\baselineskip}'), el}
  end
end
-- Printed width of a cell's text: a Hangul syllable sets about twice as wide as
-- a Latin letter or digit.
local function text_width(cell)
  local s, w = pandoc.utils.stringify(cell.contents), 0
  for _, c in utf8.codes(s) do
    if c >= 0xAC00 and c <= 0xD7A3 then w = w + 2 else w = w + 1 end
  end
  return w
end

-- Column widths in proportion to the widest body cell. A header counts at half
-- its width because it may wrap onto two lines, and the body may not.
function Table(el)
  local n = #el.colspecs
  local need = {}
  for i = 1, n do need[i] = 4 end
  local function visit(rows, weight)
    for _, row in ipairs(rows) do
      for i, cell in ipairs(row.cells) do
        if i <= n then need[i] = math.max(need[i], math.ceil(text_width(cell) * weight)) end
      end
    end
  end
  visit(el.head.rows, 0.5)
  for _, body in ipairs(el.bodies) do visit(body.body, 1.0) end
  local total = 0
  for i = 1, n do total = total + need[i] end
  for i = 1, n do el.colspecs[i][2] = need[i] / total end
  return el
end
function Blocks(blocks)
  local out = {}
  for i,block in ipairs(blocks) do
    local next = blocks[i+1]
    if block.t == 'Para' and next and next.t == 'Para'
       and #next.content == 1 and next.content[1].t == 'Math'
       and next.content[1].mathtype == 'DisplayMath' then
      local lines = math.max(6, math.ceil(utf8.len(pandoc.utils.stringify(block))/80)+4)
      table.insert(out,pandoc.RawBlock('latex','\\Needspace{' .. lines .. '\\baselineskip}'))
    end
    table.insert(out,block)
  end
  return out
end
function BlockQuote(el)
  if pandoc.utils.stringify(el):match('^Algorithm 1') then
    local blocks = {pandoc.RawBlock('latex', '\\begin{PaperAlgorithm}')}
    for _,b in ipairs(el.content) do
      if b.t == 'OrderedList' then
        for number,item in ipairs(b.content) do
          -- Four leading nonbreaking spaces in Markdown mark one nesting level.
          -- Keep the content in the manuscript; only its layout belongs here.
          assert(#item == 1 and (item[1].t == 'Plain' or item[1].t == 'Para'),
                 'Algorithm lines must contain one paragraph')
          local inlines = item[1].content:clone()
          local spaces = 0
          while inlines[1] and inlines[1].t == 'Str'
                and inlines[1].text:sub(1, 2) == '\194\160' do
            local text = inlines[1].text
            while text:sub(1, 2) == '\194\160' do
              spaces = spaces + 1
              text = text:sub(3)
            end
            if text == '' then inlines:remove(1) else inlines[1].text = text end
          end
          assert(spaces % 4 == 0, 'Algorithm indentation must use groups of four spaces')
          if inlines[1] and inlines[1].t == 'Space' then inlines:remove(1) end
          local latex = pandoc.write(pandoc.Pandoc({pandoc.Plain(inlines)}), 'latex')
          table.insert(blocks, pandoc.RawBlock('latex',
            '\\PaperAlgorithmLine{' .. (b.start + number - 1) .. '}{'
            .. (1.25 * spaces / 4) .. '}{' .. latex .. '}'))
        end
      else
        table.insert(blocks,b)
        if pandoc.utils.stringify(b):match('^Algorithm 1') then
          table.insert(blocks, pandoc.RawBlock('latex',
            '\\par\\vspace{3pt}\\hrule height 0.4pt\\vspace{4pt}'))
        end
      end
    end
    table.insert(blocks,pandoc.RawBlock('latex','\\end{PaperAlgorithm}'))
    return blocks
  end
end
function Para(el)
  local index = nil
  for i,v in ipairs(el.content) do if v.t == 'Image' then index=i; break end end
  if not index then
    if pandoc.utils.stringify(el):match('^표 %d %[') then
      return {pandoc.RawBlock('latex','\\Needspace{16\\baselineskip}'),el}
    end
    return nil
  end
  local before,after = {},{}
  for i=1,index-1 do table.insert(before,el.content[i]) end
  for i=index+1,#el.content do
    if #after>0 or (el.content[i].t~='SoftBreak' and el.content[i].t~='Space') then table.insert(after,el.content[i]) end
  end
  local blocks={pandoc.RawBlock('latex','\\begin{figure}[!htbp]\\centering')}
  if #before>0 then table.insert(blocks,pandoc.Plain(before)) end
  table.insert(blocks,pandoc.Plain({el.content[index]}))
  table.insert(blocks,pandoc.RawBlock('latex','\\par\\vspace{4pt}{\\small\\raggedright'))
  if #after>0 then table.insert(blocks,pandoc.Plain(after)) end
  table.insert(blocks,pandoc.RawBlock('latex','\\par}\\end{figure}'))
  return blocks
end
