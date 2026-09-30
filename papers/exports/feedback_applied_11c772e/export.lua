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
function Table(el)
  local widths
  if #el.colspecs == 8 then
    widths = {.20,.10,.10,.11,.11,.10,.13,.15}
  elseif tostring(el.colspecs[1][1]) == 'AlignLeft' then
    widths = {.20,.10,.14,.14,.14,.14,.14}
  else
    widths = {.10,.12,.16,.16,.16,.15,.15}
  end
  for i,width in ipairs(widths) do el.colspecs[i][2] = width end
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
    local blocks = {pandoc.RawBlock('latex', '\\par\\noindent\\begin{minipage}{\\linewidth}')}
    for _,b in ipairs(el.content) do table.insert(blocks,b) end
    table.insert(blocks,pandoc.RawBlock('latex','\\end{minipage}\\par'))
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
