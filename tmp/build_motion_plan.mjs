// Dependency-free OOXML builder run through the supplied Node runtime.
// Preset: standard_business_brief; header: memo_masthead (without a rule).
// Named overrides: TableText 10pt; Equation Cambria 10pt; Metadata 9pt;
// title/subtitle 23/14pt per memo_masthead; running furniture 9pt muted.
import fs from 'node:fs/promises';
import path from 'node:path';
import { deflateRawSync } from 'node:zlib';

const root = nodeRepl.cwd;
const source = await fs.readFile(path.join(root, 'docs/plans/THREE_STAGE_MOTION_PLAN.md'), 'utf8');
const esc = s => s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main';
const R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships';
const xml = s => '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + s;
const run = s => `<w:r><w:t xml:space="preserve">${esc(s)}</w:t></w:r>`;
const links = [];
function rich(s) {
  return s.split(/(https:\/\/[^\s]+|\*\*[^*]+\*\*|`[^`]+`)/g).map(part => {
    if(part.startsWith('**')&&part.endsWith('**')) return `<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">${esc(part.slice(2,-2))}</w:t></w:r>`;
    if(part.startsWith('`')&&part.endsWith('`')) return `<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="19"/></w:rPr><w:t xml:space="preserve">${esc(part.slice(1,-1))}</w:t></w:r>`;
    if (!part.startsWith('https://')) return run(part);
    const id = 'link'+(links.length+1); links.push([id,part]);
    return `<w:hyperlink r:id="${id}"><w:r><w:rPr><w:rStyle w:val="Hyperlink"/></w:rPr><w:t>${esc(part)}</w:t></w:r></w:hyperlink>`;
  }).join('');
}
const para = (text,style='Normal',extra='') => `<w:p><w:pPr><w:pStyle w:val="${style}"/>${extra}</w:pPr>${rich(text)}</w:p>`;
function table(lines) {
  const rows = lines.filter(l=>!/^\|\s*---/.test(l)).map(l=>l.slice(1,-1).split('|').map(s=>s.trim()));
  const columns=rows[0].length;
  const widths=columns===3?[2150,3450,3760]:columns===4?[2250,3600,1630,1880]:columns===7?[750,2000,650,650,800,800,3710]:[1260,1790,1430,2090,2790];
  if(widths.length!==columns||rows.some(r=>r.length!==columns)) throw Error('Inconsistent table columns');
  if(widths.reduce((a,b)=>a+b,0)!==9360) throw Error('Table width');
  return '<w:tbl><w:tblPr><w:tblW w:w="9360" w:type="dxa"/><w:tblInd w:w="120" w:type="dxa"/><w:tblBorders>'+['top','left','bottom','right','insideH','insideV'].map(s=>`<w:${s} w:val="single" w:sz="4" w:color="CBD2D9"/>`).join('')+'</w:tblBorders><w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:start w:w="120" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/><w:end w:w="120" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>'+widths.map(w=>`<w:gridCol w:w="${w}"/>`).join('')+'</w:tblGrid>'+rows.map((row,i)=>'<w:tr><w:trPr><w:cantSplit/>'+(i===0?'<w:tblHeader/>':'')+'</w:trPr>'+row.map((cell,j)=>`<w:tc><w:tcPr><w:tcW w:w="${widths[j]}" w:type="dxa"/>${i===0?'<w:shd w:fill="F2F4F7"/>':''}<w:vAlign w:val="top"/></w:tcPr>${para(cell,i===0?'TableHeader':'TableText')}</w:tc>`).join('')+'</w:tr>').join('')+'</w:tbl>'+para('','TableAfter');
}
const lines=source.split(/\r?\n/); let body=''; let listId=1; const numberIds=[];
for(let i=0;i<lines.length;i++) {
  const line=lines[i]; if(!line.trim()) continue;
  if(line.startsWith('|')) { const rows=[]; while(i<lines.length&&lines[i].startsWith('|')) rows.push(lines[i++]); i--; body+=table(rows); continue; }
  if(line.startsWith('# ')) body+=para(line.slice(2),'Title');
  else if(line.startsWith('## Three-stage')) body+=para(line.slice(3),'Subtitle');
  else if(line.startsWith('## ')) body+=para(line.slice(3),'Heading1');
  else if(line.startsWith('### ')) body+=para(line.slice(4),'Heading2');
  else if(line.startsWith('25 September')) body+=para(line,'Metadata');
  else if(line.startsWith('F_r =')||line.startsWith('theta[a,u] =')) body+=para(line,'Equation');
  else if(/^\[\d\]/.test(line)) body+=para(line,'Reference');
  else if(line.startsWith('- ')) body+=para(line.slice(2),'ListText','<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>');
  else if(/^\d+\. /.test(line)) {if(!/^\d+\. /.test(lines[i-1]||'')){listId++;numberIds.push(listId);}body+=para(line.replace(/^\d+\. /,''),'ListText',`<w:numPr><w:ilvl w:val="0"/><w:numId w:val="${listId}"/></w:numPr>`);}
  else body+=para(line);
}
function style(id,size,color,before,after,options={}) {
  return `<w:style w:type="paragraph" ${id==='Normal'?'w:default="1"':''} w:styleId="${id}"><w:name w:val="${id}"/>${id==='Normal'?'':'<w:basedOn w:val="Normal"/>'}<w:pPr><w:spacing w:before="${before}" w:after="${after}" w:line="264" w:lineRule="auto"/><w:widowControl/>${options.keep?'<w:keepNext/>':''}${options.outline!==undefined?`<w:outlineLvl w:val="${options.outline}"/>`:''}</w:pPr><w:rPr><w:rFonts w:ascii="${options.font||'Calibri'}" w:hAnsi="${options.font||'Calibri'}"/><w:color w:val="${color}"/><w:sz w:val="${size}"/><w:szCs w:val="${size}"/>${options.bold?'<w:b/>':''}</w:rPr></w:style>`;
}
const styles = xml(`<w:styles xmlns:w="${W}"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/><w:lang w:val="en-NZ"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:before="0" w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>`+
style('Normal',22,'20252B',0,120)+style('Title',46,'0B2545',0,80,{bold:true,keep:true})+style('Subtitle',28,'373737',0,240,{keep:true})+style('Heading1',32,'2E74B5',320,160,{bold:true,keep:true,outline:0})+style('Heading2',26,'2E74B5',240,120,{bold:true,keep:true,outline:1})+style('Heading3',24,'1F4D78',160,80,{bold:true,keep:true,outline:2})+style('Metadata',18,'657080',0,200)+style('TableText',20,'20252B',0,80)+style('TableHeader',20,'20252B',0,80,{bold:true})+style('TableAfter',4,'20252B',0,40)+style('Equation',20,'1F4D78',80,160,{font:'Cambria'})+style('Reference',20,'20252B',80,80)+style('Furniture',18,'657080',0,0)+'<w:style w:type="character" w:styleId="Hyperlink"><w:name w:val="Hyperlink"/><w:rPr><w:color w:val="2E74B5"/><w:u w:val="single"/></w:rPr></w:style></w:styles>');
const listStyle='<w:style w:type="paragraph" w:styleId="ListText"><w:name w:val="ListText"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="0" w:after="160" w:line="280" w:lineRule="auto"/><w:widowControl/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:color w:val="20252B"/><w:sz w:val="22"/></w:rPr></w:style>';
const numbering=xml(`<w:numbering xmlns:w="${W}">`+[ ['0','bullet','•'],['1','decimal','%1.'] ].map(([id,format,label])=>`<w:abstractNum w:abstractNumId="${id}"><w:multiLevelType w:val="singleLevel"/><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="${format}"/><w:lvlText w:val="${label}"/><w:lvlJc w:val="left"/><w:pPr><w:tabs><w:tab w:val="num" w:pos="720"/></w:tabs><w:ind w:left="720" w:hanging="360"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/></w:rPr></w:lvl></w:abstractNum>`).join('')+'<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>'+numberIds.map(id=>`<w:num w:numId="${id}"><w:abstractNumId w:val="1"/><w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride></w:num>`).join('')+'</w:numbering>');
const document=xml(`<w:document xmlns:w="${W}" xmlns:r="${R}"><w:body>${body}<w:sectPr><w:headerReference w:type="default" r:id="header"/><w:footerReference w:type="default" r:id="footer"/><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr></w:body></w:document>`);
const relns='http://schemas.openxmlformats.org/package/2006/relationships';
const rel=(id,type,target,mode='')=>`<Relationship Id="${id}" Type="${R}/${type}" Target="${esc(target)}"${mode?' TargetMode="External"':''}/>`;
const entries = {
  '[Content_Types].xml':xml('<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'+[['document','document.main'],['styles','styles'],['numbering','numbering'],['header1','header'],['footer1','footer'],['settings','settings']].map(([name,type])=>`<Override PartName="/word/${name}.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.${type}+xml"/>`).join('')+'</Types>'),
  '_rels/.rels':xml(`<Relationships xmlns="${relns}">${rel('document','officeDocument','word/document.xml')}</Relationships>`),
  'word/document.xml':document,
  'word/styles.xml':styles.replace('</w:styles>',listStyle+'</w:styles>'),
  'word/numbering.xml':numbering,
  'word/settings.xml':xml(`<w:settings xmlns:w="${W}"><w:zoom w:percent="100"/><w:defaultTabStop w:val="720"/><w:updateFields w:val="true"/></w:settings>`),
  'word/_rels/document.xml.rels':xml(`<Relationships xmlns="${relns}">${rel('styles','styles','styles.xml')}${rel('numbering','numbering','numbering.xml')}${rel('settings','settings','settings.xml')}${rel('header','header','header1.xml')}${rel('footer','footer','footer1.xml')}${links.map(([id,url])=>rel(id,'hyperlink',url,true)).join('')}</Relationships>`),
  'word/header1.xml':xml(`<w:hdr xmlns:w="${W}">${para('DISCRETE MOTION PROJECT  /  THREE-STAGE PLAN','Furniture')}</w:hdr>`),
  'word/footer1.xml':xml(`<w:ftr xmlns:w="${W}"><w:p><w:pPr><w:pStyle w:val="Furniture"/><w:jc w:val="right"/></w:pPr>${run('25 September 2026  |  Page ')}<w:fldSimple w:instr="PAGE"><w:r><w:t>1</w:t></w:r></w:fldSimple></w:p></w:ftr>`),
};
// ZIP (deflate, fixed timestamp), keeping artifact creation independent of installs.
function crc32(bytes) { let c=0xffffffff; for(const byte of bytes) { c^=byte; for(let i=0;i<8;i++) c=(c>>>1)^((c&1)?0xedb88320:0); } return (c^0xffffffff)>>>0; }
let offset=0; const chunks=[], central=[];
for(const [name,content] of Object.entries(entries)) {
  const nb=Buffer.from(name), raw=Buffer.from(content), data=deflateRawSync(raw), crc=crc32(raw);
  const h=Buffer.alloc(30); h.writeUInt32LE(0x04034b50); h.writeUInt16LE(20,4); h.writeUInt16LE(8,8); h.writeUInt16LE(33,12); h.writeUInt32LE(crc,14); h.writeUInt32LE(data.length,18); h.writeUInt32LE(raw.length,22); h.writeUInt16LE(nb.length,26);
  const c=Buffer.alloc(46); c.writeUInt32LE(0x02014b50); c.writeUInt16LE(20,4); c.writeUInt16LE(20,6); c.writeUInt16LE(8,10); c.writeUInt16LE(33,14); c.writeUInt32LE(crc,16); c.writeUInt32LE(data.length,20); c.writeUInt32LE(raw.length,24); c.writeUInt16LE(nb.length,28); c.writeUInt32LE(offset,42);
  chunks.push(h,nb,data); central.push(c,nb); offset+=h.length+nb.length+data.length;
}
const cd=Buffer.concat(central), end=Buffer.alloc(22); end.writeUInt32LE(0x06054b50); end.writeUInt16LE(Object.keys(entries).length,8); end.writeUInt16LE(Object.keys(entries).length,10); end.writeUInt32LE(cd.length,12); end.writeUInt32LE(offset,16);
const out=path.join(root,'docs/plans/THREE_STAGE_MOTION_PLAN.docx');
await fs.writeFile(out,Buffer.concat([...chunks,cd,end]));
await fs.mkdir(path.join(root,'tmp/motion_plan_qa'),{recursive:true});
await fs.writeFile(path.join(root,'tmp/motion_plan_qa/build_audit.json'),JSON.stringify({preset:'standard_business_brief',header:'memo_masthead',paragraphs:(document.match(/<w:p>/g)||[]).length,tables:(document.match(/<w:tbl>/g)||[]).length,words:source.split(/\s+/).length,parts:Object.keys(entries),visual_qa:'pending'},null,2));
nodeRepl.write({out,bytes:(await fs.stat(out)).size,words:source.split(/\s+/).length});
