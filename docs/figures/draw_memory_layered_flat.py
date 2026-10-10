"""Four horizontal flat tiers, top-down index, explicit within-tier hyperedges."""
from pathlib import Path
helper=Path(__file__).with_name('draw_memory_reference_style.py')
exec(helper.read_text(encoding='utf-8').split('# One compact enclosing frame')[0],globals())
W,H=2240,1440
ROOT=Path(__file__).resolve().parent


def plane(top,bottom,fill):
    # Layer planes are display guides, not hyperedges.
    background.append(f'<path d="M 290 {top} H 1775 L 1735 {bottom} H 250 Z" '
                      f'fill="{fill}" fill-opacity="0.38" stroke="#c9d0d2" stroke-width="0.85"/>')


def badge(x,y,value):
    marks.append(f'<circle cx="{x}" cy="{y}" r="16" fill="white" stroke="#8a9c9b" stroke-width="1.2"/>')
    text(x,y+6,str(value),18,weight='bold',color='#69817a')


def band_label(y,number,title,detail):
    badge(124,y-42,number)
    text(124,y,title,20,weight='bold')
    text(124,y+31,detail,16,color='#78898e')


def experience(cx,title):
    box(cx-103,367,206,54,title,'记录索引／语义表示',size=21)


def transition(a,b):
    mx=(a+b)/2
    blob(f'M {a-12} 777 C {a+25} 760 {b-25} 760 {b+12} 777 '
         f'C {b+26} 801 {mx+35} 850 {mx} 847 C {mx-35} 850 {a-26} 801 {a-12} 777 Z',
         COL['pink'],'#b9a4af',.61)
    line((a+14,785),(b-15,785),color=INK,width=1.2,arrow=True)
    node(mx,824,'先于',label=(mx,846),rx=9,ry=5)
    line((a,791),(mx,824),color='#b1a3aa',width=.8)
    line((b,791),(mx,824),color='#b1a3aa',width=.8)


def process(cx,name,phases):
    blob(f'M {cx-159} 675 C {cx-80} 656 {cx+91} 656 {cx+159} 675 '
         f'C {cx+210} 706 {cx+239} 816 {cx+184} 854 '
         f'C {cx+112} 892 {cx-134} 890 {cx-197} 849 '
         f'C {cx-235} 810 {cx-211} 702 {cx-159} 675 Z',COL['yellow'],'#b7aa8c',.58)
    node(cx,699,name,label=(cx,679),rx=14,ry=8)
    text(cx,735,'独立事件实例／过程超边',16,color='#8b8068')
    for dx,phase in zip((-175,0,175),phases):
        line((cx,707),(cx+dx,785),color='#a8aa9e',width=.85)
        node(cx+dx,785,phase,label=(cx+dx,766),rx=12,ry=7)
    transition(cx-175,cx)
    transition(cx,cx+175)
    text(cx,891,'转移超边：保留源阶段 → 目标阶段',16,color='#8c7680')


def flow(y,title,detail,fill):
    box(1844,y,354,72,title,detail,fill=fill,size=20)


rect(18,18,2204,1404,fill='#fff',stroke='#53636c',r=14)
rect(1820,133,391,1220,fill='#f7faf6',stroke='none',r=0)
path('M 1813 133 V 1371',color='#98a6a8',width=1,dash='7 6')
text(1010,61,'分层扁平的时空语义超图记忆',30,weight='bold')
text(1010,102,'主题索引 → 历史经历 → 具体事件与阶段 → 事实证据',19,color='#687d85')

for top,bottom,fill in [(153,301,'#f1edf8'),(345,613,'#eef5f1'),
                        (657,973,'#faf5e7'),(1018,1338,'#edf3f7')]:
    plane(top,bottom,fill)
for y,number,title,detail in [(229,1,'主题索引层','任务入口与聚合'),(492,2,'历史经历层','五要素联合绑定'),
                             (821,3,'事件与阶段层','各自的演化过程'),(1200,4,'事实证据层','观测、出处与拓扑')]:
    band_label(y,number,title,detail)
for start,end in [(303,342),(615,654),(975,1015)]:
    line((219,start),(219,end),color='#91a1a6',width=1.1,arrow=True)

# Tier 1 is flat: one task key, with a separate auxiliary semantic index.
blob('M 578 182 C 741 168 1160 168 1220 183 C 1257 203 1224 259 1156 272 '
     'C 1000 284 638 280 588 261 C 552 244 545 199 578 182 Z',COL['purple'],'#a8a1bb',.43)
node(946,209,'拥堵分析主题',label=(946,188),rx=15,ry=8)
text(946,249,'按任务键聚合历史经历',17,color='#777087')
box(1340,190,306,68,'语义辅助索引','向量表示／离线 TF-IDF',fill='#fff',size=20)
text(1493,282,'编码问题与经历内容；不替代条件检查',15,color='#7e8b91')
for cx in (624,1330):
    path(f'M 946 219 C 946 287 {cx} 289 {cx} 362',color='#8d96a0',dash='5 4',width=1.25,arrow=True)
text(946,325,'同一任务主题下的不同历史经历',17,color='#71868a')

# Tier 2: wide shallow episode envelopes, overlapping at exactly three shown nodes.
blob('M 402 365 C 483 344 992 340 1152 359 C 1185 378 1198 521 1152 553 '
     'C 1056 597 456 598 390 559 C 351 535 350 395 402 365 Z',COL['green'],'#9cab9c',.53)
blob('M 791 366 C 955 343 1483 341 1590 362 C 1640 386 1637 531 1594 560 '
     'C 1497 600 884 593 789 555 C 750 529 753 392 791 366 Z',COL['blue'],'#96adbb',.53)
experience(624,'历史经验甲')
experience(1330,'历史经验乙')
for x,kind,value in [(810,'时间','早峰适用窗口'),(967,'地点','路段甲'),(1124,'事件','拥堵事件类别')]:
    node(x,461,kind,label=(x,440),color='#aa7e75',rx=12,ry=7)
    text(x,490,value,16,color='#837b70')
    for cx in (624,1330):
        line((cx,422),(x,461),color='#9daeb0',width=.85)
for cx,attrs in [(624,[(478,'行动甲','核验观测'),(672,'结果甲','核验完成')]),
                 (1330,[(1274,'行动乙','审查一致性'),(1480,'结果乙','审查完成')])]:
    for x,kind,value in attrs:
        node(x,534,kind,label=(x,514),rx=11,ry=6)
        text(x,562,value,16)
        line((cx,422),(x,534),color='#9caeae',width=.85)
text(541,593,'经历超边甲',16,color='#6e8570')
text(1414,593,'经历超边乙',16,color='#6d8797')
text(967,593,'共享时空与事件条件，各自保留行动和结果',16,color='#778a83')
for cx in (624,1330):
    # Layer-to-layer mapping is explicit and does not collapse two cases.
    path(f'M {cx} 423 V 648 Q {cx} 663 {cx} 684',color='#718f90',dash='4 5',width=1.15,arrow=True)
text(969,640,'经验甲 → 事件甲；经验乙 → 事件乙',17,color='#6e8585')

# Tier 3: separate process groups, horizontal transitions, a genuinely shared state.
process(624,'具体事件甲',('增长','恢复','稳定'))
process(1330,'具体事件乙',('增长','持续','稳定'))
node(967,924,'共享前提状态',label=(967,902),color='#aa7e75',rx=12,ry=7)
text(967,952,'排队增长；当前查询满足',16,color='#837b70')
path('M 449 793 C 449 924 742 924 953 924',color='#8ca29b',dash='4 4',width=.95,arrow=True)
path('M 1155 793 C 1155 921 1054 924 981 924',color='#8ca29b',dash='4 4',width=.95,arrow=True)
for cx in (624,1330):
    text(cx,948,'阶段证据：状态、阶段时间、行动与观测',15,color='#73877e')

# Tier 4: observations share one entity, and have distinct times and sources.
blob('M 426 1053 C 528 1030 915 1052 1000 1109 C 1060 1172 981 1244 899 1257 '
     'C 728 1292 420 1267 366 1218 C 321 1163 346 1081 426 1053 Z',COL['blue'],'#94adbc',.54)
blob('M 1520 1053 C 1418 1030 1031 1052 946 1109 C 886 1172 965 1244 1047 1257 '
     'C 1218 1292 1526 1267 1580 1218 C 1625 1163 1600 1081 1520 1053 Z',COL['green'],'#9bad9c',.54)
blob('M 940 1174 C 1010 1157 1330 1220 1358 1254 C 1381 1286 1235 1309 1113 1297 '
     'C 1007 1284 917 1202 940 1174 Z',COL['yellow'],'#b8aa86',.55)
node(550,1091,'历史观测甲',label=(550,1067))
node(1396,1091,'历史观测乙',label=(1396,1067))
node(433,1201,'观测时刻甲',label=(433,1229),rx=11,ry=6)
node(677,1201,'门架来源',label=(677,1229),rx=11,ry=6)
node(1269,1201,'事件记录来源',label=(1269,1229),rx=11,ry=6)
node(1513,1201,'观测时刻乙',label=(1513,1229),rx=11,ry=6)
node(973,1191,'共享实体：路段甲',label=(973,1168),color='#aa7e75',rx=13,ry=7)
for obs,targets in [((550,1091),[(433,1201),(677,1201),(973,1191)]),
                    ((1396,1091),[(1269,1201),(1513,1201),(973,1191)])]:
    for target in targets:
        line(obs,target,color='#9caeac',width=.9)
node(1160,1270,'甲在乙上游',label=(1160,1296),rx=10,ry=6)
node(1312,1270,'路段乙',label=(1312,1296),rx=10,ry=6)
for target in [(973,1191),(1312,1270)]:
    line((1160,1270),target,color='#aaa087',width=.9)
text(540,1305,'观测超边甲',16,color='#6a8492')
text(1453,1305,'观测超边乙',16,color='#708872')
for start,target in [((449,792),(550,1081)),((1155,792),(1396,1081))]:
    path(f'M {start[0]} {start[1]} C {start[0]-50} 978 {target[0]-15} 990 {target[0]} {target[1]}',
         color='#7d9a92',dash='4 5',width=1.1,arrow=True)
text(539,1000,'引用本事件的观测证据',15,color='#7c928d')
text(1395,1000,'引用本事件的观测证据',15,color='#7c928d')

# Retrieval pipeline accurately mirrors the implemented sequence.
text(2020,174,'联合检索与复用',23,weight='bold')
for y,title,detail,fill in [
    (224,'当前查询与任务定位','问题、任务键、时空与状态',COL['purple']),
    (382,'当前条件与证据过滤','阶段、时效、来源对齐与冲突',COL['yellow']),
    (540,'语义召回','向量余弦；离线使用 TF-IDF',COL['purple']),
    (698,'经历与阶段绑定检查','核对同一经历的超边与转移',COL['yellow']),
    (856,'受约束的超边扩展','同任务族；至少两类共享条件',COL['green']),
    (1014,'联合排序与候选集合','相关性＋条件覆盖＋联合支持',COL['green']),
    (1172,'Jev 单次复用门控','当前查询＋候选＋绑定与证据',COL['blue'])]:
    flow(y,title,detail,fill)
for start,end in [(296,382),(454,540),(612,698),(770,856),(928,1014),(1086,1172)]:
    line((2021,start+5),(2021,end-5),color='#7a9288',width=1.3,arrow=True)
line((2021,1249),(2021,1290),color='#7a9288',width=1.2,arrow=True)
text(2021,1323,'复用／综合／停止',22,weight='bold',color='#587181')
path('M 1600 547 H 1793 V 892 H 1839',color='#7a9990',dash='5 5',width=1.1,arrow=True)
text(1677,526,'共享条件匹配',15,color='#7b928c')

node(325,1383,rx=10,ry=6);text(345,1389,'节点',15,anchor='start')
blob('M 419 1371 C 437 1367 458 1371 466 1385 C 465 1399 422 1401 414 1389 C 408 1381 410 1375 419 1371 Z',COL['blue'])
text(478,1389,'闭合包络＝超边',15,anchor='start')
line((659,1383),(696,1383),dash='4 3');text(708,1389,'跨层索引／证据引用',15,anchor='start')
text(1263,1389,'浅色平面＝展示层；不是四个数据库；节点可跨层引用；阶段先后不代表因果。',15,color='#75858b')
defs='''<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 Z" fill="#667881"/></marker></defs>'''
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'+defs+\
    '<style>text{font-family:"Microsoft YaHei","SimHei",sans-serif}path,ellipse,rect{vector-effect:non-scaling-stroke}</style>'+\
    '<rect width="100%" height="100%" fill="white"/>'+''.join(background+edges+marks+labels)+'</svg>'
target=ROOT/'hypergraph-memory-layered-flat.svg'
target.write_text(svg,encoding='utf-8')
print(target)
