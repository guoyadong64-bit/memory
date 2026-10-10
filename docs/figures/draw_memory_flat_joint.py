"""Flat native SVG: overlapping episodes, shared query anchors, independent cases."""
from pathlib import Path
helper = Path(__file__).with_name('draw_memory_reference_style.py')
exec(helper.read_text(encoding='utf-8').split('# One compact enclosing frame')[0],globals())
W,H=2280,1240
ROOT=Path(__file__).resolve().parent


def anchor(x,y,title,detail):
    node(x,y,title,label=(x,y-22),color='#b47571',rx=14,ry=8)
    text(x,y+30,detail,16,color='#88736f')


def pipeline(y,title,detail,fill=COL['green']):
    box(1856,y,359,70,title,detail,fill=fill,size=20)


rect(18,18,2244,1204,fill='#fff',stroke='#52616a',r=14)
rect(29,131,282,1029,fill='#f8f9f8',stroke='none',r=0)
rect(1820,131,431,1029,fill='#f6faf5',stroke='none',r=0)
for x in (318,1814):
    path(f'M {x} 131 V 1183',color='#9aa6aa',dash='7 6',width=1)
text(1140,62,'时空语义联合检索与证据约束的超图记忆',30,weight='bold')
text(1140,101,'扁平展示：同一经历联合绑定多类要素；独立事件保留阶段；共享条件支持受约束召回',18,color='#72818a')

# Query context is transient input, explicitly outside persistent memory.
text(170,171,'当前查询',23,weight='bold')
box(48,229,246,70,'问题／任务','核验监测；可提供任务键',fill=COL['purple'],size=20)
box(48,351,246,77,'时空条件','早峰时段 · 路段甲',fill=COL['yellow'],size=20)
box(48,482,246,77,'事件／状态','拥堵 · 增长阶段',fill=COL['pink'],size=20)
box(48,616,246,81,'当前观测','来源一＋来源二；保留时刻',fill=COL['blue'],size=20)
text(170,774,'研究机制',22,weight='bold')
for y,title,details in [(833,'同一经历内的条件绑定',['五类要素联合保存','阶段链与结果出处可追溯']),
                        (972,'受约束的共享条件召回',['至少两类兼容节点共享','每条经历独立满足当前条件'])]:
    text(170,y,title,18,weight='bold',color='#6b817b')
    for i,value in enumerate(details):
        text(170,y+36+i*29,value,16)
text(170,1128,'查询观测不自动成为长期记忆',15,color='#7a888f')

# Two episodes intersect at actual shared context nodes on a single plane.
blob('M 671 276 C 818 230 1119 321 1153 454 C 1197 606 1180 938 1114 1073 '
     'C 1057 1145 529 1144 462 1042 C 390 930 405 481 458 372 '
     'C 500 308 575 281 671 276 Z',COL['green'],'#9bac9d',.45)
blob('M 1366 276 C 1219 230 918 321 884 454 C 840 606 857 938 923 1073 '
     'C 980 1145 1508 1144 1575 1042 C 1647 930 1632 481 1579 372 '
     'C 1537 308 1462 281 1366 276 Z',COL['blue'],'#98adbb',.45)
node(1018,212,'任务主题索引',label=(1018,187),rx=15,ry=8)
text(1018,244,'按任务键组织经验族',17,color='#7c738a')
for cx in (674,1362):
    line((1018,221),(cx,278),dash='5 4',color='#9d99aa',width=1,arrow=True)
    box(cx-102,282,204,61,'经验甲' if cx==674 else '经验乙',
        '问题与经历内容的语义索引',fill='#fff',size=21)
text(672,373,'经历超边甲',19,weight='bold',color='#718771')
text(1362,373,'经历超边乙',19,weight='bold',color='#6f8799')

anchor(1018,423,'时间','共享早峰适用窗口')
anchor(1018,552,'地点','共享路段甲')
anchor(1018,678,'事件类别','同属拥堵事件')
anchor(1018,820,'前提状态','排队增长；当前相容')
for cx in (674,1362):
    for target in [(1018,423),(1018,552),(1018,678),(1018,820)]:
        path(f'M {cx} 346 C {cx} 392 {target[0]+(-75 if cx==674 else 75)} {target[1]-22} {target[0]} {target[1]}',
             color='#acb7b5',width=.85)
for x,title,detail in [(501,'行动甲','核验监测'),(1536,'行动乙','审查一致性')]:
    node(x,469,title,label=(x,445))
    text(x,499,detail,16)
for x,title,detail in [(497,'结果甲','核验完成'),(1540,'结果乙','审查完成')]:
    node(x,632,title,label=(x,607))
    text(x,662,detail,16)
for cx,action,outcome in [(674,(501,469),(497,632)),(1362,(1536,469),(1540,632))]:
    line((cx,346),action,color='#a1afae',width=.85)
    line((cx,346),outcome,color='#a1afae',width=.85)

# Separate instances; a repeated event category is not a shared event instance.
for cx,name in [(674,'具体事件甲'),(1362,'具体事件乙')]:
    triangle(cx,591,867,216,COL['yellow'],'#bbad8c',.51)
    node(cx,614,name,label=(cx,590),rx=14,ry=8)
    text(cx,650,'独立的历史事件实例',16,color='#8e836d')
    line((cx,349),(cx,601),dash='4 4',color='#8f9e9e',width=1)
    for dx in (-155,0,155):
        line((cx,623),(cx+dx,804),color='#b2ada0',width=.9)
    phases=('增长','恢复','稳定') if cx==674 else ('增长','持续','稳定')
    for dx,label in zip((-155,0,155),phases):
        node(cx+dx,804,label,label=(cx+dx,785),rx=12,ry=7)
    for dx in (-77.5,77.5):
        mx=cx+dx
        blob(f'M {mx-81} 796 C {mx-50} 779 {mx+49} 781 {mx+81} 798 '
             f'C {mx+94} 819 {mx+33} 853 {mx} 852 C {mx-43} 850 {mx-92} 818 {mx-81} 796 Z',
             COL['pink'],'#b7a3ae',.65)
        line((mx-63,804),(mx+63,804),color=INK,width=1.1,arrow=True)
        node(mx,835,rx=8,ry=5)
        line((mx-67,808),(mx,835),width=.8,color='#ad9fa9')
        line((mx+67,808),(mx,835),width=.8,color='#ad9fa9')
    text(cx,888,'过程超边＋有向阶段转移',16,color='#8b7e70')
    text(cx,925,'各阶段分别绑定状态、行动与证据',16,color='#768982')

# Historical observations keep separate times; only their data source is shared here.
blob('M 668 963 C 743 938 978 1000 1031 1055 C 1054 1093 753 1110 628 1101 '
     'C 572 1099 560 1019 605 978 C 625 963 647 966 668 963 Z',COL['purple'],'#aaa1bd',.55)
blob('M 1370 963 C 1295 938 1060 1000 1007 1055 C 984 1093 1285 1110 1410 1101 '
     'C 1466 1099 1478 1019 1433 978 C 1413 963 1391 966 1370 963 Z',COL['purple'],'#aaa1bd',.55)
node(674,992,'历史观测甲',label=(674,970))
node(1362,992,'历史观测乙',label=(1362,970))
node(660,1083,'观测日期甲',label=(660,1110),rx=10,ry=6)
node(1376,1083,'观测日期乙',label=(1376,1110),rx=10,ry=6)
node(1018,1070,'共享数据来源',label=(1018,1103),rx=12,ry=7)
for cx,date in [(674,(660,1083)),(1362,(1376,1083))]:
    line((cx-155,813),(cx,985),dash='4 4',color='#90a29f',width=.9)
    line((cx,999),date,color='#a4a0b5',width=.9)
    line((cx,999),(1018,1070),color='#a4a0b5',width=.9)
    path(f'M {cx} 992 C {cx+(90 if cx==674 else -90)} 913 {cx+(100 if cx==674 else -100)} 679 1004 552',
         dash='4 4',color='#a8acaf',width=.8)
text(1018,1142,'红色节点为查询兼容的共享条件；共享事实不等于两条经验共同生效。',16,color='#7f7872')

# Retrieval follows the implementation rather than a separate decorative arrow.
text(2036,171,'联合检索 → 复用判断',22,weight='bold')
for y,title,detail,fill in [
    (229,'当前条件与证据过滤','时空、阶段、状态、来源与时效',COL['yellow']),
    (359,'语义召回','向量余弦；离线使用 TF-IDF',COL['purple']),
    (489,'经历与阶段绑定检查','在同一经历内核对超边与转移',COL['yellow']),
    (619,'受约束的超边邻域扩展','同任务族；两类共享条件；一跳',COL['green']),
    (749,'联合排序与 Top-K','相关性＋条件覆盖＋联合支持',COL['green']),
    (879,'Jev 单次复用门控','当前查询＋候选＋绑定与证据',COL['blue'])]:
    pipeline(y,title,detail,fill)
for start,end in [(299,359),(429,489),(559,619),(689,749),(819,879)]:
    line((2035,start+4),(2035,end-5),color='#7b928a',width=1.25,arrow=True)
line((2035,953),(2035,1005),color='#7b928a',arrow=True)
text(2035,1040,'复用／综合／停止',22,weight='bold',color='#5b7585')
text(2035,1092,'每条候选独立满足当前条件',17,color='#71857d')
text(2035,1130,'相似度与共享节点不能替代验证',17,color='#71857d')
path('M 1663 803 H 1770 V 652 H 1851',color='#7f9e8b',dash='5 4',width=1.2,arrow=True)
text(1690,787,'联合匹配',15,color='#78917f')

node(393,1188,rx=10,ry=6);text(414,1194,'节点',15,anchor='start')
blob('M 484 1175 C 507 1167 530 1179 534 1190 C 529 1206 488 1207 480 1192 C 476 1185 479 1179 484 1175 Z',COL['blue'])
text(546,1194,'闭合区域＝超边',15,anchor='start')
line((723,1188),(760,1188),dash='4 3');text(774,1194,'索引／证据引用',15,anchor='start')
text(1328,1194,'同一平面上的超图投影；五要素不是五层；向量是辅助索引；阶段先后不代表因果。',15,color='#75858c')
defs='''<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 Z" fill="#667881"/></marker></defs>'''
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'+defs+\
    '<style>text{font-family:"Microsoft YaHei","SimHei",sans-serif}path,ellipse,rect{vector-effect:non-scaling-stroke}</style>'+\
    '<rect width="100%" height="100%" fill="white"/>'+''.join(background+edges+marks+labels)+'</svg>'
target=ROOT/'hypergraph-memory-flat-joint.svg'
target.write_text(svg,encoding='utf-8')
print(target)
