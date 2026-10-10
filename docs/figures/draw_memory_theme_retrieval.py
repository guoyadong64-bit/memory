"""Native SVG: separate event branches, five-element episodes, joint retrieval."""
from pathlib import Path

helper = Path(__file__).with_name('draw_memory_reference_style.py')
exec(helper.read_text(encoding='utf-8').split('# One compact enclosing frame')[0], globals())
W, H = 2140, 1370
ROOT = Path(__file__).resolve().parent


def episode(cx, title, values, fill):
    blob(f'M {cx} 307 C {cx+95} 306 {cx+209} 394 {cx+229} 454 '
         f'C {cx+248} 508 {cx+146} 538 {cx} 538 '
         f'C {cx-144} 538 {cx-247} 508 {cx-229} 454 '
         f'C {cx-209} 394 {cx-94} 306 {cx} 307 Z',fill,'#9ca8ad',.70)
    box(cx-84,328,168,48,title,size=21)
    for dx,kind,value in zip((-180,-90,0,90,180),('时间','地点','事件','行动','结果'),values):
        line((cx,377),(cx+dx,452),color='#a1abad',width=.9)
        node(cx+dx,452,kind,label=(cx+dx,429),rx=11,ry=6)
        text(cx+dx,487,value,15)
    text(cx,519,'一条经验超边联合关联五类要素',16,color='#687e7a')


def process(cx, label, phases, fill):
    blob(f'M {cx} 632 C {cx+70} 631 {cx+164} 703 {cx+179} 744 '
         f'C {cx+198} 783 {cx+87} 803 {cx} 803 '
         f'C {cx-91} 803 {cx-197} 783 {cx-179} 744 '
         f'C {cx-165} 701 {cx-73} 632 {cx} 632 Z',fill,'#a7ab9c',.65)
    node(cx,649,label,label=(cx,624),rx=14,ry=8)
    text(cx,683,'独立事件实例／过程超边',16,color='#788277')
    for x,name in zip((cx-150,cx,cx+150),phases):
        line((cx,658),(x,745),color='#a4abad',width=.9)
        node(x,745,name,label=(x,727),rx=12,ry=7)
    for x in (cx-75,cx+75):
        blob(f'M {x-79} 735 C {x-50} 721 {x+47} 721 {x+79} 737 '
             f'C {x+92} 755 {x+38} 793 {x} 794 '
             f'C {x-43} 792 {x-93} 757 {x-79} 735 Z',COL['pink'],'#b8a2ad',.65)
        line((x-61,745),(x+61,745),color=INK,width=1.15,arrow=True)
        node(x,776,rx=8,ry=5)
        line((x-138/2,746),(x,776),color='#a89ca4',width=.8)
        line((x+138/2,746),(x,776),color='#a89ca4',width=.8)
    text(cx,824,'有向转移超边：保存阶段先后',16,color='#8b717e')


def stage_evidence(cx,values,fill):
    blob(f'M {cx-141} 876 C {cx-92} 851 {cx+92} 851 {cx+141} 876 '
         f'C {cx+170} 911 {cx+128} 973 {cx+53} 979 '
         f'C {cx-40} 995 {cx-168} 958 {cx-151} 909 Z',fill,'#9eaeb2',.68)
    for dx,label,value in zip((-100,0,100),('状态','行动','结果'),values):
        line((cx,753),(cx+dx,900),dash='4 4',color='#8a9e9c',width=.9)
        node(cx+dx,900,label,label=(cx+dx,884),rx=10,ry=6)
        text(cx+dx,929,value,15)
    node(cx,958,'观测证据',label=(cx,982),rx=10,ry=6)
    for dx in (-100,0,100):
        line((cx+dx,906),(cx,958),color='#a0adae',width=.8)
    text(cx,1017,'阶段证据超边',16,color='#6e8585')


def flow(y,title,detail,fill=COL['green'],height=67):
    box(1737,y,332,height,title,detail,fill=fill,size=20)


rect(18,18,2104,1334,fill='#fff',stroke='#52616a',r=14)
rect(27,126,285,1137,fill='#f8f9f8',stroke='none',r=0)
rect(1709,126,404,1137,fill='#f6faf5',stroke='none',r=0)
for x in (315,1704):
    path(f'M {x} 126 V 1304',color='#8d989e',width=1,dash='7 6')
text(1065,61,'时空演化超图记忆：经历组织与联合检索',29,weight='bold')
text(1065,98,'主题 → 历史经验 → 具体事件与阶段 → 状态、行动、结果与观测证据',18,color='#687b83')

# Input column: explicit mapping precedes storage; no raw multimodal parsing claim.
text(171,162,'经历输入与组织',22,weight='bold')
for y,title,detail,fill in [(234,'交互与任务经历','问题、执行记录、反馈',COL['purple']),
                           (351,'多源观测','门架、事件记录、拓扑',COL['blue'])]:
    box(55,y,231,67,title,detail,fill=fill,size=19)
path('M 171 302 V 350',color='#91a0a6',arrow=True)
path('M 171 419 V 476',color='#91a0a6',arrow=True)
box(55,480,231,72,'字段、实体与单位对齐','显式映射配置',fill='#fff',size=18)
line((171,552),(171,611),arrow=True)
box(47,615,247,82,'结构化记忆','时间 · 地点 · 事件 · 行动 · 结果',fill=COL['yellow'],size=19)
path('M 294 656 H 340 V 354 H 552',color='#778b92',width=1.4,arrow=True)
text(370,337,'组织入库',16,color='#6b818a',anchor='start')
text(170,786,'三个设计重点',21,weight='bold')
for y,head,lines in [(848,'时空条件感知',['经历与适用前提联合保存','当前条件不满足则排除']),
                     (1000,'阶段化演化记忆',['每个事件保留自己的阶段','状态、行动、结果可追溯']),
                     (1152,'多源证据一致性',['保留出处与观测时间','检查新鲜度、对齐与冲突'])]:
    text(170,y,head,19,weight='bold',color='#667f7c')
    for i,value in enumerate(lines):
        text(170,y+35+i*29,value,16)

# One task topic, with two independent historical cases below it.
triangle(957,155,538,555,COL['purple'],'#b1abc3',.24)
node(957,183,'拥堵分析主题',label=(957,159),rx=15,ry=8)
text(957,219,'任务主题索引／聚合超边',17,color='#767087')
for cx in (663,1253):
    line((957,191),(cx,324),color='#8d959e',dash='5 4',width=1.15,arrow=True)
episode(663,'历史经验甲',('九时至十时','路段甲','事故拥堵','核对观测','排队缩短'),COL['green'])
episode(1253,'历史经验乙',('九时至十一时','路段乙','需求拥堵','评估恢复','流量稳定'),COL['blue'])
for y,title in [(264,'主题索引层'),(570,'历史经验层'),(1039,'事实证据层')]:
    # Layer captions are offset left; boundaries are thin, paper-style rules.
    text(406,y,title,17,weight='bold',color='#718087')
path('M 361 586 H 1657',color='#c4cbce',width=.8,dash='4 5')
path('M 361 1048 H 1657',color='#c4cbce',width=.8,dash='4 5')
text(398,667,'具体事件层',17,weight='bold',color='#718087')
for cx in (663,1253):
    line((cx,541),(cx,635),dash='5 4',color='#829a97',width=1.15,arrow=True)
text(957,582,'两条经历分别关联各自的事件；共享事实通过超边交叠关联',16,color='#6c8380')
process(663,'具体事件甲',('增长','恢复','稳定'),COL['yellow'])
process(1253,'具体事件乙',('形成','持续','缓解'),COL['yellow'])
stage_evidence(663,('排队缩短','核对基线','恢复确认'),COL['green'])
stage_evidence(1253,('流量偏高','联合核验','持续确认'),COL['blue'])

# Two observation hyperedges share exactly one visible source and time node.
blob('M 542 1080 C 635 1069 989 1071 1004 1110 C 1055 1151 1032 1211 978 1235 '
     'C 776 1262 591 1250 537 1208 C 489 1172 488 1090 542 1080 Z',COL['blue'],'#a0b2bd',.68)
blob('M 1368 1080 C 1275 1069 921 1071 906 1110 C 855 1151 878 1211 932 1235 '
     'C 1134 1262 1319 1250 1373 1208 C 1421 1172 1422 1090 1368 1080 Z',COL['green'],'#9dae9f',.68)
blob('M 631 1160 C 752 1138 1215 1138 1280 1164 C 1321 1185 1190 1238 957 1243 '
     'C 798 1244 576 1213 631 1160 Z',COL['yellow'],'#b9ac8b',.45)
for obs,entity in [((553,1107),(636,1180)),((1357,1107),(1274,1180))]:
    for target in (entity,(957,1114),(957,1212)):
        line(obs,target,color='#9aaeb1',width=.9)
node(553,1107,'流量观测甲',label=(553,1087))
node(1357,1107,'流量观测乙',label=(1357,1087))
node(636,1180,'路段甲',label=(636,1161))
node(1274,1180,'路段乙',label=(1274,1161))
node(957,1114,'同一观测时刻',label=(957,1092))
node(957,1212,'同一门架数据来源',label=(957,1245))
node(1105,1180,'甲在乙上游',label=(1105,1160),rx=10,ry=6)
for target in ((636,1180),(1274,1180)):
    line((1105,1180),target,color='#aaa38e',width=.9)
for cx,target in [(663,(553,1107)),(1253,(1357,1107))]:
    line((cx,964),target,dash='4 4',color='#819e98',width=1.15,arrow=True)
text(623,1275,'观测超边甲',16,color='#6e8794')
text(1271,1275,'观测超边乙',16,color='#708b75')
text(957,1282,'共享时间、来源；关联空间拓扑',15,color='#887c67')

# Actual retrieval order: task gate -> feasibility/evidence -> heuristic ranking -> Jev.
text(1910,162,'联合检索与复用',22,weight='bold')
flow(229,'当前查询','任务、问题、时间、地点与状态',COL['blue'])
flow(366,'任务主题定位','按任务键选择经验族',COL['purple'])
flow(503,'时空与适用前提筛选','时间窗口、空间范围、事件与阶段',COL['yellow'],78)
flow(652,'当前观测证据检查','时效、来源、单位对齐与冲突',COL['green'],78)
flow(801,'联合相关性排序','时间＋空间＋语义＋事件／状态',COL['green'],78)
text(1903,911,'当前语义项：问题词项重合',16,color='#7c8e88')
flow(959,'候选记忆集合','保留排名靠前的可用经验',COL['green'])
flow(1095,'Jev 单次复用门控','当前查询＋候选经验＋证据',COL['blue'])
for a,b in [(296,362),(433,499),(581,648),(730,797),(879,955),(1026,1091)]:
    line((1903,a+4),(1903,b-4),color='#78908d',width=1.3,arrow=True)
line((1903,1166),(1903,1211),color='#78908d',arrow=True)
text(1903,1244,'复用／综合／停止',21,weight='bold',color='#536e7f')
path('M 1610 955 H 1677 V 992 H 1733',color='#7a9a90',dash='5 4',width=1.25,arrow=True)
text(1610,938,'召回候选',15,color='#6b8a7e')

node(396,1315,rx=10,ry=6); text(418,1321,'节点',15,anchor='start')
blob('M 485 1304 C 503 1297 527 1302 535 1316 C 532 1331 487 1332 479 1318 C 476 1312 479 1307 485 1304 Z',COL['blue'])
text(549,1321,'闭合区域＝超边',15,anchor='start')
line((731,1315),(768,1315),dash='4 3'); text(780,1321,'索引／引用',15,anchor='start')
text(1248,1321,'层次为展示视图；主题按任务键组织；阶段先后不代表因果；统一保存于 SQLite。',15,color='#75858c')
defs='''<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 Z" fill="#667881"/></marker></defs>'''
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'+defs+\
    '<style>text{font-family:"Microsoft YaHei","SimHei",sans-serif}path,ellipse,rect{vector-effect:non-scaling-stroke}</style>'+\
    '<rect width="100%" height="100%" fill="white"/>'+''.join(background+edges+marks+labels)+'</svg>'
target=ROOT/'hypergraph-memory-theme-retrieval.svg'
target.write_text(svg,encoding='utf-8')
print(target)
