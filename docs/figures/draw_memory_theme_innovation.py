"""Top-down topic index with explicitly highlighted implemented enhancements."""
from pathlib import Path

# Reuse the small SVG primitives without executing the previous diagram layout.
helper = Path(__file__).with_name('draw_memory_reference_style.py')
exec(helper.read_text(encoding='utf-8').split('# One compact enclosing frame')[0], globals())
W, H = 2000, 1260
ROOT = Path(__file__).resolve().parent


def badge(x,y,number,color):
    marks.append(f'<circle cx="{x}" cy="{y}" r="15" fill="white" stroke="{color}" stroke-width="1.5"/>')
    text(x,y+6,str(number),17,weight='bold',color=color)


def record(x,y,title,subtitle,fill=COL['green'],ghost=False):
    blob(f'M {x-80} {y-24} C {x-34} {y-42} {x+59} {y-38} {x+98} {y+4} '
         f'C {x+130} {y+36} {x+103} {y+132} {x+49} {y+150} '
         f'C {x-15} {y+165} {x-113} {y+143} {x-126} {y+92} '
         f'C {x-133} {y+51} {x-115} {y-3} {x-80} {y-24} Z',fill,
         '#bcc1c1' if ghost else '#9bac9b',.5 if ghost else .75)
    box(x-90,y,180,51,title,subtitle,fill='#fff',size=19)
    for dx,name in [(-77,'适用条件'),(0,'行动'),(77,'结果')]:
        line((x,y+51),(x+dx,y+104),color='#9fa5a6',width=1)
        node(x+dx,y+104,name,label=(x+dx,y+130),ghost=ghost)
    text(x,y+177,'经验超边',16,color='#7d8982')


def feature(y,number,title,lines,color):
    badge(1660,y,number,color)
    text(1690,y+7,title,22,anchor='start',weight='bold',color=color)
    for i,value in enumerate(lines):
        text(1690,y+48+i*30,value,18,anchor='start')


rect(18,18,1964,1224,fill='#fff',stroke='#63717b',r=14)
rect(1566,127,390,1046,fill='#fafbfa',stroke='none',r=0)
path('M 1560 127 V 1188',color='#a3adb2',dash='6 6')
text(787,59,'主题索引下的时空演化超图记忆',29,weight='bold')
text(787,96,'主题 → 历史经验 → 具体事件 → 事实证据',19,color='#64727a')
node(124,93,rx=11,ry=6); text(149,99,'节点',16,anchor='start')
blob('M 222 80 C 242 74 264 83 265 95 C 266 109 223 108 216 97 C 209 89 215 84 222 80 Z',COL['blue'])
text(277,99,'超边',16,anchor='start')
line((1265,91),(1300,91),dash='4 3'); text(1310,98,'索引／引用',16,anchor='start')
line((1434,91),(1470,91),color=INK,arrow=True); text(1480,98,'阶段先后',16,anchor='start')

for y,name,detail in [(193,'主题索引层','按任务键组织'),(383,'历史经验层','保存可复用经历'),(664,'具体事件层','事件实例与阶段'),(1041,'事实证据层','观测、实体与条件')]:
    text(141,y,name,21,weight='bold')
    text(141,y+32,detail,16,color='#7b8990')
for y in (279,559,870):
    path(f'M 45 {y} H 1529',color='#c4cbce',width=.8,dash='4 5')

# Topic A jointly indexes two experiences from the same historical event.
triangle(744,137,476,395,COL['purple'],'#b3acc8',.35)
triangle(1304,137,468,174,COL['yellow'],'#c7bb9a',.35)
node(744,171,'交通事件分析',label=(744,145),rx=15,ry=8)
node(1304,171,'恢复评估',label=(1304,145),rx=15,ry=8)
text(744,218,'主题聚合超边',17,color='#716b84')
text(1304,218,'主题聚合超边',17,color='#847c69')
for x in (535,950):
    line((744,181),(x,317),dash='5 4',color='#8c9299',width=1.15,arrow=True)
line((1304,181),(1304,317),dash='5 4',color='#b4b9b9',width=1,arrow=True)
record(535,320,'经验甲','增长阶段入口')
record(950,320,'经验乙','恢复阶段入口',COL['blue'])
record(1304,320,'经验丙','其他历史事件','#eeeeeb',ghost=True)
badge(391,347,1,'#98705e')
text(742,541,'同一事件可形成多个阶段入口的经验',18,color='#657b75')

# Two record IDs point to the same process instance, rather than duplicating it.
path('M 535 500 C 535 537 679 551 740 579',color='#7c9291',dash='5 4',width=1.25,arrow=True)
path('M 950 500 C 950 537 805 551 756 579',color='#7c9291',dash='5 4',width=1.25,arrow=True)
path('M 1304 500 V 579',color='#b4b9b9',dash='5 4',width=1,arrow=True)

# Process hyperedge and overlapping state-transition hyperedges.
blob('M 751 572 C 841 570 995 632 1030 669 C 1077 725 977 739 757 739 '
     'C 548 740 414 724 462 669 C 512 622 681 575 751 572 Z',COL['yellow'],'#b9aa85',.74)
blob('M 488 650 C 553 629 683 637 750 669 C 777 694 677 736 620 746 '
     'C 564 738 463 694 488 650 Z',COL['pink'],'#b9a6af',.7)
blob('M 747 650 C 808 629 938 637 1003 669 C 1033 694 934 736 875 746 '
     'C 821 738 721 694 747 650 Z',COL['green'],'#9bad9d',.7)
node(750,591,'具体事件甲',label=(750,572),rx=15,ry=8)
text(750,628,'过程超边',17,color='#8a7d64')
for x in (495,750,1005):
    line((750,598),(x,672),color='#a7a799',width=1)
for x,name in [(495,'增长阶段'),(750,'恢复阶段'),(1005,'稳定阶段')]:
    node(x,672,name,label=(x,653),rx=13,ry=7)
for x in (622,877):
    line((x-111,672),(x+111,672),color=INK,width=1.4,arrow=True)
    text(x,660,'先于',15,color='#9c745b')
    for target in [(x-127,672),(x+128,672)]:
        line(target,(x,713),color='#a99f99',width=.9)
    node(x,713,'源→目标',label=(x,741),rx=10,ry=6)
text(622,635,'转移超边一',16,color='#83727a')
text(877,635,'转移超边二',16,color='#69816c')
badge(451,611,2,'#ab8450')

# Stage evidence, under the same event view (not a new top-level layer).
blob('M 435 776 C 464 761 539 761 559 780 C 575 800 563 840 539 851 '
     'C 507 864 440 858 425 835 C 415 816 416 790 435 776 Z',COL['green'],'#9caf9f',.7)
blob('M 683 776 C 710 761 789 761 810 780 C 830 804 811 843 788 853 '
     'C 755 865 693 855 677 836 C 664 816 665 790 683 776 Z',COL['blue'],'#9eafba',.7)
for start,members in [((495,678),[(448,791),(540,791),(495,829)]),
                       ((750,678),[(697,791),(801,791),(750,829)])]:
    for target in members:
        line(start,target,dash='4 4',color='#8fa4a0',width=1)
node(448,791,'排队增长',label=(438,775),rx=10,ry=6)
node(540,791,'校验观测',label=(548,775),rx=10,ry=6)
node(495,829,'阶段时间',label=(495,853),rx=10,ry=6)
node(697,791,'排队缩短',label=(687,775),rx=10,ry=6)
node(801,791,'核对基线',label=(813,775),rx=10,ry=6)
node(750,829,'执行结果',label=(750,853),rx=10,ry=6)
text(1130,817,'阶段证据超边：联合保存状态、行动与结果',17,color='#6c827b')

triangle(1304,572,723,103,'#eeeeeb','#c0c7c7',.45)
node(1304,598,'具体事件乙',label=(1304,577),ghost=True)
for x in (1255,1353):
    line((1304,605),(x,686),color='#b9c1c3',dash='4 4')
    node(x,686,ghost=True)
text(1304,754,'其他事件的阶段与事实',16,color='#8c999f')

# Provenance-bearing observations share one space node and one time node.
blob('M 460 911 C 517 907 642 951 676 986 C 714 1031 657 1109 583 1128 '
     'C 511 1135 402 1117 387 1062 C 371 1008 414 916 460 911 Z',COL['blue'],'#95adbb',.72)
blob('M 743 910 C 802 913 884 1005 879 1060 C 871 1119 788 1140 712 1126 '
     'C 650 1116 580 1089 577 1050 C 574 997 683 917 743 910 Z',COL['green'],'#9eaf9e',.72)
blob('M 629 984 C 719 952 1003 939 1041 950 C 1098 967 1164 1071 1132 1102 '
     'C 1084 1133 754 1064 624 1017 C 598 1007 603 994 629 984 Z',COL['yellow'],'#bcae8e',.64)

for target in [(414,1044),(630,1010),(588,1100)]:
    line((464,934),target,color='#97aab2',width=1)
for target in [(843,1044),(630,1010),(588,1100)]:
    line((743,934),target,color='#94a996',width=1)
for target in [(630,1010),(1039,970)]:
    line((1126,1082),target,color='#ada38f',width=1)
node(464,934,'流量观测',label=(464,912))
node(414,1044,'门架来源',label=(414,1073))
node(743,934,'事件观测',label=(743,912))
node(843,1044,'事件来源',label=(843,1073))
node(630,1010,'路段甲',label=(630,989))
node(588,1100,'九时观测时间',label=(588,1128))
node(1039,970,'路段乙',label=(1039,948))
node(1126,1082,'乙在甲上游',label=(1126,1113))
text(468,1156,'观测超边一',17,color='#6b8797')
text(775,1156,'观测超边二',17,color='#768b78')
text(1110,1156,'空间关系超边',17,color='#94886e')
badge(398,936,3,'#648e83')
text(933,1035,'共享实体',16,color='#92866e')

# Only real, named reference paths: no arbitrary graph-wide dotted lines.
path('M 488 679 C 331 714 341 864 456 926',dash='4 4',color='#72958c',width=1.3,arrow=True)
path('M 502 679 C 632 735 653 862 735 926',dash='4 4',color='#72958c',width=1.3,arrow=True)
text(377,895,'引用观测证据',16,color='#67867c')
text(698,895,'引用事件证据',16,color='#67867c')
text(741,1188,'多个来源保留独立出处，通过共享实体与时间节点关联',17,color='#6b8278')

# Three visible structural enhancements and a separate reuse gate.
text(1765,168,'关键增强',24,weight='bold')
feature(290,1,'时空条件感知',[
    '经验联合关联时间、地点与状态',
    '前提不满足时排除该经验',
    '避免仅凭语义相似直接复用'], '#98705e')
feature(569,2,'阶段化演化记忆',[
    '保存阶段、行动、后续结果与先后',
    '根据当前阶段选择经验入口',
    '保存时序，不把先后当作因果'], '#ab8450')
feature(884,3,'多源证据一致性',[
    '同一实体的多源观测联合关联',
    '显式对齐单位、实体与观测时间',
    '新鲜度和来源冲突参与复用检查'], '#648e83')
path('M 1605 1067 H 1929',color='#b7c3c7',width=.8)
text(1769,1103,'快速判断模型：复用门控',21,weight='bold',color='#526e80')
text(1769,1138,'当前条件 + 候选经验 + 观测证据',18)
text(1769,1173,'一次判断 → 复用／综合／停止',18)

text(773,1224,'主题当前由任务键组织；本图为索引视图；经验、节点、超边与关联统一保存在数据库中。',16,color='#738087')
defs='''<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 Z" fill="#667881"/></marker></defs>'''
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'+defs+\
    '<style>text{font-family:"Microsoft YaHei","SimHei",sans-serif}path,ellipse,rect{vector-effect:non-scaling-stroke}</style>'+\
    '<rect width="100%" height="100%" fill="white"/>'+''.join(background+edges+marks+labels)+'</svg>'
target=ROOT/'hypergraph-memory-theme-innovation.svg'
target.write_text(svg,encoding='utf-8')
print(target)
