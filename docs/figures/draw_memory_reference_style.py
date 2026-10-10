"""Deterministic SVG diagram. Every relation has explicitly placed endpoints."""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
W, H = 1840, 1030
background, edges, marks, labels = [], [], [], []
INK = '#253342'
COL = {'blue':'#dcecf6','green':'#e1eee3','yellow':'#faf0d1','pink':'#f0e3ec','purple':'#e9e5f5'}


def text(x,y,value,size=19,anchor='middle',weight='normal',color=INK,italic=False):
    labels.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}"'+
                  (' font-style="italic"' if italic else '')+f'>{escape(value)}</text>')


def path(d,color='#89939c',width=1.2,dash=None,arrow=False):
    edges.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'+
                 (f' stroke-dasharray="{dash}"' if dash else '')+
                 (' marker-end="url(#arrow)"' if arrow else '')+'/>' )


def line(a,b,**kw):
    path(f'M {a[0]} {a[1]} L {b[0]} {b[1]}',**kw)


def node(x,y,name=None,label=None,color=INK,rx=12,ry=7,ghost=False):
    marks.append(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="white" stroke="{color}" stroke-width="1.4" opacity="{.5 if ghost else 1}"/>')
    if name:
        xx,yy = label or (x,y-15)
        text(xx,yy,name,size=17,color='#84909a' if ghost else INK)


def rect(x,y,w,h,fill='#fff',stroke='#667480',r=3,layer=background):
    layer.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')


def blob(d,color,stroke='#8a9ba9',opacity=.8):
    background.append(f'<path d="{d}" fill="{color}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="1.15"/>')


def triangle(cx,top,bottom,half,color,stroke='#91a2ab',opacity=.8):
    blob(f'M {cx} {top} C {cx+half*.28} {top-4},{cx+half*.7} {bottom-65},{cx+half} {bottom-27} '
         f'C {cx+half+20} {bottom+10},{cx-half-20} {bottom+10},{cx-half} {bottom-27} '
         f'C {cx-half*.7} {bottom-65},{cx-half*.28} {top-4},{cx} {top} Z',color,stroke,opacity)


def box(x,y,w,h,title,subtitle=None,fill='#fff',size=20):
    rect(x,y,w,h,fill=fill,layer=marks)
    text(x+w/2,y+(h/2+7 if not subtitle else h/2-3),title,size=size,weight='bold')
    if subtitle:
        text(x+w/2,y+h/2+21,subtitle,size=16)


# One compact enclosing frame, as in the reference architecture figure.
rect(15,18,1810,990,fill='#fff',stroke='#49525a',r=18)
background.append('<path d="M 28 20 H 280 V 1006 H 28 Q 17 1006 17 994 V 32 Q 17 20 28 20 Z" fill="#f8f8f7"/>')
rect(282,20,998,986,fill='#fffdf6',stroke='none',r=0)
rect(1282,20,238,986,fill='#f6fbf3',stroke='none',r=0)
background.append('<path d="M 1522 20 H 1810 Q 1823 20 1823 33 V 993 Q 1823 1006 1810 1006 H 1522 Z" fill="#f1f8fc"/>')
for x in (281,1281,1521):
    path(f'M {x} 20 V 1007',color='#747d82',width=1.2,dash='9 7')
for y in (303,561,768):
    path(f'M 305 {y} H 1257',color='#bac1c4',dash='5 5',width=.8)
text(148,61,'多源输入与对齐',22,weight='bold')
text(781,61,'多源时空与过程演化超图记忆',26,weight='bold')
text(1400,61,'检索',22,weight='bold')
text(1670,61,'复用判断',22,weight='bold')

# Legend uses the same small oval-node / enclosing-hyperedge notation as the reference.
node(610,88,rx=10,ry=6)
text(633,94,'节点',15,anchor='start')
blob('M 722 76 C 737 68 753 78 759 89 C 755 102 722 102 713 92 C 707 86 714 80 722 76 Z',COL['blue'])
node(726,88,rx=5,ry=3); node(745,90,rx=5,ry=3)
text(772,94,'超边',15,anchor='start')
line((847,88),(883,88),dash='4 3')
text(894,94,'跨层引用',15,anchor='start')
line((1010,88),(1046,88),color=INK,arrow=True)
text(1058,94,'阶段先后',15,anchor='start')

# Input rows and explicit normalization pipeline.
for y,title,detail,fill in [(165,'门架观测','流量、速度、采样时间',COL['blue']),
                           (270,'事件记录','事件类型、位置、处置状态',COL['pink']),
                           (375,'空间拓扑','实体标识、包含与上下游',COL['yellow'])]:
    box(49,y,201,65,title,detail,fill=fill,size=20)
    path(f'M 250 {y+32} H 262',color='#89959d')
path('M 262 197 V 458 H 149 V 480',color='#687680',arrow=True)
box(64,483,171,47,'字段映射',fill='#fff')
box(64,575,171,47,'单位统一',fill='#fff')
box(64,667,171,47,'实体对齐',fill='#fff')
box(48,800,205,60,'统一结构化观测',fill=COL['blue'],size=19)
for y1,y2 in [(530,575),(622,667),(714,800)]:
    line((149,y1+3),(149,y2-4),arrow=True,color='#687680')
path('M 253 830 H 415',color=INK,width=1.7,arrow=True)
text(334,819,'归一化入库',16)
text(146,938,'显式配置字段、单位和别名',16)
text(146,964,'原始数据由上游模块处理',16,color='#687680')

# Four logical layers, kept small and integrated rather than separate colored panels.
for y,number,title in [(179,'第四层','任务组织'),(402,'第三层','过程演化'),(647,'第二层','阶段经验'),(868,'第一层','观测与上下文')]:
    text(354,y,number,19,weight='bold')
    text(354,y+28,title,18,weight='bold')
for y1,y2,caption,cy in [(362,250,'任务聚合',317),(599,474,'过程关联',544),(817,714,'证据绑定',783)]:
    line((354,y1),(354,y2),color='#7f8b91',width=1.2,arrow=True)
    rect(309,cy-21,90,29,fill='#fffdf6',stroke='none',r=0,layer=marks)
    text(354,cy,caption,16,color='#68777d')

# Layer 4: task families and actual record indexes (rectangles).
triangle(675,125,260,137,COL['purple'],'#a4a0bc')
triangle(1060,125,260,137,COL['yellow'],'#b9ac89')
node(675,146,'分析流程',label=(675,126))
node(1060,146,'恢复评估',label=(1060,126))
for root,positions in [((675,146),[(592,222),(752,222)]),((1060,146),[(977,222),(1137,222)])]:
    for pos in positions:
        line(root,pos,color='#8d9498',width=1)
for x,title in [(554,'经验甲'),(714,'经验乙'),(939,'经验丙'),(1099,'经验丁')]:
    rect(x,208,76,28,fill='white',stroke='#b16258' if title=='经验甲' else '#7b8791',layer=marks)
    text(x+38,228,title,16,color='#914f45' if title=='经验甲' else INK)
text(675,286,'任务聚合超边',17)
text(1060,286,'任务聚合超边',17)
path('M 592 237 C 595 281 675 302 735 339',dash='5 4',color='#8f969a')
path('M 752 237 L 750 339',dash='5 4',color='#8f969a')
path('M 977 237 C 1024 282 1110 306 1160 350',dash='5 4',color='#adb3b6')
text(660,321,'同类经验 → 历史过程',16,color='#6e7982')

# Layer 3: one process, two overlapping transition hyperedges, shared recovery node.
blob('M 748 333 C 825 330 960 405 1001 468 C 1040 531 500 543 488 482 C 480 435 661 337 748 333 Z',COL['yellow'],'#b5a37c',.78)
blob('M 544 443 C 584 414 695 416 755 449 C 786 476 713 514 650 531 C 593 522 526 483 544 443 Z',COL['pink'],'#b5a3a6',.65)
blob('M 747 443 C 787 414 897 416 958 449 C 990 476 917 514 852 531 C 795 522 729 483 747 443 Z',COL['green'],'#a4b4a3',.65)
for x in (550,750,950):
    line((750,350),(x,453),color='#9d9b91',width=1)
node(750,350,'事件甲',label=(750,333))
for x,title in [(550,'增长阶段'),(750,'恢复阶段'),(950,'稳定阶段')]:
    node(x,453,title,label=(x,440))
for x in (650,850):
    line((x-85,453),(x+83,453),color=INK,width=1.2,arrow=True)
    text(x,442,'先于',14,color='#a06c4f')
    line((x-100,453),(x,500),color='#a49b91',width=.9)
    line((x+100,453),(x,500),color='#a49b91',width=.9)
    node(x,500,'源→目标',label=(x,522),rx=10,ry=6)
text(650,549,'转移超边一',15,color='#786b67')
text(850,549,'转移超边二',15,color='#647464')
text(750,384,'过程超边',16)
triangle(1157,346,464,70,'#eeeeeb','#bec2be',.45)
for x,y in [(1157,367),(1118,432),(1196,432)]:
    node(x,y,ghost=True)
line((1157,367),(1118,432),color='#c0c4c4',dash='4 4'); line((1157,367),(1196,432),color='#c0c4c4',dash='4 4')
text(1157,495,'其他事件实例',16,color='#8d9699')

# Layer 2: stage-member projections. All lines end on real node coordinates.
blob('M 478 597 C 507 583 610 583 627 610 C 653 649 603 704 554 716 '
     'C 507 718 461 698 456 669 C 455 641 459 612 478 597 Z',COL['green'],'#98af9b')
blob('M 755 597 C 786 580 887 583 912 606 C 939 635 943 686 917 706 '
     'C 887 727 775 725 752 704 C 731 679 728 623 755 597 Z',COL['blue'],'#93a9b6')
triangle(1120,595,702,65,'#eeeeeb','#bfc4c4',.45)
growth_members=[(487,612),(610,612),(545,675)]
recovery_members=[(765,612),(890,612),(771,675),(891,675)]
for pos in growth_members:
    line((550,460),pos,dash='4 4',color='#9aa4a4',width=1)
for pos in recovery_members:
    line((750,460),pos,dash='4 4',color='#9aa4a4',width=1)
for pos in [(1089,658),(1151,658)]:
    line((950,460),pos,dash='4 4',color='#c0c4c4',width=.9)
node(487,612,'排队增长',label=(487,596)); node(610,612,'校验观测',label=(610,596))
node(545,675,'阶段时间',label=(545,701))
node(765,612,'排队缩短',label=(765,596)); node(890,612,'核对基线',label=(890,596))
node(771,675,'执行结果',label=(771,701)); node(891,675,'阶段时间',label=(891,701))
for pos in [(1089,658),(1151,658)]:
    node(*pos,ghost=True)
text(550,743,'阶段证据超边',17)
text(827,743,'阶段证据超边',17)
text(1120,724,'其他阶段经验',16,color='#909a9e')
path('M 543 460 C 438 519 423 735 520 830',dash='4 4',color='#829c98',width=1.1)
path('M 557 460 C 688 535 665 746 735 830',dash='4 4',color='#829c98',width=1.1)
text(477,786,'引用流量证据',16,color='#617b77')
text(704,786,'引用事件证据',16,color='#617b77')

# Layer 1: the two observation hyperedges share EXACTLY ONE entity and one time node.
blob('M 511 810 C 540 807 639 836 671 870 C 700 910 646 944 564 949 C 496 957 439 935 430 904 C 418 867 477 815 511 810 Z',COL['blue'],'#8cabbc',.8)
blob('M 735 810 C 772 806 834 848 850 898 C 855 934 772 958 696 945 C 653 938 606 922 603 896 C 600 858 685 816 735 810 Z',COL['green'],'#96ae9b',.8)
blob('M 647 853 C 733 831 1027 822 1065 840 C 1115 857 1130 915 1090 940 C 1045 958 739 918 647 879 C 623 868 626 859 647 853 Z',COL['yellow'],'#b9aa84',.68)
for target in [(465,900),(650,865),(620,910)]:
    line((520,830),target,color='#8b9fa7',width=1)
for target in [(805,900),(650,865),(620,910)]:
    line((735,830),target,color='#90a295',width=1)
line((1080,920),(650,865),color='#a89d84',width=1)
line((1080,920),(1050,850),color='#a89d84',width=1)
node(520,830,'流量观测',label=(520,813)); node(465,900,'门架来源',label=(465,928))
node(735,830,'事件观测',label=(735,813)); node(805,900,'事件来源',label=(805,928))
node(650,865,'路段甲',label=(650,847)); node(620,910,'观测时间',label=(620,938))
node(1050,850,'路段乙',label=(1050,831)); node(1080,920,'乙在甲上游',label=(1080,947))
text(515,966,'观测超边一',16)
text(762,966,'观测超边二',16)
text(1070,982,'空间关系超边',16)
text(933,891,'共享实体',15,color='#82785e')

# Retrieval and judgement, arranged like the reference's narrow right hand columns.
box(1314,169,175,54,'任务召回',fill='#edf5e7')
box(1314,426,175,65,'时空与阶段过滤',fill='#edf5e7',size=19)
box(1314,652,175,68,'观测对齐',subtitle='冲突检查',fill='#edf5e7',size=20)
for start,end in [(223,273),(373,424),(491,535),(610,650),(720,793)]:
    line((1401,start+3),(1401,end-3),color='#5f7261',arrow=True)
triangle(1401,275,367,63,COL['green'],'#9baf92')
node(1401,286,rx=9,ry=5); node(1369,340,rx=9,ry=5,color='#b36358'); node(1433,340,rx=9,ry=5)
line((1401,291),(1369,335),dash='3 3'); line((1401,291),(1433,335),dash='3 3')
text(1401,393,'匹配任务索引',16)
triangle(1401,537,604,54,COL['blue'],'#93a6a2')
node(1382,573,rx=9,ry=5,color='#b36358'); node(1422,575,rx=9,ry=5)
text(1401,631,'排除不适用经验',16)
box(1314,796,175,53,'候选记忆',fill='#e5f0e0')
text(1401,886,'来源与约束一起保留',16)
text(1401,962,'按条件召回与筛选',16,color='#647263')
box(1564,169,224,65,'当前查询',subtitle='时间、地点、阶段与观测',fill='#fff')
path('M 1564 195 H 1492',color=INK,arrow=True,width=1.25)
path('M 1196 199 H 1281 V 196 H 1311',color='#6d797c',dash='4 4',arrow=True,width=1)
box(1564,796,224,67,'快速判断模型',subtitle='相关度 · 支持度 · 选择',fill='#dceef8',size=20)
path('M 1491 822 H 1561',color=INK,arrow=True,width=1.5)
path('M 1676 237 V 793',color='#7a91a1',arrow=True,width=1.15)
rect(1598,434,157,68,fill='#f1f8fc',stroke='none',r=0,layer=marks)
text(1676,459,'携带当前条件',18)
text(1676,487,'与历史记忆比较',18)
box(1564,917,224,50,'复用／综合／停止',fill='#fff',size=20)
line((1676,866),(1676,914),arrow=True,color='#657d8c')

text(780,1000,'逻辑关系展开示意；完整经验还保存适用条件、回答与来源，所有结构统一落盘。',16,color='#677279')

defs='''<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 Z" fill="#43535d"/></marker></defs>'''
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'+defs+\
    '<style>text{font-family:"Microsoft YaHei","SimHei",sans-serif}path,ellipse,rect{vector-effect:non-scaling-stroke}</style>'+\
    '<rect width="100%" height="100%" fill="white"/>'+''.join(background+edges+marks+labels)+'</svg>'
target=ROOT/'hypergraph-memory-reference-style.svg'
target.write_text(svg,encoding='utf-8')
print(target)
