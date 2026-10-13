"""A compact four-tier schematic with small nodes and explicit hyperedges.

Conceptual examples only; no experimental measurements. Four tiers are views of
one stored hypergraph. Drawn and exported entirely with the Python backend.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
EXTRA = REPO / "runtime" / "figure-deps"
if EXTRA.is_dir():
    sys.path.insert(0, str(EXTRA))
os.environ.setdefault("MPLCONFIGDIR", str(REPO / "runtime" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.path import Path as MPath
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon

FONT = Path("C:/Windows/Fonts/msyh.ttc")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    FONT_NAME = font_manager.FontProperties(fname=str(FONT)).get_name()
else:
    FONT_NAME = "Noto Sans CJK SC"
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_NAME, "DejaVu Sans"],
    "font.size": 9.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
    "savefig.facecolor": "white",
})

W, H = 1600, 1120
fig = plt.figure(figsize=(13.6, 9.52), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")
INK = "#273D51"
MUTED = "#647580"
LINE = "#748591"
PURPLE = "#E4DDF0"
GREEN = "#DCEAD9"
BLUE = "#D9E8F1"
YELLOW = "#FAEDBC"
PINK = "#F0DDE2"


def txt(x, y, value, size=9.5, color=INK, weight="normal", ha="center", va="center", **kw):
    return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                   ha=ha, va=va, zorder=10, **kw)


def line(a, b, color=LINE, lw=.9, dashed=False, arrow=False, z=4, rad=0):
    if arrow:
        patch = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8,
                                connectionstyle=f"arc3,rad={rad}", color=color,
                                linewidth=lw, linestyle=(0, (3, 3)) if dashed else "-",
                                shrinkA=0, shrinkB=0, zorder=z)
        ax.add_patch(patch)
    else:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                linestyle=(0, (3, 3)) if dashed else "-", zorder=z)


def curve(points, color=LINE, lw=.9, dashed=True, z=3):
    ax.add_patch(PathPatch(MPath(points, [MPath.MOVETO] + [MPath.CURVE4] * 3),
                          fill=False, edgecolor=color, lw=lw,
                          linestyle=(0, (3, 3)) if dashed else "-", zorder=z))


def blob(points, fill, edge=INK, alpha=.7, lw=.85, z=2):
    # Cubic path: one start, triples of Bezier control points, one close.
    codes = [MPath.MOVETO] + [MPath.CURVE4] * (len(points) - 2) + [MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(points, codes), facecolor=fill, edgecolor=edge,
                          alpha=alpha, lw=lw, joinstyle="round", zorder=z))


def node(x, y, label=None, dy=23, fill="white", rx=9, ry=5.4, lw=1.05, color=INK, fs=9.3):
    ax.add_patch(Ellipse((x, y), rx*2, ry*2, facecolor=fill,
                         edgecolor=color, lw=lw, zorder=8))
    if label:
        txt(x, y+dy, label, fs)


def record(x, y, name):
    # A record index marker in the display, not an added database node type.
    ax.add_patch(FancyBboxPatch((x-13, y-9), 26, 18,
                 boxstyle="round,pad=0,rounding_size=4", facecolor="white",
                 edgecolor=INK, linewidth=1.05, zorder=8))
    txt(x, y, name, 7.5)


def plane(y, x0, x1, height, fill):
    ax.add_patch(Polygon([(x0+30, y), (x1+30, y), (x1, y+height), (x0, y+height)],
                  closed=True, facecolor=fill, edgecolor="#D6D4C7", lw=.6, zorder=0))


def tier(y, number, title, detail):
    txt(53, y-23, f"0{number}", 10.8, "#9B8B71", ha="left")
    txt(53, y+2, title, 12.4, weight="bold", ha="left")
    txt(53, y+30, detail, 9, MUTED, ha="left")


def flow(y, title, sub, fill):
    ax.add_patch(FancyBboxPatch((1272, y), 278, 64,
                  boxstyle="round,pad=0,rounding_size=4", facecolor=fill,
                  edgecolor="#70888A", linewidth=.7, zorder=5))
    txt(1411, y+22, title, 10.5, weight="bold")
    txt(1411, y+45, sub, 8.2, MUTED)


# A compact page. The dominant panel is the four-tier hypergraph.
txt(625, 42, "时空语义超图记忆", 19, weight="bold")
txt(625, 75, "主题组织经历，经历关联事件，事件追溯证据", 10.2, MUTED)
ax.add_patch(FancyBboxPatch((1249, 103), 325, 942,
             boxstyle="round,pad=0,rounding_size=8", facecolor="#F3F8F3",
             edgecolor="#A5B7AB", linewidth=.65, zorder=0))
txt(1411, 132, "联合检索与复用", 13, weight="bold")

plane(125, 365, 1095, 124, "#FCFAF1")
plane(338, 273, 1150, 193, "#FCFAF1")
plane(596, 253, 1170, 197, "#FCFAF1")
plane(861, 232, 1190, 176, "#FCFAF1")
tier(185, 1, "主题索引层", "任务族聚合")
tier(428, 2, "历史经历层", "五要素联合绑定")
tier(683, 3, "事件演化层", "阶段顺序与状态")
tier(938, 4, "多源事实层", "来源、时间与空间")

# The top-down references terminate at particular displayed record/process/obs
# markers. Pale fan boundaries emulate the reference diagrams' graph language.
curve([(715, 187), (715, 258), (480, 280), (480, 368)], lw=1.05)
curve([(715, 187), (715, 258), (960, 280), (960, 368)], lw=1.05)
line((704, 184), (318, 401), dashed=True, lw=.55, color="#A7ADB1", z=1)
line((726, 184), (1138, 401), dashed=True, lw=.55, color="#A7ADB1", z=1)
txt(528, 288, "索引经历甲", 8.4, MUTED)
txt(920, 288, "索引经历乙", 8.4, MUTED)

# Layer 1. Topic is the existing task key, not a learned topic cluster.
blob([(624, 157), (667, 142), (765, 142), (806, 157),
      (839, 174), (822, 209), (790, 218),
      (746, 231), (645, 223), (618, 209),
      (595, 196), (596, 169), (624, 157), (624, 157)], PURPLE, alpha=.73)
node(715, 182, "拥堵分析主题", dy=-31, rx=13, ry=7, fs=11)
txt(715, 212, "同一任务键", 8.5, MUTED)
txt(980, 190, "主题 → 历史经历", 9.5, MUTED)

# Layer 2. Two episode hyperedges overlap at exactly the two drawn shared nodes.
blob([(333, 383), (417, 339), (625, 351), (747, 389),
      (798, 407), (793, 495), (743, 508),
      (623, 540), (380, 524), (330, 497),
      (292, 476), (293, 412), (333, 383), (333, 383)], GREEN)
blob([(690, 389), (813, 351), (1020, 339), (1104, 383),
      (1144, 412), (1145, 476), (1107, 497),
      (1057, 524), (814, 540), (694, 508),
      (645, 495), (640, 407), (690, 389), (690, 389)], BLUE)
record(480, 379, "甲")
record(960, 379, "乙")
txt(480, 352, "历史经历甲", 11, weight="bold")
txt(960, 352, "历史经历乙", 11, weight="bold")
node(383, 420, "适用时间甲", dy=-22)
node(417, 484, "行动甲", dy=20)
node(561, 484, "结果甲", dy=20)
node(1057, 420, "适用时间乙", dy=-22)
node(879, 484, "行动乙", dy=20)
node(1023, 484, "结果乙", dy=20)
node(718, 420, "地点：路段甲", dy=-24, fill="#FFF9E5", rx=11)
node(718, 482, "事件：拥堵", dy=23, fill="#FFF9E5", rx=11)
# Membership guides are deliberately sparse; closed contours carry the binding.
for x, targets in [(480, [(383, 420), (417, 484), (561, 484), (718, 420), (718, 482)]),
                   (960, [(1057, 420), (879, 484), (1023, 484), (718, 420), (718, 482)])]:
    for target in targets:
        line((x, 388), target, color="#91A1A2", lw=.55, z=4)
txt(480, 550, "经历超边甲", 8.5, "#5F7860")
txt(960, 550, "经历超边乙", 8.5, "#577A91")
txt(719, 550, "共享条件", 8.4, "#8B744A")

# Same spatial entity, displayed again in the evidence tier. Route this reference
# through the empty central gutter rather than through either event case.
curve([(729, 420), (839, 454), (732, 714), (739, 933)],
      color="#B09B6D", lw=.65, z=1)
line((739, 933), (715, 969), color="#B09B6D", lw=.65, dashed=True, z=1)
txt(744, 822, "同一实体引用", 7.9, "#9A8457")

# Layer 3. Each episode owns a separate event case. The selected first phase
# is shown twice as a reference, not duplicated in storage.
for cx, stages, hue in [(480, ("增长", "恢复", "稳定"), GREEN),
                         (960, ("增长", "持续", "稳定"), BLUE)]:
    curve([(cx, 389), (cx-10, 482), (cx+10, 565), (cx, 622)], lw=1.1)
    blob([(cx-139, 643), (cx-111, 602), (cx+98, 599), (cx+136, 643),
          (cx+166, 677), (cx+181, 728), (cx+147, 748),
          (cx+90, 776), (cx-95, 777), (cx-150, 748),
          (cx-181, 728), (cx-167, 674), (cx-139, 643), (cx-139, 643)], YELLOW, alpha=.72)
    node(cx, 632, "具体事件甲" if cx == 480 else "具体事件乙", dy=-27, rx=11, fs=10.2)
    positions = [cx-125, cx, cx+125]
    for x, label in zip(positions, stages):
        line((cx, 639), (x, 707), dashed=True, lw=.7)
        node(x, 712, label, dy=-22, fill=hue if label == "增长" else "white", rx=10)
    for a, b in zip(positions, positions[1:]):
        # A transition hyperedge contains source, target and direction descriptor.
        blob([(a-13, 704), (a+19, 692), (b-15, 692), (b+13, 705),
              (b+30, 724), (a+92, 750), (a+63, 751),
              (a+30, 750), (a-30, 724), (a-13, 704), (a-13, 704)], PINK, alpha=.66, z=3)
        line((a+11, 712), (b-11, 712), arrow=True, lw=.85, color=INK, z=7)
        node((a+b)/2, 743, rx=6, ry=3.5, lw=.75)
    txt(cx, 780, "阶段先后 · 状态与证据绑定", 8.5, "#817457")

# Explicit phase-to-evidence trace, one for each independent event.
curve([(355, 719), (355, 807), (470, 816), (470, 895)], lw=1.2, color="#527E72")
curve([(835, 719), (835, 807), (960, 816), (960, 895)], lw=1.2, color="#527E72")
txt(486, 829, "阶段甲引用观测甲", 8.5, "#527E72")
txt(973, 829, "阶段乙引用观测乙", 8.5, "#527E72")

# Layer 4. Independent observations retain own provenance and timestamps;
# observation and spatial-relation hyperedges share the same entity reference.
blob([(428, 893), (459, 870), (481, 873), (510, 894),
      (563, 929), (722, 939), (744, 971),
      (765, 1000), (619, 1026), (485, 1025),
      (385, 1024), (302, 1000), (319, 973),
      (338, 944), (399, 918), (428, 893), (428, 893)], BLUE)
blob([(920, 894), (949, 873), (971, 870), (1002, 893),
      (1031, 918), (1092, 944), (1111, 973),
      (1128, 1000), (1045, 1024), (945, 1025),
      (811, 1026), (665, 1000), (686, 971),
      (708, 939), (867, 929), (920, 894), (920, 894)], GREEN)
blob([(703, 961), (741, 944), (864, 975), (902, 995),
      (932, 1012), (940, 1045), (900, 1047),
      (836, 1056), (720, 1007), (703, 987),
      (695, 978), (695, 969), (703, 961), (703, 961)], YELLOW, alpha=.83, z=3)
node(470, 901, "历史观测甲", dy=-24, rx=11, fs=10.2)
node(960, 901, "历史观测乙", dy=-24, rx=11, fs=10.2)
node(351, 970, "门架来源", dy=23)
node(494, 996, "观测时刻甲", dy=25)
node(1080, 970, "事件记录来源", dy=23)
node(943, 996, "观测时刻乙", dy=25)
node(715, 975, "路段甲", dy=-23, fill="#FFF9E5", rx=11)
node(888, 1020, "路段乙", dy=25, rx=9)
node(815, 1015, rx=6, ry=3.5)
txt(814, 1044, "上游关系", 7.8, "#8B744A")
for a, ends in [((470, 907), [(351, 970), (494, 996), (715, 975)]),
                ((960, 907), [(1080, 970), (943, 996), (715, 975)])]:
    for b in ends:
        line(a, b, color="#839998", lw=.6)
line((726, 978), (877, 1017), color="#8B744A", lw=.85, arrow=True, z=7)
txt(563, 942, "观测超边甲", 8, "#577A91")
txt(884, 942, "观测超边乙", 8, "#5F7860")

# Compact retrieval sequence; this is a simplified ordering of implementation.
steps = [
    (175, "当前任务", "时间 · 空间 · 问题 · 任务信息", PURPLE),
    (310, "条件与当前证据过滤", "检查时效、阶段、状态与来源", YELLOW),
    (445, "相关性召回", "可选语义向量／离线词法匹配", PURPLE),
    (580, "超边绑定与邻域扩展", "同任务族；至少两类兼容共享条件", GREEN),
    (715, "联合排序", "相关性 ＋ 条件覆盖 ＋ 联合支持", GREEN),
    (850, "Jev 单次复用判断", "当前查询 ＋ 候选经历 ＋ 证据", BLUE),
]
for y, title, sub, fill in steps:
    flow(y, title, sub, fill)
for (y, *_), (next_y, *_) in zip(steps, steps[1:]):
    line((1411, y+66), (1411, next_y-9), arrow=True, color="#54786A", lw=.9)
line((1411, 916), (1411, 954), arrow=True, color="#54786A")
txt(1411, 979, "复用 / 综合 / 停止", 11.5, weight="bold")
txt(1411, 1011, "输出前复核条件与图绑定", 8.4, MUTED)

# Diagram-to-retriever association, without implying record mixing.
line((1150, 454), (1218, 454), dashed=True, lw=.8)
line((1218, 454), (1218, 612), dashed=True, lw=.8)
line((1218, 612), (1267, 612), dashed=True, arrow=True, lw=.8)
txt(1201, 520, "共享\n条件", 8.1, MUTED)

# A concise legend, outside all graph layers.
node(302, 1088, rx=8)
txt(319, 1088, "节点", 8.3, ha="left")
ax.add_patch(Ellipse((407, 1088), 42, 19, facecolor=BLUE, edgecolor=INK, lw=.8, zorder=5))
txt(438, 1088, "彩色包络：超边", 8.3, ha="left")
line((614, 1088), (649, 1088), dashed=True, lw=.9)
txt(660, 1088, "跨层索引 / 证据引用", 8.3, ha="left")
line((879, 1088), (914, 1088), arrow=True, color=INK, lw=.9)
txt(925, 1088, "阶段先后 / 空间方向", 8.3, ha="left")
txt(1397, 1088, "四层展示 · SQLite 统一存储", 8, MUTED)
txt(743, 1110, "浅底平面：展示层；同名路段为同一实体引用；彩色包络展示超边的局部成员。", 7.4, MUTED)

prefix = ROOT / "hypergraph-memory-paper-flat"
fig.savefig(prefix.with_suffix(".svg"))
fig.savefig(prefix.with_suffix(".pdf"))
fig.savefig(prefix.with_suffix(".png"), dpi=300)
plt.close(fig)
print(prefix)
