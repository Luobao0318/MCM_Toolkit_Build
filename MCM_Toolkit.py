import sys
import os
import re
import traceback
import numpy as np
import pandas as pd
import matplotlib

# ==========================================
# 0. 环境与渲染配置
# ==========================================
import matplotlib.backends.backend_pdf 
# 强制使用 Qt 后端并关闭交互模式，防止弹出独立的 Figure 1 窗口
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
plt.ioff() 

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
from scipy import stats

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QPushButton, QLabel, QMessageBox, QSplitter, 
                             QComboBox, QFileDialog, QListWidget, QGroupBox)
from PyQt6.QtGui import QFont, QColor, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtCore import Qt
import qdarkstyle

# ==========================================
# 1. 代码处理 (纠错与自动补全)
# ==========================================

class CodeProcessor:
    @staticmethod
    def auto_fix_code(code):
        """算法驱动的语法纠错与自动 Import"""
        logs = []
        # 常见拼写纠正字典
        typo_map = {
            r'\bplt\.ploting\b': 'plt.plot',
            r'\bnp\.linepace\b': 'np.linspace',
            r'\bpd\.read_csc\b': 'pd.read_csv',
            r'\bplt\.tight_lyout\b': 'plt.tight_layout',
            r'\bplt\.histgram\b': 'plt.hist',
            r'\bax\.set_titl\b': 'ax.set_title',
            r'\bfig\.add_subp\b': 'fig.add_subplot',
            r'\bplt\.show\(\)\b': '# plt.show() handled by GUI'
        }
        for typo, correct in typo_map.items():
            if re.search(typo, code):
                code = re.sub(typo, correct, code)
                logs.append(f"Auto-Fix: 修复拼写 '{correct}'")

        # 自动补全 Import
        header = "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport networkx as nx\nfrom mpl_toolkits.mplot3d import Axes3D\n"
        import_mapping = {
            r'Sankey': "from matplotlib.sankey import Sankey",
            r'WordCloud': "from wordcloud import WordCloud",
            r'stats\.': "from scipy import stats",
            r'gaussian_kde': "from scipy.stats import gaussian_kde"
        }
        for pattern, stmt in import_mapping.items():
            if re.search(pattern, code) and stmt not in code:
                header += stmt + "\n"
                logs.append(f"Auto-Fix: 补全模块 '{stmt}'")
        
        return header + "\n" + code, logs

    @staticmethod
    def apply_academic_style(palette="deep"):
        """样式配置"""
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'figure.dpi': 120,
            'axes.unicode_minus': False,
            'mathtext.fontset': 'stix',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--'
        })
        sns.set_palette(palette)

# ==========================================
# 2. UI 组件
# ==========================================

class PythonHighlighter(QSyntaxHighlighter):
    """语法高亮"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rules = []
        kw_fmt = QTextCharFormat(); kw_fmt.setForeground(QColor("#ff79c6")); kw_fmt.setFontWeight(QFont.Weight.Bold)
        for w in ["def", "class", "if", "else", "for", "while", "import", "return", "from", "as", "with"]:
            self.rules.append((f"\\b{w}\\b", kw_fmt))
        str_fmt = QTextCharFormat(); str_fmt.setForeground(QColor("#f1fa8c"))
        self.rules.append((r"\".*\"", str_fmt)); self.rules.append((r"\'.*\'", str_fmt))
        com_fmt = QTextCharFormat(); com_fmt.setForeground(QColor("#6272a4"))
        self.rules.append((r"#[^\n]*", com_fmt))

    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)

# ==========================================
# 3. 主程序
# ==========================================

class MCMPlotterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCM/ICM Algorithm Plotting Pro (O-Award Edition)")
        self.setGeometry(100, 100, 1600, 950)
        self.current_fig = None
        self.templates = self.init_templates()
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Toolbar
        t_bar = QHBoxLayout()
        self.btn_run = QPushButton("▶ 运行脚本 (RUN)"); self.btn_run.clicked.connect(self.run_code)
        self.btn_run.setStyleSheet("background-color: #2e7d32; font-weight: bold; height: 40px; color: white;")
        
        self.combo_pal = QComboBox()
        self.combo_pal.addItems(["deep", "muted", "bright", "pastel", "dark", "viridis", "magma"])
        
        self.btn_png = QPushButton("🖼 导出 PNG (600 DPI)"); self.btn_png.clicked.connect(self.export_png)
        self.btn_pdf = QPushButton("💾 导出 PDF (矢量)"); self.btn_pdf.clicked.connect(self.export_pdf)
        
        t_bar.addWidget(self.btn_run); t_bar.addWidget(QLabel("配色:")); t_bar.addWidget(self.combo_pal)
        t_bar.addStretch(); t_bar.addWidget(self.btn_png); t_bar.addWidget(self.btn_pdf)

        # Main Splitter
        split = QSplitter(Qt.Orientation.Horizontal)
        
        # Left Panel (List + Editor)
        l_box = QWidget(); l_lyt = QVBoxLayout(l_box)
        self.list_tpl = QListWidget(); self.list_tpl.addItems(sorted(self.templates.keys()))
        self.list_tpl.setFixedHeight(250); self.list_tpl.itemDoubleClicked.connect(self.load_tpl)
        self.editor = QTextEdit(); self.editor.setFont(QFont("Consolas", 11))
        self.highlighter = PythonHighlighter(self.editor.document())
        
        l_lyt.addWidget(QLabel("1. 图表功能模板库 (双击载入):"))
        l_lyt.addWidget(self.list_tpl)
        l_lyt.addWidget(QLabel("2. Python 算法编辑器:"))
        l_lyt.addWidget(self.editor)
        
        # Right Panel (Preview)
        self.r_box = QWidget(); self.r_lyt = QVBoxLayout(self.r_box)
        self.canvas_placeholder = QLabel("预览区域 (等待运行...)"); self.canvas_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.r_lyt.addWidget(self.canvas_placeholder)
        
        split.addWidget(l_box); split.addWidget(self.r_box)
        split.setSizes([550, 1050])
        
        # Console
        self.console = QTextEdit(); self.console.setReadOnly(True); self.console.setFixedHeight(130)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        
        layout.addLayout(t_bar); layout.addWidget(split); layout.addWidget(self.console)

    def init_templates(self):
        t = {}
        # --- 基础与折线类 ---
        t["📈 折线图 (Line)"] = "plt.figure()\nx = np.linspace(0,10,100)\nplt.plot(x, np.sin(x), lw=2, label='Sin Wave')\nplt.title('Basic Line Chart')\nplt.legend()"
        t["📍 带标记折线图"] = "plt.figure()\nplt.plot(np.arange(10), np.random.rand(10), 'o-', mfc='white', ms=8, mew=2)\nplt.title('Line with Markers')"
        t["☁️ 带阴影标记图"] = "x = np.linspace(0, 10, 20); y = np.sin(x)\nplt.figure()\nplt.plot(x, y, 'o-')\nplt.fill_between(x, y-0.2, y+0.2, alpha=0.2)\nplt.title('Shadow Bound Plot')"
        t["🪜 阶梯图 (Stairs)"] = "plt.figure()\nplt.step(range(10), np.random.rand(10), where='mid', lw=2)\nplt.title('Step Plot')"
        t["📐 面积图 (Area)"] = "plt.figure()\nplt.fill_between(range(10), np.random.rand(10), color='skyblue', alpha=0.5)\nplt.title('Area Chart')"
        t["📍 针状图 (Stem)"] = "plt.figure()\nplt.stem(range(10), np.random.randn(10))"
        
        # --- 柱状图类 ---
        t["📊 柱状图 (单组多色)"] = "plt.figure()\ncats = ['A','B','C','D','E']\nplt.bar(cats, np.random.rand(5)*10, color=sns.color_palette('viridis', 5))"
        t["📋 横向单组多色柱状图"] = "plt.figure()\ncats = ['A','B','C','D','E']\nplt.barh(cats, np.random.rand(5)*10, color=sns.color_palette('magma', 5))"
        t["📚 堆叠图 (Stacked)"] = "plt.figure()\nx = ['G1','G2','G3']\ny1, y2 = np.random.rand(3), np.random.rand(3)\nplt.bar(x, y1, label='Part A'); plt.bar(x, y2, bottom=y1, label='Part B')\nplt.legend()"
        t["📑 堆叠图 (横向)"] = "plt.figure()\nx = ['G1','G2','G3']\ny1, y2 = np.random.rand(3), np.random.rand(3)\nplt.barh(x, y1); plt.barh(x, y2, left=y1)"
        t["➕ 正负柱状图"] = "plt.figure()\ny = np.random.uniform(-5,5,10)\nplt.bar(range(10), y, color=['r' if v<0 else 'g' for v in y])\nplt.axhline(0, color='black', lw=1)"
        t["🏢 三维柱状图 (高度赋色)"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\nx, y = np.random.rand(2, 8); dz = np.random.rand(8)\nax.bar3d(x, y, np.zeros(8), 0.1, 0.1, dz, color=plt.cm.viridis(dz))"
        t["🏗 三维堆叠图 (3D Stacked)"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\nx, y = [0,1,2], [0,1,2]\nax.bar3d(x, y, np.zeros(3), 0.5, 0.5, [1,2,1], color='r', alpha=0.6)\nax.bar3d(x, y, [1,2,1], 0.5, 0.5, [2,1,2], color='b', alpha=0.6)"

        # --- 散点与极坐标 ---
        t["✨ 散点图 (Scatter)"] = "plt.figure()\nplt.scatter(np.random.rand(50), np.random.rand(50), s=np.random.rand(50)*200, alpha=0.6)"
        t["🔘 极坐标散点图"] = "plt.figure(); ax = plt.subplot(111, polar=True)\nax.scatter(np.random.rand(50)*2*np.pi, np.random.rand(50), color='r')"
        t["🌌 三维散点图"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\nax.scatter(np.random.rand(30), np.random.rand(30), np.random.rand(30), s=100)"
        t["👥 分组散点图"] = "df = pd.DataFrame({'x':np.random.rand(30), 'y':np.random.rand(30), 'g':np.random.choice(['A','B'],30)})\nsns.scatterplot(data=df, x='x', y='y', hue='g', s=100)"

        # --- 3D 填充与曲面 ---
        t["🌊 三维填充折线图 (Fixed)"] = """fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(0, 10, 100)
for i in range(4):
    y = np.sin(x + i) + 1.5
    # 在3D中使用add_collection3d投影2D路径
    art = ax.fill_between(x, 0, y, alpha=0.4)
    ax.add_collection3d(art, zs=i, zdir='y')
ax.set_ylim(0, 4); ax.set_xlabel('X'); ax.set_ylabel('Layer'); ax.set_zlabel('Value')
"""
        t["🧊 三维折线图"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\nz = np.linspace(0,10,100); ax.plot(np.sin(z), np.cos(z), z, lw=2)"
        t["⛰️ 曲面图 (Surface)"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\nX,Y = np.meshgrid(np.linspace(-2,2,40), np.linspace(-2,2,40))\nax.plot_surface(X, Y, X*np.exp(-X**2-Y**2), cmap='viridis')"
        t["🕸 网格曲面图"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\nX,Y = np.meshgrid(np.linspace(-2,2,20), np.linspace(-2,2,20))\nax.plot_wireframe(X, Y, X+Y, color='gray')"
        t["🌋 带等高线的曲面图"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\nX,Y = np.meshgrid(np.linspace(-2,2,30), np.linspace(-2,2,30))\nZ = np.sin(X)*np.cos(Y)\nax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)\nax.contour(X, Y, Z, zdir='z', offset=-1.5, cmap='coolwarm')"

        # --- 统计与高级类 ---
        t["🏔 山脊图 (Ridgeline)"] = "plt.figure(figsize=(8,5))\nfor i in range(5): sns.kdeplot(np.random.randn(100)+i*2, fill=True, alpha=0.6, label=f'C{i}')\nplt.title('Ridgeline Plot')"
        t["🕸 雷达图 (Radar/Spider)"] = "labels=['A','B','C','D','E']; stats=[20,34,30,35,27]; angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()\nstats+=stats[:1]; angles+=angles[:1]\nax=plt.subplot(111, polar=True); ax.fill(angles, stats, alpha=0.25); ax.plot(angles, stats, 'o-', lw=2)"
        t["🔥 热力图 (Heatmap)"] = "plt.figure(figsize=(8,6)); sns.heatmap(np.random.rand(10,10), cmap='YlGnBu', annot=False)"
        t["🫧 相关性气泡热图 (Fixed)"] = """x, y = np.meshgrid(range(6), range(6))
z = np.random.rand(6, 6)
plt.figure(figsize=(7,6))
plt.scatter(x.flatten(), y.flatten(), s=z.flatten()*1500, c=z.flatten(), cmap='RdYlBu', alpha=0.6, edgecolors='white')
plt.colorbar(label='Correlation Strength')
"""
        t["🔍 局部放大图 (Zoom)"] = "fig, ax = plt.subplots(); x=np.linspace(0,10,100); ax.plot(x, np.sin(x))\naxins = ax.inset_axes([0.6, 0.6, 0.35, 0.35]); axins.plot(x, np.sin(x))\naxins.set_xlim(2,4); axins.set_ylim(0.5,1.2); ax.indicate_inset_zoom(axins)"
        t["📦 箱线图 (Filled)"] = "data = [np.random.normal(0, std, 100) for std in range(1, 4)]\nb = plt.boxplot(data, patch_artist=True)\nfor patch, color in zip(b['boxes'], sns.color_palette('Set2')): patch.set_facecolor(color)"

        # --- 特殊类 ---
        t["🔀 桑基图 (Sankey)"] = "from matplotlib.sankey import Sankey\nplt.figure(); Sankey(flows=[0.25, 0.15, -0.2, -0.2], labels=['In1', 'In2', 'Out1', 'Out2']).finish()"
        t["☁️ 进阶词云图"] = "wc = WordCloud(background_color='white', width=800, height=400).generate('MCM ICM Math Python Modeling Award')\nplt.figure(figsize=(10,5)); plt.imshow(wc); plt.axis('off')"
        t["🕸 有向图 (Network)"] = "G = nx.DiGraph(); G.add_edges_from([(1,2),(2,3),(3,1),(1,4)]); plt.figure(); nx.draw(G, with_labels=True, node_color='orange')"
        t["🌳 框架图 (Tree)"] = "G = nx.balanced_tree(r=2, h=3); plt.figure(); nx.draw(G, with_labels=True, node_size=500, node_color='lightgreen')"
        t["🥧 饼图 (Pie)"] = "plt.figure(); plt.pie([15,30,45,10], labels=['A','B','C','D'], autopct='%1.1f%%', explode=[0,0.1,0,0])"
        t["🎂 三维饼图 (模拟)"] = "plt.figure(); plt.pie([20,50,30], labels=['X','Y','Z'], shadow=True, explode=(0.05,0.05,0.05))"
        t["📊 直方图 (Histogram)"] = "plt.figure(); plt.hist(np.random.randn(1000), bins=30, edgecolor='black', alpha=0.7)"
        t["🎨 伪彩图 (Pcolormesh)"] = "plt.figure(); plt.pcolormesh(np.random.rand(20,20), cmap='inferno')"
        
        return t

    def load_tpl(self, item):
        self.editor.setText(self.templates[item.text()])

    def run_code(self):
        self.console.clear()
        raw_code = self.editor.toPlainText()
        if not raw_code.strip(): return
        
        processed_code, logs = CodeProcessor.auto_fix_code(raw_code)
        for l in logs: self.log(l)
        
        CodeProcessor.apply_academic_style(self.combo_pal.currentText())
        
        try:
            # 清理
            plt.close('all')
            # 建立执行沙盒
            ctx = {'np': np, 'pd': pd, 'plt': plt, 'sns': sns, 'nx': nx, 'WordCloud': WordCloud, 'stats': stats}
            exec(processed_code, ctx)
            
            fig = plt.gcf()
            self.current_fig = fig
            self.update_canvas(fig)
            self.log(">>> [Success] 执行成功，已更新预览。")
        except Exception:
            err = traceback.format_exc()
            self.log(f"[Error] 脚本运行失败:\n{err}")
            QMessageBox.critical(self, "Runtime Error", "请检查控制台输出的错误信息。")

    def update_canvas(self, fig):
        for i in reversed(range(self.r_lyt.count())): 
            widget = self.r_lyt.itemAt(i).widget()
            if widget: widget.setParent(None)
            
        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.r_lyt.addWidget(self.toolbar)
        self.r_lyt.addWidget(self.canvas)
        self.canvas.draw()

    def export_png(self):
        if not self.current_fig: return
        path, _ = QFileDialog.getSaveFileName(self, "导出 PNG", "plot_600dpi.png", "PNG (*.png)")
        if path:
            self.current_fig.savefig(path, dpi=600, bbox_inches='tight')
            self.log(f"Export: 高清 PNG 已保存 -> {path}")

    def export_pdf(self):
        if not self.current_fig: return
        path, _ = QFileDialog.getSaveFileName(self, "导出 PDF", "plot_vector.pdf", "PDF (*.pdf)")
        if path:
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(path) as pdf:
                pdf.savefig(self.current_fig, bbox_inches='tight')
            self.log(f"Export: 矢量 PDF 已保存 -> {path}")

    def log(self, m):
        self.console.append(m)

# ==========================================
# 4. 启动程序
# ==========================================

if __name__ == "__main__":
    # 高 DPI 支持
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    
    app = QApplication(sys.argv)
    # 加载暗主题
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))
    
    win = MCMPlotterApp()
    win.show()
    sys.exit(app.exec())
