import sys
import os
import re
import traceback
import numpy as np
import pandas as pd
import matplotlib

# 显式导入 PDF 后端并禁用交互模式（防止弹出 Figure 1）
import matplotlib.backends.backend_pdf 
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
plt.ioff() 

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QPushButton, QLabel, QMessageBox, QSplitter, 
                             QComboBox, QFileDialog, QListWidget, QGroupBox)
from PyQt6.QtGui import QFont, QColor, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtCore import Qt
import qdarkstyle

# ==========================================
# 1. 代码纠错与自动补全
# ==========================================

class CodeProcessor:
    @staticmethod
    def auto_fix_code(code):
        """算法驱动的自动纠错：修复拼写错误并自动补全 Import"""
        logs = []
        # 常见拼写纠正
        typo_map = {
            r'\bplt\.ploting\b': 'plt.plot',
            r'\bnp\.linepace\b': 'np.linspace',
            r'\bpd\.read_csc\b': 'pd.read_csv',
            r'\bplt\.tight_lyout\b': 'plt.tight_layout',
            r'\bplt\.histgram\b': 'plt.hist',
            r'\bax\.set_titl\b': 'ax.set_title',
            r'\bfig\.add_subp\b': 'fig.add_subplot',
            r'\bplt\.show\(\)\b': '# plt.show() removed'
        }
        for typo, correct in typo_map.items():
            if re.search(typo, code):
                code = re.sub(typo, correct, code)
                logs.append(f"Auto-Fix: Corrected typo to '{correct}'")

        # 自动补全 Import
        header = "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport networkx as nx\n"
        import_mapping = {
            r'Axes3D': "from mpl_toolkits.mplot3d import Axes3D",
            r'stats\.': "from scipy import stats",
            r'Sankey': "from matplotlib.sankey import Sankey",
            r'WordCloud': "from wordcloud import WordCloud",
            r'gaussian_kde': "from scipy.stats import gaussian_kde"
        }
        for pattern, stmt in import_mapping.items():
            if re.search(pattern, code) and stmt not in code:
                header += stmt + "\n"
                logs.append(f"Auto-Fix: Added missing import '{stmt}'")
        
        return header + "\n" + code, logs

    @staticmethod
    def apply_academic_style(palette="deep"):
        """风格设置"""
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
# 2. 语法高亮组件
# ==========================================

class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rules = []
        keyword_fmt = QTextCharFormat(); keyword_fmt.setForeground(QColor("#ff79c6")); keyword_fmt.setFontWeight(QFont.Weight.Bold)
        for w in ["def", "class", "if", "else", "for", "while", "import", "return", "from", "as", "with"]:
            self.rules.append((f"\\b{w}\\b", keyword_fmt))
        string_fmt = QTextCharFormat(); string_fmt.setForeground(QColor("#f1fa8c"))
        self.rules.append((r"\".*\"", string_fmt)); self.rules.append((r"\'.*\'", string_fmt))
        comment_fmt = QTextCharFormat(); comment_fmt.setForeground(QColor("#6272a4"))
        self.rules.append((r"#[^\n]*", comment_fmt))

    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)

# ==========================================
# 3. 主程序窗口
# ==========================================

class MCMPlotterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCM/ICM Algorithm Plotting Laboratory (Ultra-HD Edition)")
        self.setGeometry(100, 100, 1600, 950)
        self.current_fig = None
        self.templates = self.init_templates()
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 顶部工具栏
        top_bar = QHBoxLayout()
        self.btn_run = QPushButton("▶ 运行脚本 (RUN)"); self.btn_run.clicked.connect(self.run_code)
        self.btn_run.setStyleSheet("background-color: #2e7d32; font-weight: bold; height: 35px;")
        
        self.combo_palette = QComboBox()
        self.combo_palette.addItems(["deep", "muted", "bright", "pastel", "dark", "viridis", "magma"])
        
        self.btn_export_png = QPushButton("🖼 导出 PNG (600 DPI)"); self.btn_export_png.clicked.connect(self.export_png)
        self.btn_export_pdf = QPushButton("💾 导出 PDF (矢量)"); self.btn_export_pdf.clicked.connect(self.export_pdf)
        
        top_bar.addWidget(self.btn_run)
        top_bar.addWidget(QLabel("配色方案:"))
        top_bar.addWidget(self.combo_palette)
        top_bar.addWidget(self.btn_export_png)
        top_bar.addWidget(self.btn_export_pdf)

        # 主内容区分割
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧：列表 + 编辑器
        left_box = QWidget(); left_layout = QVBoxLayout(left_box)
        self.list_tpl = QListWidget(); self.list_tpl.addItems(sorted(self.templates.keys()))
        self.list_tpl.setFixedHeight(250); self.list_tpl.itemDoubleClicked.connect(self.apply_template)
        
        self.editor = QTextEdit(); self.editor.setFont(QFont("Consolas", 11))
        self.highlighter = PythonHighlighter(self.editor.document())
        self.editor.setText(self.templates["📈 折线图 (Line Plot)"])
        
        left_layout.addWidget(QLabel("图表算法库 (双击载入):"))
        left_layout.addWidget(self.list_tpl)
        left_layout.addWidget(QLabel("Python 代码编辑器:"))
        left_layout.addWidget(self.editor)

        # 右侧：预览区
        right_box = QWidget(); self.right_layout = QVBoxLayout(right_box)
        self.canvas_placeholder = QLabel("点击运行生成高清预览"); self.canvas_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_layout.addWidget(self.canvas_placeholder)

        splitter.addWidget(left_box); splitter.addWidget(right_box)
        splitter.setSizes([600, 1000])

        # 底部控制台
        self.console = QTextEdit(); self.console.setReadOnly(True); self.console.setFixedHeight(120)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")

        layout.addLayout(top_bar)
        layout.addWidget(splitter)
        layout.addWidget(self.console)

    def init_templates(self):
        # 算法库
        t = {}
        # 基础
        t["📈 折线图 (Line Plot)"] = "plt.figure(figsize=(8,5))\nx = np.linspace(0,10,100)\nplt.plot(x, np.sin(x), lw=2, label='Data')\nplt.title('Academic Line Plot')\nplt.legend()"
        t["📍 带标记折线图"] = "plt.figure()\nplt.plot([1,2,3,4], [1,4,2,3], marker='o', mfc='white', ms=8, mew=2)\nplt.title('Line with Markers')"
        t["☁️ 带阴影标记图"] = "x = np.linspace(0, 10, 20)\ny = np.sin(x)\nplt.figure()\nplt.plot(x, y, 'o-')\nplt.fill_between(x, y-0.2, y+0.2, alpha=0.2)\nplt.title('Shadow Bound Plot')"
        t["🪜 阶梯图 (Stairs)"] = "plt.figure()\nplt.step(range(10), np.random.rand(10), where='mid')"
        t["📐 面积图 (Area)"] = "plt.figure()\nplt.fill_between(range(10), np.random.rand(10), alpha=0.5)"
        t["📍 针状图 (Stem)"] = "plt.figure()\nplt.stem(range(10), np.random.rand(10))"
        
        # 柱状图类
        t["📊 柱状图 (单组多色)"] = "plt.figure()\nplt.bar(['A','B','C','D'], [10,25,15,30], color=sns.color_palette('viridis', 4))"
        t["📋 横向单组多色柱状图"] = "plt.figure()\nplt.barh(['A','B','C','D'], [10,25,15,30], color=sns.color_palette('rocket', 4))"
        t["📚 堆叠图 (Stacked)"] = "plt.figure()\nx=['G1','G2']\nplt.bar(x, [10,15], label='A'); plt.bar(x, [5,8], bottom=[10,15], label='B')\nplt.legend()"
        t["📑 堆叠图 (横向)"] = "plt.figure()\nx=['G1','G2']\nplt.barh(x, [10,15]); plt.barh(x, [5,8], left=[10,15])"
        t["➕ 正负柱状图"] = "plt.figure()\ny = np.random.uniform(-1,1,10)\nplt.bar(range(10), y, color=['r' if v<0 else 'g' for v in y])"
        
        # 3D类
        t["🧊 三维折线图"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\nz = np.linspace(0,10,100); ax.plot(np.sin(z), np.cos(z), z)"
        t["🌊 三维填充折线图"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\nx = np.linspace(0,10,50)\nfor i in range(3): ax.add_collection3d(plt.fill_between(x, 0, np.sin(x)+i, alpha=0.3), zs=i, zdir='y')"
        t["🏢 三维柱状图 (高度赋色)"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\nx, y = np.random.rand(2, 10); dz = np.random.rand(10)\nax.bar3d(x, y, np.zeros(10), 0.1, 0.1, dz, color=plt.cm.viridis(dz))"
        t["🏗 三维堆叠柱状图"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\nax.bar3d([0,1],[0,1],[0,0],0.5,0.5,[1,2],color='r'); ax.bar3d([0,1],[0,1],[1,2],0.5,0.5,[1,1],color='b')"
        t["⛰️ 曲面图 (Surface)"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\nX,Y = np.meshgrid(np.linspace(-2,2,30), np.linspace(-2,2,30))\nax.plot_surface(X, Y, X*np.exp(-X**2-Y**2), cmap='magma')"
        t["🕸 网格曲面图 (Wireframe)"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\nX,Y = np.meshgrid(np.linspace(-2,2,20), np.linspace(-2,2,20))\nax.plot_wireframe(X, Y, X+Y)"
        t["🌋 带等高线的曲面"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\nX,Y = np.meshgrid(np.linspace(-2,2,30), np.linspace(-2,2,30))\nZ = np.sin(X)*np.cos(Y)\nax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)\nax.contour(X, Y, Z, zdir='z', offset=-1.5, cmap='viridis')"

        # 统计类
        t["🏔 山脊图 (Ridgeline)"] = "plt.figure()\nfor i in range(5): sns.kdeplot(np.random.randn(100)+i, fill=True, alpha=0.5)"
        t["🕸 雷达图 (Radar)"] = "labels=['A','B','C','D']; stats=[20,30,40,10]; angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()\nstats+=stats[:1]; angles+=angles[:1]\nax = plt.subplot(111, polar=True); ax.fill(angles, stats, alpha=0.25); ax.plot(angles, stats, 'o-')"
        t["📦 箱线图 (多色填充)"] = "data = [np.random.normal(0, std, 100) for std in range(1, 4)]\nb = plt.boxplot(data, patch_artist=True)\nfor p, c in zip(b['boxes'], sns.color_palette('Set2', 3)): p.set_facecolor(c)"
        t["🔥 热力图 (Heatmap)"] = "sns.heatmap(np.random.rand(10,10), annot=False, cmap='YlGnBu')"
        t["🔍 局部放大图"] = "fig, ax = plt.subplots(); ax.plot(np.linspace(0,10,100), np.sin(np.linspace(0,10,100)))\naxins = ax.inset_axes([0.5, 0.5, 0.4, 0.4]); axins.plot(np.linspace(0,10,100), np.sin(np.linspace(0,10,100)))\naxins.set_xlim(2,3); axins.set_ylim(0.5,1); ax.indicate_inset_zoom(axins)"
        t["🫧 相关性气泡热图"] = "x, y = np.meshgrid(range(5), range(5)); z = np.random.rand(5,5)\nplt.scatter(x.flat, y.flat, s=z.flat*1000, c=z.flat, cmap='RdYlBu', alpha=0.6)"
        
        # 特殊图表
        t["🔀 桑基图 (Sankey)"] = "from matplotlib.sankey import Sankey\nSankey(flows=[0.25, 0.15, -0.20, -0.20], labels=['In1', 'In2', 'Out1', 'Out2']).finish()"
        t["☁️ 进阶词云图"] = "wc = WordCloud(background_color='white').generate('MCM ICM Math Model Award Python Plot')\nplt.imshow(wc); plt.axis('off')"
        t["🕸 有向图 (Directed)"] = "G = nx.DiGraph(); G.add_edges_from([(1,2),(2,3),(3,1)]); nx.draw(G, with_labels=True)"
        t["🌳 框架图 (Tree)"] = "G = nx.balanced_tree(r=2, h=3); nx.draw(G, with_labels=True, node_size=500)"
        t["🎨 伪彩图 (Pcolormesh)"] = "plt.pcolormesh(np.random.rand(20,20), cmap='magma')"
        t["🥧 三维饼图 (模拟)"] = "plt.pie([10,20,70], labels=['A','B','C'], shadow=True, explode=(0,0.1,0))"
        t["📊 直方图 (Histogram)"] = "plt.hist(np.random.randn(1000), bins=30, edgecolor='black', alpha=0.7)"
        t["✨ 散点图 (Scatter)"] = "plt.scatter(np.random.rand(50), np.random.rand(50), s=100, alpha=0.5)"
        t["🔘 极坐标散点图"] = "ax = plt.subplot(111, polar=True); ax.scatter(np.random.rand(50)*2*np.pi, np.random.rand(50))"
        t["🌌 三维散点图"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d'); ax.scatter(np.random.rand(50), np.random.rand(50), np.random.rand(50))"
        t["👥 分组散点图"] = "sns.scatterplot(x=np.random.rand(30), y=np.random.rand(30), hue=np.random.choice(['A','B'], 30))"
        
        return t

    def apply_template(self, item):
        self.editor.setText(self.templates[item.text()])

    def run_code(self):
        self.console.clear()
        raw_code = self.editor.toPlainText()
        processed_code, logs = CodeProcessor.auto_fix_code(raw_code)
        for l in logs: self.log(l)
        
        CodeProcessor.apply_academic_style(self.combo_palette.currentText())
        
        try:
            plt.close('all')
            # 建立沙盒环境
            exec_env = {'np': np, 'pd': pd, 'plt': plt, 'sns': sns, 'nx': nx, 'WordCloud': WordCloud}
            exec(processed_code, exec_env)
            
            fig = plt.gcf()
            self.update_canvas(fig)
            self.current_fig = fig
            self.log(">>> 执行成功！")
        except Exception as e:
            self.log(f"[ERROR] {traceback.format_exc()}")
            QMessageBox.critical(self, "运行时错误", str(e))

    def update_canvas(self, fig):
        # 清理右侧布局
        for i in reversed(range(self.right_layout.count())): 
            widget = self.right_layout.itemAt(i).widget()
            if widget: widget.setParent(None)
            
        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.right_layout.addWidget(self.toolbar)
        self.right_layout.addWidget(self.canvas)
        self.canvas.draw()

    def export_png(self):
        if not self.current_fig: return
        path, _ = QFileDialog.getSaveFileName(self, "导出 PNG", "plot_600dpi.png", "PNG (*.png)")
        if path:
            self.current_fig.savefig(path, dpi=600, bbox_inches='tight')
            self.log(f"已导出高清 PNG: {path}")

    def export_pdf(self):
        if not self.current_fig: return
        path, _ = QFileDialog.getSaveFileName(self, "导出 PDF", "plot_vector.pdf", "PDF (*.pdf)")
        if path:
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(path) as pdf:
                pdf.savefig(self.current_fig, bbox_inches='tight')
            self.log(f"已导出矢量 PDF: {path}")

    def log(self, msg):
        self.console.append(msg)

# ==========================================
# 4. 启动
# ==========================================

if __name__ == "__main__":
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))
    
    window = MCMPlotterApp()
    window.show()
    sys.exit(app.exec())
