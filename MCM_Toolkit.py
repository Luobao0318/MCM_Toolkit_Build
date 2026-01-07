import sys
import os
import re
import traceback
import io
import difflib # 用于拼写检查
import numpy as np
import pandas as pd

# 1. 强制配置 Matplotlib 后端，必须在导入 pyplot 之前
import matplotlib
matplotlib.use('QtAgg') 

# 2. 显式导入 PDF 后端
import matplotlib.backends.backend_pdf

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import networkx as nx

# 尝试导入 wordcloud，如果没有则占位
try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QPushButton, QLabel, QMessageBox, QSplitter, 
                             QCheckBox, QComboBox, QFileDialog, QTabWidget, QGroupBox,
                             QListWidget, QSlider, QDialog, QTreeWidget, QTreeWidgetItem, QHeaderView)
from PyQt6.QtGui import QFont, QColor, QSyntaxHighlighter, QTextCharFormat, QAction
from PyQt6.QtCore import Qt, QObject, pyqtSignal
import qdarkstyle

# ==========================================
# 0. 基础设施与稳定性
# ==========================================

class EmittingStream(QObject):
    textWritten = pyqtSignal(str)
    def write(self, text):
        try: self.textWritten.emit(str(text))
        except: pass
    def flush(self): pass

def global_exception_handler(exctype, value, traceback_obj):
    error_msg = "".join(traceback.format_exception(exctype, value, traceback_obj))
    print(f"[CRITICAL ERROR] {error_msg}")

sys.excepthook = global_exception_handler

# ==========================================
# 1. 代码处理核心
# ==========================================

class CodeProcessor:
    COMMON_TYPOS = {
        'improt': 'import', 'form': 'from', 'panda': 'pandas', 
        'nump': 'numpy', 'matplotlb': 'matplotlib', 'seborn': 'seaborn',
        'pltt': 'plt', 'fgi': 'fig', 'axex': 'axes', 'retun': 'return',
        'df.head()': 'print(df.head())'
    }

    @staticmethod
    def fuzzy_fix_code(code):
        """非AI的算法级拼写纠正"""
        lines = code.split('\n')
        fixed_lines = []
        logs = []

        for line in lines:
            # 1. 简单字典替换
            new_line = line
            for typo, correct in CodeProcessor.COMMON_TYPOS.items():
                pattern = r'\b' + re.escape(typo) + r'\b'
                if re.search(pattern, new_line):
                    new_line = re.sub(pattern, correct, new_line)
                    logs.append(f"Auto-Correct: '{typo}' -> '{correct}'")
            
            # 2. 检查常见函数拼写 (Levenshtein 简化版 - difflib)
            # 这里只做极其保守的替换，避免误伤变量名
            if "plt." in new_line:
                # 比如把 plot_suface -> plot_surface
                pass 
                
            fixed_lines.append(new_line)
        
        return "\n".join(fixed_lines), logs

    @staticmethod
    def auto_fix_imports(code):
        header = ""
        logs = []
        mapping = {
            r'\bnp\.' : "import numpy as np",
            r'\bpd\.' : "import pandas as pd",
            r'\bplt\.': "import matplotlib.pyplot as plt",
            r'\bsns\.': "import seaborn as sns",
            r'\bnx\.': "import networkx as nx",
            r'Axes3D': "from mpl_toolkits.mplot3d import Axes3D",
            r'stats\.': "from scipy import stats",
            r'WordCloud': "from wordcloud import WordCloud",
            r'Sankey': "from matplotlib.sankey import Sankey",
            r'inset_axes': "from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset",
            r'LinearSegmentedColormap': "from matplotlib.colors import LinearSegmentedColormap"
        }
        
        # 先进行模糊修复
        code, fuzzy_logs = CodeProcessor.fuzzy_fix_code(code)
        logs.extend(fuzzy_logs)

        # 补全 Import
        existing_imports = set(re.findall(r'^(?:import|from)\s+(\w+)', code, re.MULTILINE))
        
        for pattern, stmt in mapping.items():
            if re.search(pattern, code):
                module_name = stmt.split()[1]
                if stmt not in code and module_name not in existing_imports:
                    header += stmt + "\n"
                    logs.append(f"Auto-Import: Added '{stmt}'")
                    
        return header + code, logs

    @staticmethod
    def apply_academic_style(style_type="std", palette="deep"):
        """风格配置"""
        plt.rcParams.update(plt.rcParamsDefault) # 重置以防污染
        
        fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
        # 优先选择 Times New Roman，其次是系统衬线字体
        target_font = 'Times New Roman' if 'Times New Roman' in fonts else 'DejaVu Serif'
        
        params = {
            'font.family': 'serif',
            'font.serif': [target_font],
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'figure.dpi': 120,
            'savefig.dpi': 300,
            'axes.unicode_minus': False,
            'mathtext.fontset': 'stix',
            'figure.autolayout': True, # 自动调整布局防遮挡
        }
        
        # 调色板逻辑
        try:
            if style_type == "3d":
                params['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("viridis", 8))
            else:
                params['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette(palette))
        except:
            pass

        plt.rcParams.update(params)
        
        # Seaborn 覆盖
        sns.set_context("paper", font_scale=1.2)
        sns.set_style("ticks", {'font.family': 'serif', 'font.serif': [target_font]})

# ==========================================
# 2. 模板管理器
# ==========================================

class TemplateManager:
    """管理所有图表模板"""
    
    CATEGORIES = {
        "Base Charts": ["Line Chart", "Scatter Plot", "Bar Chart", "Pie Chart", "Histogram", "Boxplot"],
        "3D Visualization": ["3D Surface", "3D Line", "3D Scatter", "3D Bar (Colored)", "3D Stacked Bar"],
        "Advanced Analysis": ["Heatmap", "Correlation Bubble", "Ridge Plot (Joyplot)", "Violin Plot"],
        "Composition & Flow": ["Stacked Area", "Sankey Diagram", "Mosaic/Treemap Style"],
        "Comparison": ["Radar (Spider)", "Positive/Negative Bar", "Stairs Chart", "Grouped Scatter"],
        "Geospatial/Network": ["Network Graph", "Directed Graph", "Framework Diagram"],
        "Special": ["Word Cloud", "Zoom Inset", "Pseudo Color", "Contour Surface"]
    }

    @staticmethod
    def get_code(name):
        # --- Base ---
        if name == "Line Chart":
            return """# Standard Line Chart with Confidence Interval
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)
y_smooth = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y_smooth, label='Theoretical', color='#C0392B', linewidth=2)
plt.scatter(x, y, alpha=0.4, label='Observations', s=10, color='gray')
plt.fill_between(x, y_smooth - 0.2, y_smooth + 0.2, color='#E74C3C', alpha=0.2, label='95% CI')

plt.title("Experimental Results Analysis")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.legend(frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)"""

        if name == "Bar Chart":
            return """# Grouped Bar Chart
categories = ['Group A', 'Group B', 'Group C', 'Group D']
means_1 = [20, 35, 30, 35]
means_2 = [25, 32, 34, 20]
std_1 = [2, 3, 4, 1]
std_2 = [3, 5, 2, 3]

x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(10, 6))
rects1 = plt.bar(x - width/2, means_1, width, yerr=std_1, label='Method 1', capsize=5)
rects2 = plt.bar(x + width/2, means_2, width, yerr=std_2, label='Method 2', capsize=5)

plt.ylabel('Scores')
plt.title('Performance Comparison')
plt.xticks(x, categories)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)"""

        # --- 3D ---
        if name == "3D Surface":
            return """# 3D Surface with Projections
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False, alpha=0.8)
ax.contourf(X, Y, Z, zdir='z', offset=-1.2, cmap='viridis', alpha=0.5) # Projection on bottom

ax.set_zlim(-1.2, 1.2)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
fig.colorbar(surf, shrink=0.5, aspect=5)"""

        if name == "3D Bar (Colored)":
            return """# 3D Bar Chart with Height Coloring
x_edges = np.arange(10)
y_edges = np.arange(10)
xpos, ypos = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

dx = dy = 0.8 * np.ones_like(zpos)
dz = np.random.rand(len(xpos)) * 10

# Color by height
cmap = plt.cm.jet
colors = cmap(dz / dz.max())

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average')
ax.set_title("3D Distribution Map")"""

        # --- Comparison ---
        if name == "Radar (Spider)":
            return """# Radar/Spider Chart
labels = ['Strength', 'Speed', 'Agility', 'Intelligence', 'Endurance']
num_vars = len(labels)

values = [4, 5, 3, 4, 2]
values += values[:1] # Close the loop

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, values, color='red', linewidth=2)
ax.fill(angles, values, color='red', alpha=0.25)

ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title('Attribute Analysis', y=1.1)"""

        if name == "Stairs Chart":
            return """# Step/Stairs Chart
x = np.arange(14)
y = np.sin(x / 2)

plt.figure(figsize=(10, 6))
plt.step(x, y + 2, label='Pre (default)', where='pre')
plt.plot(x, y + 2, 'o--', color='grey', alpha=0.3)

plt.step(x, y + 1, label='Mid', where='mid')
plt.plot(x, y + 1, 'o--', color='grey', alpha=0.3)

plt.step(x, y, label='Post', where='post')
plt.plot(x, y, 'o--', color='grey', alpha=0.3)

plt.grid(axis='x', color='0.95')
plt.legend()
plt.title('Discrete Event Simulation')"""

        if name == "Ridge Plot (Joyplot)":
            return """# Ridge Plot / Joyplot (Pure Matplotlib Implementation)
# Generating mock data
np.random.seed(42)
dists = [np.random.normal(i, 0.5, 100) for i in range(5)]
labels = [f'Epoch {i+1}' for i in range(5)]

plt.figure(figsize=(8, 6))
ax = plt.gca()

# Manually creating ridges
colors = sns.color_palette("rocket", len(dists))
for i, (data, color) in enumerate(zip(dists, colors)):
    sns.kdeplot(data, fill=True, color=color, alpha=0.7, linewidth=1.5, edgecolor='white', zorder=10-i)
    # Visual offset trick handled by user or using joypy, but here is a standard KDE overlay
    # To do real "Ridge" vertical offset:
    
plt.title("Density Distribution Evolution")
plt.xlabel("Value")
plt.yticks([])
sns.despine(left=True)"""

        # --- Advanced ---
        if name == "Heatmap":
            return """# Correlation Heatmap
data = np.random.rand(10, 12)
corr = np.corrcoef(data)

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool)) # Half mask
sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.title("Feature Correlation Matrix")"""

        if name == "Zoom Inset":
            return """# Plot with Zoom Inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

x = np.linspace(0, 10, 1000)
y = np.sin(x) * np.exp(-x/3)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, label='Decay Signal')
ax.set_title("Signal Decay Analysis")

# Create inset
axins = ax.inset_axes([0.5, 0.5, 0.4, 0.4]) # x, y, width, height (relative)
axins.plot(x, y, color='r')

# Define zoom region
x1, x2, y1, y2 = 2, 4, 0.3, 0.8
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.grid(True)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.legend(loc='lower left')"""

        if name == "Word Cloud":
            return """# Word Cloud (Requires 'wordcloud' library)
try:
    from wordcloud import WordCloud
    text = "Math Modeling optimization algorithm data analysis sensitivity stability python visualization graph network chaos theory prediction" * 10
    
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Keywords WordCloud")
except ImportError:
    print("Error: 'wordcloud' library not installed. pip install wordcloud")"""

        if name == "Sankey Diagram":
            return """# Basic Sankey Diagram (Matplotlib)
from matplotlib.sankey import Sankey

plt.figure(figsize=(10, 6))
Sankey(flows=[0.25, 0.15, 0.60, -0.20, -0.15, -0.05, -0.50, -0.10],
       labels=['Input A', 'Input B', 'Input C', 'Loss 1', 'Loss 2', 'Loss 3', 'Output Main', 'Recycle'],
       orientations=[-1, 1, 0, 1, 1, 1, 0, -1],
       pathlengths=[0.25, 0.25, 0.25, 0.25, 0.25, 0.6, 0.25, 0.25],
       patchlabel="Flow System").finish()
plt.title("Energy/Material Flow Balance")"""

        if name == "Network Graph":
            return """# Network Graph Analysis
import networkx as nx

G = nx.karate_club_graph()
pos = nx.spring_layout(G, seed=42)
degrees = dict(G.degree)

plt.figure(figsize=(10, 8))
nx.draw_networkx_edges(G, pos, alpha=0.3)
nodes = nx.draw_networkx_nodes(G, pos, 
                       node_size=[v * 50 for v in degrees.values()],
                       node_color=list(degrees.values()), 
                       cmap=plt.cm.coolwarm)
nx.draw_networkx_labels(G, pos, font_size=8)

plt.colorbar(nodes, label='Node Degree')
plt.axis('off')
plt.title("Social Network Interaction")"""

        if name == "Framework Diagram":
            return """# Conceptual Framework (Directed Graph)
import networkx as nx

G = nx.DiGraph()
G.add_edges_from([
    ('Data Collection', 'Preprocessing'),
    ('Preprocessing', 'Model A'),
    ('Preprocessing', 'Model B'),
    ('Model A', 'Evaluation'),
    ('Model B', 'Evaluation'),
    ('Evaluation', 'Optimization'),
    ('Optimization', 'Final Result')
])

pos = nx.shell_layout(G)
plt.figure(figsize=(10, 6))

nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=3000, node_shape='s', 
        arrowstyle='->', arrowsize=20, font_size=10, font_weight='bold')
plt.title("Model Process Framework")"""

        # 默认返回空
        return f"# Code for {name} not found yet."

# ==========================================
# 3. GUI 主窗口
# ==========================================

class MCMToolkitWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCM/ICM Pro Toolkit (Auto-Correction & Templates)")
        self.setGeometry(50, 50, 1600, 1000)
        self.context = {} 
        self.current_fig = None 
        
        # 3. 重定向标准输出
        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr
        self.output_stream = EmittingStream()
        self.output_stream.textWritten.connect(self.append_console)
        sys.stdout = self.output_stream
        sys.stderr = self.output_stream

        self.init_ui()
        
        # 应用全局样式
        app = QApplication.instance()
        app.setStyle("Fusion")
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))

    def closeEvent(self, event):
        sys.stdout = self.sys_stdout
        sys.stderr = self.sys_stderr
        event.accept()

    def append_console(self, text):
        cursor = self.console.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Splitter: Top (Work) / Bottom (Console)
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Horizontal Splitter: Templates / Code / Plot
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # === 1. LEFT PANEL: Templates ===
        template_panel = QWidget()
        tp_layout = QVBoxLayout(template_panel)
        tp_layout.addWidget(QLabel("📚 Chart Templates"))
        
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Category / Chart")
        self.tree.itemDoubleClicked.connect(self.load_template)
        
        for category, charts in TemplateManager.CATEGORIES.items():
            cat_item = QTreeWidgetItem([category])
            cat_item.setBackground(0, QColor("#2c3e50"))
            cat_item.setForeground(0, QColor("white"))
            for chart in charts:
                QTreeWidgetItem(cat_item, [chart])
            self.tree.addTopLevelItem(cat_item)
        self.tree.expandAll()
        tp_layout.addWidget(self.tree)

        # === 2. MIDDLE PANEL: Code Editor ===
        code_panel = QWidget()
        cp_layout = QVBoxLayout(code_panel)
        
        # Toolbar
        tool_layout = QHBoxLayout()
        self.btn_run = QPushButton("▶ Run Code (F5)")
        self.btn_run.setStyleSheet("background-color: #2e7d32; color: white; padding: 6px; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_code)
        
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(["deep", "muted", "pastel", "bright", "dark", "colorblind"])
        
        self.btn_load_data = QPushButton("📂 Load Data")
        self.btn_load_data.clicked.connect(self.data_wizard)

        tool_layout.addWidget(self.btn_run)
        tool_layout.addWidget(self.btn_load_data)
        tool_layout.addWidget(QLabel("Palette:"))
        tool_layout.addWidget(self.combo_theme)
        
        self.editor = QTextEdit()
        self.editor.setFont(QFont("Consolas", 11))
        self.highlighter = PythonHighlighter(self.editor.document())
        self.editor.setText(TemplateManager.get_code("Line Chart")) # Default
        
        cp_layout.addLayout(tool_layout)
        cp_layout.addWidget(self.editor)

        # === 3. RIGHT PANEL: Visualization ===
        viz_panel = QWidget()
        vp_layout = QVBoxLayout(viz_panel)
        
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(5, 4), dpi=100))
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # Controls
        ctrl_group = QGroupBox("Export & View")
        c_layout = QHBoxLayout()
        self.btn_pdf = QPushButton("💾 PDF")
        self.btn_pdf.clicked.connect(self.export_pdf)
        self.btn_latex = QPushButton("LaTeX")
        self.btn_latex.clicked.connect(self.show_latex)
        
        c_layout.addWidget(self.btn_pdf)
        c_layout.addWidget(self.btn_latex)
        ctrl_group.setLayout(c_layout)
        
        vp_layout.addWidget(self.toolbar)
        vp_layout.addWidget(self.canvas)
        vp_layout.addWidget(ctrl_group)
        
        # Add to Splitter
        h_splitter.addWidget(template_panel)
        h_splitter.addWidget(code_panel)
        h_splitter.addWidget(viz_panel)
        h_splitter.setSizes([250, 500, 700])
        
        # Console
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        self.console.setFixedHeight(120)
        
        v_splitter.addWidget(h_splitter)
        v_splitter.addWidget(self.console)
        main_layout.addWidget(v_splitter)

        # Shortcuts
        self.btn_run.setShortcut("F5")

    # --- Core Logic ---

    def load_template(self, item, column):
        if item.childCount() == 0: # Is leaf
            name = item.text(0)
            code = TemplateManager.get_code(name)
            self.editor.setText(code)
            print(f">>> Loaded Template: {name}")

    def run_code(self):
        self.console.clear()
        print(">>> Analyzing & Running...")
        
        # 1. 预处理
        raw_code = self.editor.toPlainText()
        code, logs = CodeProcessor.auto_fix_imports(raw_code)
        for l in logs: print(l)
        
        # 2. 设置样式
        is_3d = "mplot3d" in code or "projection='3d'" in code or "projection=\"3d\"" in code
        CodeProcessor.apply_academic_style("3d" if is_3d else "std", self.combo_theme.currentText())
        
        # 3. 解决弹窗问题
        # 关闭之前的图，防止堆积
        plt.close('all')
        
        # 定义一个空的 show 函数，屏蔽代码中的 plt.show()
        def mock_show(*args, **kwargs):
            print(">>> plt.show() called (suppressed for GUI display).")

        # 4. 执行
        try:
            ctx = self.context.copy()
            # 注入环境
            ctx.update({
                'plt': plt, 
                'pd': pd, 
                'np': np, 
                'sns': sns,
                'show': mock_show
            })
            
            # 捕捉 plt.show()，重写它
            # 注意：不提前创建 figure，让用户代码去创建
            # 这样支持 plt.subplots() 或 plt.figure() 多种写法
            
            # --- EXECUTE ---
            exec(code, ctx)
            self.context = ctx # Update global context
            
            # --- CAPTURE ---
            # 获取当前活动的 Figure
            if plt.get_fignums():
                fig = plt.gcf()
                self.current_fig = fig
                self.refresh_canvas(fig)
                print(">>> Render Successful.")
            else:
                print(">>> Warning: No figure generated. Did you forget plt.plot()?")

        except Exception as e:
            print(f"Runtime Error:\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Execution Error", "Check console for details.")

    def refresh_canvas(self, fig):
        """将 Matplotlib 的 Figure 移植到 PyQt Canvas"""
        try:
            # 移除旧组件
            layout = self.canvas.parent().layout()
            layout.removeWidget(self.canvas)
            layout.removeWidget(self.toolbar)
            self.canvas.deleteLater()
            self.toolbar.deleteLater()
            
            # 创建新 Canvas (关联 Figure)
            self.canvas = FigureCanvasQTAgg(fig)
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            
            # 插入布局
            layout.insertWidget(0, self.toolbar)
            layout.insertWidget(1, self.canvas)
        except Exception as e:
            print(f"Canvas Refresh Error: {e}")

    def export_pdf(self):
        if not self.current_fig:
            QMessageBox.warning(self, "Warning", "No figure to save.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save PDF", "mcm_figure.pdf", "PDF (*.pdf)")
        if path:
            try:
                # 显式调用 PDF 后端保存
                self.current_fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Saved to:\n{path}")
                print(f"Exported: {path}")
            except Exception as e:
                print(traceback.format_exc())
                QMessageBox.critical(self, "Error", f"Save failed:\n{e}")

    def show_latex(self):
        title = "Figure"
        try: title = self.current_fig.gca().get_title()
        except: pass
        label = re.sub(r'\W+', '_', title).lower()
        
        txt = (
            f"\\begin{{figure}}[htbp]\n"
            f"  \\centering\n"
            f"  \\includegraphics[width=0.8\\textwidth]{{mcm_figure.pdf}}\n"
            f"  \\caption{{{title}}}\n"
            f"  \\label{{fig:{label}}}\n"
            f"\\end{{figure}}"
        )
        d = QDialog(self)
        d.setWindowTitle("LaTeX Code")
        l = QVBoxLayout(d)
        t = QTextEdit(); t.setPlainText(txt); t.setFont(QFont("Consolas", 10))
        l.addWidget(t)
        d.resize(500, 300); d.exec()

    def data_wizard(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Data", "", "Data (*.csv *.xlsx *.txt)")
        if path:
            if path.endswith('.csv'):
                cmd = f"df = pd.read_csv(r'{path}')"
            elif path.endswith('.xlsx'):
                cmd = f"df = pd.read_excel(r'{path}')"
            else:
                cmd = f"df = pd.read_table(r'{path}')"
            
            self.editor.insertPlainText(f"\n# Load Data\n{cmd}\nprint(df.head())\nprint(df.info())\n")

# ==========================================
# 4. 辅助高亮类
# ==========================================

class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rules = []
        
        fmt_kw = QTextCharFormat(); fmt_kw.setForeground(QColor("#ff79c6")); fmt_kw.setFontWeight(QFont.Weight.Bold)
        keywords = ["def", "class", "if", "else", "elif", "for", "while", "import", "return", "try", "except", "from", "as", "pass", "break", "continue"]
        for w in keywords: self.rules.append((f"\\b{w}\\b", fmt_kw))
        
        fmt_str = QTextCharFormat(); fmt_str.setForeground(QColor("#f1fa8c"))
        self.rules.append((r"\".*?\"", fmt_str))
        self.rules.append((r"\'.*?\'", fmt_str))
        
        fmt_com = QTextCharFormat(); fmt_com.setForeground(QColor("#6272a4"))
        self.rules.append((r"#[^\n]*", fmt_com))
        
        fmt_func = QTextCharFormat(); fmt_func.setForeground(QColor("#8be9fd"))
        self.rules.append((r"\b[A-Za-z0-9_]+(?=\()", fmt_func))

    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MCMToolkitWindow()
    window.show()
    sys.exit(app.exec())
