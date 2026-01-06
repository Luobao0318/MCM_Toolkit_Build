import sys
import os
import re
import datetime
import traceback
import io
import numpy as np
import pandas as pd
import matplotlib

# Force Qt Backend for Windows
matplotlib.use('QtAgg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QPushButton, QLabel, QMessageBox, QSplitter, 
                             QCheckBox, QComboBox, QFileDialog, QTabWidget, QGroupBox,
                             QListWidget, QSpinBox, QDialog, QInputDialog)
from PyQt6.QtGui import QFont, QColor, QSyntaxHighlighter, QTextCharFormat, QAction
from PyQt6.QtCore import Qt
import qdarkstyle

# ==========================================
# 0. 关键修复：防止 print 卡死 EXE
# ==========================================
class StreamRedirector(io.StringIO):
    """
    将 print 的输出吞掉，或者重定向到日志窗口。
    防止 noconsole 模式下 stdout 缓冲区满导致程序挂起。
    """
    def write(self, txt):
        # 这里可以选择将 print 内容记录下来，或者直接忽略
        pass 

# 如果没有控制台 (打包版), 则重定向 stdout/stderr
if sys.stdout is None or sys.stderr is None:
    sys.stdout = StreamRedirector()
    sys.stderr = StreamRedirector()

# ==========================================
# 1. Logic Engine (MCM_Toolkit Core)
# ==========================================

class CodeProcessor:
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
            r'Axes3D': "from mpl_toolkits.mplot3d import Axes3D"
        }
        for pattern, stmt in mapping.items():
            if re.search(pattern, code) and stmt not in code:
                header += stmt + "\n"
                logs.append(f"Auto-Fix: Added '{stmt}'")
        return header + code, logs

    @staticmethod
    def apply_academic_style(style_type="std"):
        try:
            fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
            target_font = 'Times New Roman' if 'Times New Roman' in fonts else 'DejaVu Serif'
            
            params = {
                'font.family': 'serif',
                'font.serif': [target_font],
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'legend.fontsize': 12,
                'figure.dpi': 100,
                'axes.unicode_minus': False,
                'mathtext.fontset': 'stix',
                'figure.constrained_layout.use': True,
            }
            
            if style_type == "3d":
                params['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("flare", 6))
            else:
                params['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep"))
                
            plt.rcParams.update(params)
            sns.set_context("paper", font_scale=1.2)
            sns.set_style("ticks")
        except Exception:
            pass # 字体加载失败不应崩溃

    @staticmethod
    def beautify_figure(fig):
        for ax in fig.axes:
            if hasattr(ax, 'get_zlim'):
                CodeProcessor.beautify_3d_figure(ax)
            else:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.2)
                ax.spines['bottom'].set_linewidth(1.2)
                ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
                
                for line in ax.get_lines():
                    if line.get_linewidth() < 1.5: line.set_linewidth(2.0)
                    line.set_antialiased(True)
                
                if not ax.get_xlabel(): ax.set_xlabel("Variable X", color='gray', fontstyle='italic')
                if not ax.get_ylabel(): ax.set_ylabel("Variable Y", color='gray', fontstyle='italic')

    @staticmethod
    def beautify_3d_figure(ax):
        ax.view_init(elev=30, azim=-45)
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
        ax.zaxis.labelpad = 10
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    @staticmethod
    def auto_annotate_peaks(fig):
        count = 0
        for ax in fig.axes:
            if hasattr(ax, 'get_zlim'): continue
            
            lines = ax.get_lines()
            for line in lines:
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                
                if len(y_data) < 5: continue
                
                max_idx = np.argmax(y_data)
                x_peak, y_peak = x_data[max_idx], y_data[max_idx]
                
                ax.annotate(f'Max: {y_peak:.2f}', 
                            xy=(x_peak, y_peak), 
                            xytext=(x_peak, y_peak + (max(y_data)-min(y_data))*0.1),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                            fontsize=10, 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
                count += 1
        return count

# ==========================================
# 2. GUI Components
# ==========================================

class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rules = []
        fmt_kw = QTextCharFormat()
        fmt_kw.setForeground(QColor("#d81b60"))
        fmt_kw.setFontWeight(QFont.Weight.Bold)
        keywords = ["def", "class", "if", "else", "for", "while", "import", "return", "try", "except"]
        for w in keywords:
            self.rules.append((f"\\b{w}\\b", fmt_kw))
        
        fmt_str = QTextCharFormat()
        fmt_str.setForeground(QColor("#43a047"))
        self.rules.append((r"\".*\"", fmt_str))
        self.rules.append((r"\'.*\'", fmt_str))
        
        fmt_com = QTextCharFormat()
        fmt_com.setForeground(QColor("#757575"))
        self.rules.append((r"#[^\n]*", fmt_com))

    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)

class MCMToolkitWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCM_Toolkit (Stable Edition)")
        self.setGeometry(50, 50, 1600, 950)
        self.context = {} 
        self.init_ui()
        app = QApplication.instance()
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # === Left Panel ===
        left_panel = QWidget()
        l_layout = QVBoxLayout(left_panel)
        
        tool_layout = QHBoxLayout()
        self.btn_run = QPushButton("▶ Run & Beautify")
        self.btn_run.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold; padding: 6px;")
        self.btn_run.clicked.connect(self.run_code)
        
        self.btn_annotate = QPushButton("✨ Auto-Annotate")
        self.btn_annotate.clicked.connect(self.run_annotate)
        
        tool_layout.addWidget(self.btn_run)
        tool_layout.addWidget(self.btn_annotate)
        
        self.tabs = QTabWidget()
        
        # Editor Tab
        tab_code = QWidget()
        t1_layout = QVBoxLayout(tab_code)
        
        quick_layout = QHBoxLayout()
        self.btn_data = QPushButton("📂 Import Data")
        self.btn_data.clicked.connect(self.data_wizard)
        self.btn_sens = QPushButton("📈 Sensitivity Wizard")
        self.btn_sens.clicked.connect(self.insert_sensitivity_template)
        quick_layout.addWidget(self.btn_data)
        quick_layout.addWidget(self.btn_sens)
        
        self.editor = QTextEdit()
        self.editor.setFont(QFont("Consolas", 11))
        self.highlighter = PythonHighlighter(self.editor.document())
        self.editor.setPlaceholderText("# Paste your Python code here...")
        self.editor.setText(self.get_template("3d_surface")) 
        
        t1_layout.addLayout(quick_layout)
        t1_layout.addWidget(self.editor)
        
        # Templates Tab
        tab_gallery = QWidget()
        t2_layout = QVBoxLayout(tab_gallery)
        self.list_gallery = QListWidget()
        self.templates = {
            "3D Response Surface (Sensitivity)": self.get_template("3d_surface"),
            "Panel Layout (GridSpec)": self.get_template("panel"),
            "Geospatial Map (No-GIS)": self.get_template("map"),
            "Time Series with Trends": self.get_template("timeseries")
        }
        self.list_gallery.addItems(self.templates.keys())
        self.list_gallery.itemDoubleClicked.connect(lambda item: self.editor.setPlainText(self.templates[item.text()]))
        t2_layout.addWidget(self.list_gallery)
        
        self.tabs.addTab(tab_code, "💻 Editor")
        self.tabs.addTab(tab_gallery, "📚 Templates")
        
        l_layout.addLayout(tool_layout)
        l_layout.addWidget(self.tabs)

        # === Right Panel ===
        right_panel = QWidget()
        r_layout = QVBoxLayout(right_panel)
        
        # Matplotlib Canvas
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(5, 4), dpi=100))
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        exp_layout = QHBoxLayout()
        self.btn_pdf = QPushButton("💾 Save PDF (Vector)")
        self.btn_pdf.clicked.connect(self.export_pdf)
        self.btn_latex = QPushButton("📝 Get LaTeX")
        self.btn_latex.clicked.connect(self.show_latex)
        exp_layout.addWidget(self.btn_pdf)
        exp_layout.addWidget(self.btn_latex)
        
        r_layout.addWidget(self.toolbar)
        r_layout.addWidget(self.canvas)
        r_layout.addLayout(exp_layout)
        
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left_panel)
        split.addWidget(right_panel)
        split.setSizes([600, 900])
        
        main_layout.addWidget(split)

    # --- 关键修复：安全的 Run Code ---
    def run_code(self):
        raw_code = self.editor.toPlainText()
        code, _ = CodeProcessor.auto_fix_imports(raw_code)
        
        is_3d = "mplot3d" in code or "projection='3d'" in code or "projection=\"3d\"" in code
        CodeProcessor.apply_academic_style("3d" if is_3d else "std")
        
        # 1. 安全清除旧图
        try:
            plt.close('all')
            self.canvas.figure.clf()
        except:
            pass

        try:
            # 2. 屏蔽 plt.show 防止阻塞
            def no_op_show(*args, **kwargs): pass
            
            # 3. 准备执行上下文
            # 我们注入一个假的 show，并确保 print 不会打印到黑洞
            safe_context = self.context.copy()
            safe_context.update({
                'plt': plt, 
                'show': no_op_show  # 覆盖 show
            })
            
            # 4. 执行
            exec(code, safe_context)
            
            # 更新全局上下文状态 (保持变量)
            self.context.update(safe_context)
            
            # 5. 获取图形
            if 'fig' in safe_context and isinstance(safe_context['fig'], Figure):
                fig = safe_context['fig']
            else:
                fig = plt.gcf()
            
            # 6. 美化
            CodeProcessor.beautify_figure(fig)
            
            # 7. 刷新画布 (使用更安全的方法)
            self.refresh_canvas(fig)
            
        except Exception as e:
            QMessageBox.critical(self, "Runtime Error", str(e) + "\n\n" + traceback.format_exc())

    def run_annotate(self):
        try:
            fig = self.canvas.figure
            count = CodeProcessor.auto_annotate_peaks(fig)
            self.canvas.draw()
            if count == 0:
                QMessageBox.information(self, "Info", "No suitable 2D peaks detected to annotate.")
        except Exception as e:
            print(e)

    def refresh_canvas(self, fig):
        # 移除旧组件
        layout = self.canvas.parent().layout()
        layout.removeWidget(self.canvas)
        layout.removeWidget(self.toolbar)
        
        # 彻底删除旧对象
        self.canvas.deleteLater()
        self.toolbar.deleteLater()
        
        # 创建新对象
        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # 插入回布局
        layout.insertWidget(0, self.toolbar)
        layout.insertWidget(1, self.canvas)

    def get_template(self, key):
        if key == "3d_surface":
            return "# === Sensitivity Analysis: 3D Response Surface ===\nfrom mpl_toolkits.mplot3d import Axes3D\n\nparam_alpha = np.linspace(0.5, 2.0, 30)\nparam_beta = np.linspace(0.5, 2.0, 30)\nX, Y = np.meshgrid(param_alpha, param_beta)\n\n# Objective Function\nR = np.sqrt(X**2 + Y**2)\nZ = np.sin(3*R) / R \n\nfig = plt.figure(figsize=(10, 8))\nax = fig.add_subplot(111, projection='3d')\n\nsurf = ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none', alpha=0.9)\n\nax.set_title('Sensitivity Analysis: Impact of Alpha & Beta', pad=20)\nax.set_xlabel('Parameter Alpha')\nax.set_ylabel('Parameter Beta')\nax.set_zlabel('Objective Function J')\nfig.colorbar(surf, shrink=0.5, aspect=10, label='Performance Metric')"
        elif key == "panel":
            return "# === Panel Composer ===\nfig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)\nax1, ax2, ax3, ax4 = axes.flatten()\n\nx = np.linspace(0, 10, 100)\nax1.plot(x, np.sin(x), 'b-', label='Model A')\nax1.set_title('(a) Primary Prediction')\nax1.legend()\n\nax2.plot(x, np.cos(x), 'r--', label='Model B')\nax2.set_title('(b) Alternative Scenario')\n\nsns.histplot(np.random.randn(500), ax=ax3, kde=True, color='green')\nax3.set_title('(c) Error Distribution')\n\ndata = np.random.rand(10, 10)\nsns.heatmap(data, ax=ax4, cmap='Blues', cbar=False)\nax4.set_title('(d) Correlation Matrix')\nfig.suptitle('Figure 1: Comprehensive Model Analysis', fontsize=16, weight='bold')"
        elif key == "map":
            return "# === Geospatial Map ===\nnp.random.seed(42)\nlon = np.random.uniform(-100, -80, 50)\nlat = np.random.uniform(30, 45, 50)\nvalues = np.random.rand(50) * 100\n\nfig, ax = plt.subplots(figsize=(10, 6))\npolygon_x = [-105, -75, -75, -105, -105]\npolygon_y = [25, 25, 50, 50, 25]\nax.fill(polygon_x, polygon_y, color='#f0f0f0', label='Region')\nax.plot(polygon_x, polygon_y, color='gray', linewidth=2)\n\nscatter = ax.scatter(lon, lat, s=values*5, c=values, cmap='coolwarm', alpha=0.8, edgecolors='black')\nplt.colorbar(scatter, label='Intensity')\nax.set_title('Geospatial Distribution')\nax.set_xlabel('Longitude')\nax.set_ylabel('Latitude')\nax.legend()"
        elif key == "timeseries":
            return "# === Time Series ===\nt = np.arange(0, 100)\ny = 0.5*t + np.random.normal(0, 5, 100)\nfig, ax = plt.subplots(figsize=(10, 5))\nax.plot(t, y, 'o', alpha=0.5, label='Observed')\nax.plot(t, 0.5*t, 'r-', linewidth=3, label='Trend')\nax.set_title('Time Series Forecasting')\nax.legend()"
        return ""

    def insert_sensitivity_template(self):
        code = "\n# === Automated Sensitivity Analysis ===\nk_values = np.linspace(0.5, 1.5, 5) \nt = np.linspace(0, 10, 100)\nfig, ax = plt.subplots(figsize=(10, 6))\nfor k in k_values:\n    y = 100 * np.exp(-k * t)\n    ax.plot(t, y, label=f'k = {k:.2f}')\nax.set_title('Sensitivity of Decay to Parameter k')\nax.legend(title='Parameter k')\n"
        self.editor.insertPlainText(code)

    def data_wizard(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Data", "", "Excel/CSV (*.xlsx *.csv)")
        if path:
            fn = os.path.basename(path)
            cmd = f"pd.read_csv(r'{path}')" if path.endswith('.csv') else f"pd.read_excel(r'{path}')"
            self.editor.insertPlainText(f"\n# Loaded: {fn}\ndf = {cmd}\nprint(df.head())\n")

    def export_pdf(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF", "figure.pdf", "PDF (*.pdf)")
        if path:
            self.canvas.figure.savefig(path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Success", f"Saved to:\n{path}")

    def show_latex(self):
        try:
            title = self.canvas.figure.gca().get_title()
        except:
            title = "MCM Figure"
        label = re.sub(r'[^a-zA-Z0-9]', '_', title).lower()
        latex = f"\\begin{{figure}}[htbp]\n  \\centering\n  \\includegraphics[width=0.8\\textwidth]{{figure.pdf}}\n  \\caption{{{title}}}\n  \\label{{fig:{label}}}\n\\end{{figure}}"
        QInputDialog.getText(self, "LaTeX Code", "Copy this:", text=latex)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    w = MCMToolkitWindow()
    w.show()
    sys.exit(app.exec())
