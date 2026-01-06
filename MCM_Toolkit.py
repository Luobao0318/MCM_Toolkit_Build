import sys
import os
import re
import traceback
import io
import numpy as np
import pandas as pd
import matplotlib

# 强制使用 Qt 后端
matplotlib.use('QtAgg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QPushButton, QLabel, QMessageBox, QSplitter, 
                             QCheckBox, QComboBox, QFileDialog, QTabWidget, QGroupBox,
                             QListWidget, QSlider, QDialog)
from PyQt6.QtGui import QFont, QColor, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtCore import Qt, QObject, pyqtSignal
import qdarkstyle

# ==========================================
# 0. 系统稳定性
# ==========================================

class EmittingStream(QObject):
    """
    重定向 stdout/stderr 到 GUI，防止 noconsole 模式下 print 导致崩溃
    """
    textWritten = pyqtSignal(str)

    def write(self, text):
        # 这里的 try-except 防止在程序关闭时写入流导致崩溃
        try:
            self.textWritten.emit(str(text))
        except:
            pass
    
    def flush(self):
        pass

# 全局异常钩子：捕获所有未处理异常，防止程序静默退出
def global_exception_handler(exctype, value, traceback_obj):
    error_msg = "".join(traceback.format_exception(exctype, value, traceback_obj))
    print(f"[CRITICAL ERROR] {error_msg}") # 这会尝试输出到 GUI 控制台
    # 在 GUI 线程中尝试弹窗（如果是严重错误）
    # 注意：这里不直接弹窗防止递归崩溃，仅做记录

sys.excepthook = global_exception_handler

# ==========================================
# 1. 逻辑
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
            r'Axes3D': "from mpl_toolkits.mplot3d import Axes3D",
            r'scipy\.stats': "from scipy import stats"
        }
        for pattern, stmt in mapping.items():
            if re.search(pattern, code) and stmt not in code:
                header += stmt + "\n"
                logs.append(f"Auto-Fix: Added '{stmt}'")
        return header + code, logs

    @staticmethod
    def apply_academic_style(style_type="std", palette="deep"):
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
                params['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette(palette))
                
            plt.rcParams.update(params)
            sns.set_context("paper", font_scale=1.2)
            sns.set_style("ticks")
        except Exception:
            pass

    @staticmethod
    def beautify_figure(fig):
        for ax in fig.axes:
            if hasattr(ax, 'get_zlim'):
                CodeProcessor.beautify_3d_figure(ax)
            else:
                CodeProcessor.beautify_2d_figure(ax)

    @staticmethod
    def beautify_2d_figure(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
        if not ax.get_xlabel(): ax.set_xlabel("Variable X", color='gray', fontstyle='italic')
        if not ax.get_ylabel(): ax.set_ylabel("Variable Y", color='gray', fontstyle='italic')

    @staticmethod
    def beautify_3d_figure(ax):
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
        ax.zaxis.labelpad = 10

    @staticmethod
    def auto_annotate_peaks(fig):
        count = 0
        for ax in fig.axes:
            if hasattr(ax, 'get_zlim'): continue
            lines = ax.get_lines()
            for line in lines:
                x = line.get_xdata()
                y = line.get_ydata()
                if len(y) < 5: continue
                idx = np.argmax(y)
                ax.annotate(f'{y[idx]:.2f}', xy=(x[idx], y[idx]), 
                            xytext=(x[idx], y[idx]+(np.max(y)-np.min(y))*0.1),
                            arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5))
                count += 1
        return count

# ==========================================
# 2. UI 组件
# ==========================================

class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rules = []
        fmt_kw = QTextCharFormat()
        fmt_kw.setForeground(QColor("#ff79c6"))
        fmt_kw.setFontWeight(QFont.Weight.Bold)
        for w in ["def", "class", "if", "else", "for", "while", "import", "return", "try", "except", "from", "as"]:
            self.rules.append((f"\\b{w}\\b", fmt_kw))
        fmt_str = QTextCharFormat()
        fmt_str.setForeground(QColor("#f1fa8c"))
        self.rules.append((r"\".*\"", fmt_str))
        self.rules.append((r"\'.*\'", fmt_str))
        fmt_com = QTextCharFormat()
        fmt_com.setForeground(QColor("#6272a4"))
        self.rules.append((r"#[^\n]*", fmt_com))

    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)

class MCMToolkitWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCM_Toolkit (O-Award Edition)")
        self.setGeometry(50, 50, 1600, 1000)
        self.context = {} 
        self.current_fig = None # 关键修复：持久化存储当前 Figure 对象
        
        # 流重定向
        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr
        self.output_stream = EmittingStream()
        self.output_stream.textWritten.connect(self.append_console)
        sys.stdout = self.output_stream
        sys.stderr = self.output_stream

        self.init_ui()
        app = QApplication.instance()
        app.setStyle("Fusion")
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))

    def closeEvent(self, event):
        # 恢复标准输出，防止退出时崩溃
        sys.stdout = self.sys_stdout
        sys.stderr = self.sys_stderr
        event.accept()

    def append_console(self, text):
        cursor = self.console.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Splitter: Top (Work) / Bottom (Console)
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0,0,0,0)
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # === LEFT PANEL ===
        left_panel = QWidget()
        l_layout = QVBoxLayout(left_panel)
        
        tool_layout = QHBoxLayout()
        self.btn_run = QPushButton("▶ Run Code")
        self.btn_run.setStyleSheet("background-color: #2e7d32; color: white; padding: 6px; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_code)
        
        self.btn_anno = QPushButton("✨ Annotate")
        self.btn_anno.clicked.connect(self.run_annotate)
        
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(["deep", "muted", "bright", "pastel", "dark", "viridis", "magma"])
        
        tool_layout.addWidget(self.btn_run)
        tool_layout.addWidget(self.btn_anno)
        tool_layout.addWidget(QLabel("Palette:"))
        tool_layout.addWidget(self.combo_theme)

        # Tabs
        self.tabs = QTabWidget()
        tab_code = QWidget()
        t1_layout = QVBoxLayout(tab_code)
        
        quick_btns = QHBoxLayout()
        for label, func in [("📂 Data", self.data_wizard), ("📈 Sens", lambda: self.insert_template("sensitivity")), ("🕸 Net", lambda: self.insert_template("network"))]:
            btn = QPushButton(label)
            btn.clicked.connect(func)
            quick_btns.addWidget(btn)
        
        self.editor = QTextEdit()
        self.editor.setFont(QFont("Consolas", 11))
        self.highlighter = PythonHighlighter(self.editor.document())
        self.editor.setPlaceholderText("# Python Code Here...")
        self.editor.setText(self.get_template("3d_surface"))
        
        t1_layout.addLayout(quick_btns)
        t1_layout.addWidget(self.editor)
        self.tabs.addTab(tab_code, "Editor")
        
        l_layout.addLayout(tool_layout)
        l_layout.addWidget(self.tabs)

        # === RIGHT PANEL ===
        right_panel = QWidget()
        r_layout = QVBoxLayout(right_panel)
        
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(5, 4), dpi=100))
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # 3D Controls
        ctrl_group = QGroupBox("Controls & Export")
        c_layout = QHBoxLayout()
        
        self.sl_elev = QSlider(Qt.Orientation.Horizontal)
        self.sl_elev.setRange(0, 90); self.sl_elev.setValue(30)
        self.sl_elev.valueChanged.connect(self.update_3d_view)
        
        self.sl_azim = QSlider(Qt.Orientation.Horizontal)
        self.sl_azim.setRange(-180, 180); self.sl_azim.setValue(-45)
        self.sl_azim.valueChanged.connect(self.update_3d_view)
        
        self.btn_pdf = QPushButton("💾 PDF")
        self.btn_pdf.clicked.connect(self.export_pdf)
        self.btn_latex = QPushButton("📝 LaTeX")
        self.btn_latex.clicked.connect(self.show_latex)
        
        c_layout.addWidget(QLabel("Elev"))
        c_layout.addWidget(self.sl_elev)
        c_layout.addWidget(QLabel("Azim"))
        c_layout.addWidget(self.sl_azim)
        c_layout.addWidget(self.btn_pdf)
        c_layout.addWidget(self.btn_latex)
        ctrl_group.setLayout(c_layout)
        
        r_layout.addWidget(self.toolbar)
        r_layout.addWidget(self.canvas)
        r_layout.addWidget(ctrl_group)
        
        h_splitter.addWidget(left_panel)
        h_splitter.addWidget(right_panel)
        h_splitter.setSizes([500, 800])
        top_layout.addWidget(h_splitter)

        # === BOTTOM CONSOLE ===
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        self.console.setFixedHeight(120)
        
        v_splitter.addWidget(top_widget)
        v_splitter.addWidget(self.console)
        main_layout.addWidget(v_splitter)

    # --- Core Logic ---

    def run_code(self):
        self.console.clear()
        print(">>> Running...")
        
        raw_code = self.editor.toPlainText()
        code, logs = CodeProcessor.auto_fix_imports(raw_code)
        for l in logs: print(l)
        
        is_3d = "mplot3d" in code or "projection='3d'" in code or "projection=\"3d\"" in code
        CodeProcessor.apply_academic_style("3d" if is_3d else "std", self.combo_theme.currentText())
        
        try:
            plt.close('all')
            self.canvas.figure.clf()
        except: pass

        try:
            # Safe Context
            def no_show(*args, **kwargs): pass
            ctx = self.context.copy()
            ctx.update({'plt': plt, 'show': no_show})
            
            exec(code, ctx)
            self.context.update(ctx)
            
            if 'fig' in ctx and isinstance(ctx['fig'], Figure):
                fig = ctx['fig']
            else:
                fig = plt.gcf()
            
            # 关键：保存引用，防止GC回收导致导出失败
            self.current_fig = fig
            
            CodeProcessor.beautify_figure(fig)
            self.refresh_canvas(fig)
            
            print(">>> Done.")
        except Exception as e:
            print(f"Error:\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Error", "Check console for details.")

    def refresh_canvas(self, fig):
        try:
            layout = self.canvas.parent().layout()
            layout.removeWidget(self.canvas)
            layout.removeWidget(self.toolbar)
            self.canvas.deleteLater()
            self.toolbar.deleteLater()
            
            self.canvas = FigureCanvasQTAgg(fig)
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            
            layout.insertWidget(0, self.toolbar)
            layout.insertWidget(1, self.canvas)
            
            # Sync Sliders
            for ax in fig.axes:
                if hasattr(ax, 'get_zlim'):
                    self.sl_elev.setValue(int(ax.elev) if ax.elev else 30)
                    self.sl_azim.setValue(int(ax.azim) if ax.azim else -45)
                    break
        except Exception as e:
            print(f"Canvas Refresh Error: {e}")

    def update_3d_view(self):
        if not self.current_fig: return
        needs_draw = False
        for ax in self.current_fig.axes:
            if hasattr(ax, 'get_zlim'):
                ax.view_init(elev=self.sl_elev.value(), azim=self.sl_azim.value())
                needs_draw = True
        if needs_draw: self.canvas.draw_idle()

    def run_annotate(self):
        if not self.current_fig: return
        c = CodeProcessor.auto_annotate_peaks(self.current_fig)
        self.canvas.draw()
        print(f"Annotated {c} peaks.")

    def export_pdf(self):
        """修复后的 PDF 导出功能"""
        if not self.current_fig:
            QMessageBox.warning(self, "Warning", "No figure generated yet.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save PDF", "figure.pdf", "PDF (*.pdf)")
        if path:
            try:
                # 显式使用当前保存的 Figure 对象
                self.current_fig.savefig(path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Saved to:\n{path}")
                print(f"Exported PDF to {path}")
            except Exception as e:
                err = traceback.format_exc()
                print(err)
                QMessageBox.critical(self, "Export Error", f"Failed to save PDF:\n{str(e)}")

    def show_latex(self):
        try:
            title = self.current_fig.gca().get_title() if self.current_fig else "MCM Figure"
        except: title = "MCM Figure"
        
        label = re.sub(r'[^a-zA-Z0-9]', '_', title).lower()
        txt = f"\\begin{{figure}}[htbp]\n  \\centering\n  \\includegraphics[width=0.8\\textwidth]{{figure.pdf}}\n  \\caption{{{title}}}\n  \\label{{fig:{label}}}\n\\end{{figure}}"
        
        d = QDialog(self)
        l = QVBoxLayout(d)
        t = QTextEdit(); t.setPlainText(txt)
        l.addWidget(QLabel("LaTeX Code:")); l.addWidget(t)
        d.resize(500, 300); d.exec()

    def data_wizard(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Data (*.csv *.xlsx)")
        if path:
            cmd = f"pd.read_csv(r'{path}')" if path.endswith('.csv') else f"pd.read_excel(r'{path}')"
            self.editor.insertPlainText(f"\ndf = {cmd}\nprint(df.head())\n")

    def insert_template(self, key):
        self.editor.insertPlainText("\n" + self.get_template(key))

    def get_template(self, key):
        if key == "3d_surface":
            return "# 3D Surface\nfrom mpl_toolkits.mplot3d import Axes3D\nX, Y = np.meshgrid(np.linspace(-2,2,30), np.linspace(-2,2,30))\nZ = X * np.exp(-X**2 - Y**2)\nfig = plt.figure(figsize=(8,6))\nax = fig.add_subplot(111, projection='3d')\nax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none', alpha=0.9)\nax.set_title('3D Analysis')"
        if key == "sensitivity":
            return "# Sensitivity\nx = np.linspace(0,10,100)\ny = np.sin(x)\nplt.figure(figsize=(8,5))\nplt.plot(x, y, label='Base')\nplt.fill_between(x, y-0.2, y+0.2, alpha=0.2, label='Confidence')\nplt.legend()"
        if key == "network":
            return "# Network\nimport networkx as nx\nG = nx.karate_club_graph()\npos=nx.spring_layout(G)\nplt.figure()\nnx.draw(G, pos, with_labels=True, node_color='skyblue')"
        return ""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    w = MCMToolkitWindow()
    w.show()
    sys.exit(app.exec())
