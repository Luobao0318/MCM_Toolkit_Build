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
                             QComboBox, QFileDialog, QListWidget, QGroupBox, QTableWidget, QTableWidgetItem)
from PyQt6.QtGui import QFont, QColor, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtCore import Qt
import qdarkstyle

# ==========================================
# 1. 代码处理 (纠错与自动补全)
# ==========================================

class CodeProcessor:
    @staticmethod
    def auto_fix_code(code):
        """语法纠错与自动 Import"""
        logs = []
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
# 3. 主程序 (含数据导入功能)
# ==========================================

class MCMPlotterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCM/ICM Algorithm Plotting Pro (Data Wizard Edition)")
        self.setGeometry(100, 100, 1600, 950)
        self.current_fig = None
        self.current_df = None  # 存储导入的数据
        self.templates = self.init_templates()
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- Top Toolbar ---
        t_bar = QHBoxLayout()
        self.btn_run = QPushButton("▶ 运行绘图 (RUN)"); self.btn_run.clicked.connect(self.run_code)
        self.btn_run.setStyleSheet("background-color: #2e7d32; font-weight: bold; height: 40px; color: white;")
        
        self.combo_pal = QComboBox()
        self.combo_pal.addItems(["deep", "muted", "bright", "pastel", "dark", "viridis", "magma"])
        
        self.btn_png = QPushButton("🖼 导出 PNG (600 DPI)"); self.btn_png.clicked.connect(self.export_png)
        self.btn_pdf = QPushButton("💾 导出 PDF (矢量)"); self.btn_pdf.clicked.connect(self.export_pdf)
        
        t_bar.addWidget(self.btn_run); t_bar.addWidget(QLabel("配色:")); t_bar.addWidget(self.combo_pal)
        t_bar.addStretch(); t_bar.addWidget(self.btn_png); t_bar.addWidget(self.btn_pdf)

        # --- Middle Splitter ---
        split = QSplitter(Qt.Orientation.Horizontal)
        
        # --- Left Panel (Data + Template + Editor) ---
        l_box = QWidget(); l_lyt = QVBoxLayout(l_box)
        
        # Data Group
        data_group = QGroupBox("数据中心 (Data Wizard)")
        data_lyt = QVBoxLayout()
        data_btn_lyt = QHBoxLayout()
        self.btn_import = QPushButton("📂 导入 CSV/Excel"); self.btn_import.clicked.connect(self.import_data)
        data_btn_lyt.addWidget(self.btn_import)
        
        col_lyt = QHBoxLayout()
        self.cb_x = QComboBox(); self.cb_y = QComboBox(); self.cb_z = QComboBox()
        col_lyt.addWidget(QLabel("X:")); col_lyt.addWidget(self.cb_x)
        col_lyt.addWidget(QLabel("Y:")); col_lyt.addWidget(self.cb_y)
        col_lyt.addWidget(QLabel("Z:")); col_lyt.addWidget(self.cb_z)
        
        self.btn_apply_data = QPushButton("✨ 应用数据到模板"); self.btn_apply_data.clicked.connect(self.apply_data_to_code)
        self.btn_apply_data.setStyleSheet("background-color: #1565c0; color: white;")
        
        data_lyt.addLayout(data_btn_lyt)
        data_lyt.addLayout(col_lyt)
        data_lyt.addWidget(self.btn_apply_data)
        data_group.setLayout(data_lyt)
        
        # Template and Editor
        self.list_tpl = QListWidget(); self.list_tpl.addItems(sorted(self.templates.keys()))
        self.list_tpl.setFixedHeight(180); self.list_tpl.itemDoubleClicked.connect(self.load_tpl)
        self.editor = QTextEdit(); self.editor.setFont(QFont("Consolas", 11))
        self.highlighter = PythonHighlighter(self.editor.document())
        
        l_lyt.addWidget(data_group)
        l_lyt.addWidget(QLabel("1. 图表功能算法库 (双击载入样例):"))
        l_lyt.addWidget(self.list_tpl)
        l_lyt.addWidget(QLabel("2. Python 代码编辑器 (支持数据变量 'df'):"))
        l_lyt.addWidget(self.editor)
        
        # --- Right Panel (Preview) ---
        self.r_box = QWidget(); self.r_lyt = QVBoxLayout(self.r_box)
        self.canvas_placeholder = QLabel("绘图高清预览区域"); self.canvas_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.r_lyt.addWidget(self.canvas_placeholder)
        
        split.addWidget(l_box); split.addWidget(self.r_box)
        split.setSizes([600, 1000])
        
        # --- Bottom Console ---
        self.console = QTextEdit(); self.console.setReadOnly(True); self.console.setFixedHeight(130)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        
        layout.addLayout(t_bar); layout.addWidget(split); layout.addWidget(self.console)

    # ==========================================
    # 数据导入
    # ==========================================
    def import_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "Data Files (*.csv *.xlsx *.xls)")
        if not file_path: return
        
        try:
            if file_path.endswith('.csv'):
                self.current_df = pd.read_csv(file_path)
            else:
                self.current_df = pd.read_excel(file_path)
            
            cols = self.current_df.columns.tolist()
            self.cb_x.clear(); self.cb_y.clear(); self.cb_z.clear()
            self.cb_x.addItems(cols); self.cb_y.addItems(cols); self.cb_z.addItems(cols)
            
            self.log(f">>> [Data] 成功导入数据: {os.path.basename(file_path)} ({len(self.current_df)} 行)")
            QMessageBox.information(self, "成功", f"成功导入 {len(cols)} 列数据。请在下拉框中选择绘图字段。")
        except Exception as e:
            self.log(f">>> [Data Error] 导入失败: {e}")
            QMessageBox.critical(self, "错误", f"无法读取文件: {e}")

    def apply_data_to_code(self):
        """将用户选择的字段映射到代码中"""
        if self.current_df is None:
            QMessageBox.warning(self, "提醒", "请先导入数据文件！")
            return
        
        x_col = self.cb_x.currentText()
        y_col = self.cb_y.currentText()
        z_col = self.cb_z.currentText()
        
        # 自动生成数据代码段
        data_code = f"\n# --- Data Wizard Generated ---\n"
        data_code += f"x = df['{x_col}']\n"
        data_code += f"y = df['{y_col}']\n"
        
        # 根据当前编辑器内容简单判断是否需要Z轴
        if "projection='3d'" in self.editor.toPlainText() or "3d" in self.list_tpl.currentItem().text().lower():
            data_code += f"z = df['{z_col}']\n"
            
        data_code += "# -----------------------------\n"
        
        # 插入编辑器开头
        current_text = self.editor.toPlainText()
        # 移除之前的生成代码（如果存在）
        cleaned_text = re.sub(r'# --- Data Wizard Generated ---.*?# -----------------------------', '', current_text, flags=re.DOTALL)
        self.editor.setText(data_code + cleaned_text.strip())
        self.log(f">>> [Wizard] 已将字段 {x_col}, {y_col} 应用到编辑器")

    # ==========================================
    # 算法模板库
    # ==========================================
    def init_templates(self):
        t = {}
        t["📈 折线图 (Line)"] = "plt.figure()\n# 如果已导入数据并点击'应用数据'，下方x,y将被自动替换\nx = np.linspace(0,10,100)\ny = np.sin(x)\nplt.plot(x, y, lw=2, label='Dataset')\nplt.title('Academic Line Chart')\nplt.legend()"
        t["📊 柱状图 (Bar)"] = "plt.figure()\n# x轴常为分类，y轴为数值\nplt.bar(x, y, color=sns.color_palette('viridis', len(x)) if len(x)<20 else None)\nplt.title('Bar Chart')"
        t["✨ 散点图 (Scatter)"] = "plt.figure()\nplt.scatter(x, y, alpha=0.6, edgecolors='w')\nplt.title('Scatter Analysis')"
        t["🌊 三维填充折线图 (3D Fill)"] = """fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
# 样例：循环绘制多组填充
for i in range(4):
    curr_y = y + i # 演示偏移
    art = plt.fill_between(x, 0, curr_y, alpha=0.4)
    ax.add_collection3d(art, zs=i, zdir='y')
ax.set_xlabel('X'); ax.set_ylabel('Layer'); ax.set_zlabel('Value')
plt.close(plt.gcf().number if plt.gcf().number != fig.number else None)"""
        t["🫧 相关性气泡热图"] = """plt.figure(figsize=(7,6))
# 使用 flatten() 处理矩阵数据
plt.scatter(x.flatten() if hasattr(x, 'flatten') else x, 
            y.flatten() if hasattr(y, 'flatten') else y, 
            s=100, alpha=0.6, edgecolors='white')
plt.title('Bubble Correlation')"""
        t["⛰️ 曲面图 (3D Surface)"] = "fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')\n# 注意：曲面图通常需要网格化的X,Y,Z\nax.plot_surface(x, y, z, cmap='viridis') if 'z' in locals() else print('请选择Z轴数据')"
        t["🏔 山脊图 (Ridgeline)"] = "plt.figure(); \n# 假设数据包含分类，此处演示分组分布\nsns.kdeplot(data=df, x=self.cb_x.currentText(), hue=self.cb_y.currentText(), fill=True, alpha=0.5)"
        t["🕸 雷达图 (Radar)"] = "labels=x.values; stats=y.values; angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()\nstats=np.concatenate((stats,[stats[0]])); angles=np.concatenate((angles,[angles[0]]))\nax=plt.subplot(111, polar=True); ax.fill(angles, stats, alpha=0.3); ax.plot(angles, stats, 'o-')"
        t["🔥 热力图 (Heatmap)"] = "plt.figure(figsize=(10,8))\nsns.heatmap(df.corr(), annot=True, cmap='coolwarm')\nplt.title('Feature Correlation Matrix')"
        t["📦 箱线图 (Boxplot)"] = "plt.figure()\nsns.boxplot(data=df, x=self.cb_x.currentText(), y=self.cb_y.currentText())\nplt.title('Grouped Boxplot')"
        t["🔀 桑基图 (Sankey)"] = "from matplotlib.sankey import Sankey\nplt.figure(); Sankey(flows=[0.25, 0.15, -0.2, -0.2], labels=['In1', 'In2', 'Out1', 'Out2']).finish()"
        t["☁️ 进阶词云图"] = "text = ' '.join(df[self.cb_x.currentText()].astype(str))\nwc = WordCloud(background_color='white').generate(text)\nplt.imshow(wc); plt.axis('off')"
        
        t["🪜 阶梯图 (Stairs)"] = "plt.figure(); plt.step(x, y, where='mid')"
        t["📐 面积图 (Area)"] = "plt.figure(); plt.fill_between(x, 0, y, alpha=0.5)"
        t["➕ 正负柱状图"] = "plt.figure(); plt.bar(x, y, color=['r' if v<0 else 'g' for v in y])"
        t["🌳 框架图 (Tree)"] = "G = nx.balanced_tree(r=2, h=3); nx.draw(G, with_labels=True)"
        t["🥧 饼图 (Pie)"] = "plt.figure(); plt.pie(y[:5], labels=x[:5], autopct='%1.1f%%')"
        
        return t

    # ==========================================
    # 运行与执行逻辑
    # ==========================================
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
            plt.close('all')
            # 执行沙盒：注入 df 变量
            ctx = {
                'np': np, 'pd': pd, 'plt': plt, 'sns': sns, 'nx': nx, 
                'WordCloud': WordCloud, 'stats': stats,
                'df': self.current_df, 'self': self # 允许通过self访问UI状态
            }
            exec(processed_code, ctx)
            
            fig = plt.gcf()
            self.current_fig = fig
            self.update_canvas(fig)
            self.log(">>> [Success] 绘图已更新")
        except Exception:
            self.log(f"[Error] 运行失败:\n{traceback.format_exc()}")

    def update_canvas(self, fig):
        for i in reversed(range(self.r_lyt.count())): 
            w = self.r_lyt.itemAt(i).widget()
            if w: w.setParent(None)
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
            self.log(f">>> 已保存 PNG: {path}")

    def export_pdf(self):
        if not self.current_fig: return
        path, _ = QFileDialog.getSaveFileName(self, "导出 PDF", "plot_vector.pdf", "PDF (*.pdf)")
        if path:
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(path) as pdf:
                pdf.savefig(self.current_fig, bbox_inches='tight')
            self.log(f">>> 已保存 PDF: {path}")

    def log(self, m):
        self.console.append(m)

if __name__ == "__main__":
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))
    win = MCMPlotterApp()
    win.show()
    sys.exit(app.exec())
