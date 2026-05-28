import sys
import site

print(f"Python解释器: {sys.executable}")
print(f"Python版本: {sys.version}")
print(f"包安装路径: {site.getsitepackages()}")