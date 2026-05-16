# -*- mode: python ; coding: utf-8 -*-
import os

# 收集 static 文件，排除 uploads（用户数据）
static_files = []
static_dir = 'static'
for root, dirs, files in os.walk(static_dir):
    # 排除 uploads 目录
    if 'uploads' in dirs:
        dirs.remove('uploads')
    for f in files:
        src = os.path.join(root, f)
        dst = os.path.relpath(src, '.')
        static_files.append((src, dst))

a = Analysis(
    ['server.py'],
    pathex=[],
    binaries=[],
    datas=[('templates', 'templates')] + static_files,
    hiddenimports=[
        'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto', 
        'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto', 
        'uvicorn.protocols.websockets', 'uvicorn.protocols.websockets.auto', 
        'uvicorn.protocols.websockets.wsproto', 'uvicorn.lifespan', 'uvicorn.lifespan.on',
        'agent', 'state_manager',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ArchAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
