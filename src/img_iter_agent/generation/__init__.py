"""生图适配层：屏蔽 dmxapi 四协议族差异。

  base.py        统一 GenRequest/GeneratedImage/ImageGenerator/ModelFamily/SizeSpec
  image_io.py    路径↔base64/url↔落盘（ADR-005）
  client.py      dmxapi 底层 HTTP（按族换认证头 + 重试 + 可注入 executor）
  router.py      按 task-mode + model_hint 选协议族 dispatcher
  protocols/     四族 dispatcher（A/B/C/D），各把统一请求翻译成该族请求体
"""
