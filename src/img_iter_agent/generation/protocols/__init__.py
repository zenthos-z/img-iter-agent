"""四协议族 dispatcher。

每个 dispatcher 实现 `generate(req, *, client, out_dir) -> GeneratedImage`，
把统一 GenRequest 翻译成该族的请求体，调用 client，把返回(url/b64)落盘成文件路径。

各族差异（doc.dmxapi.cn 实测核对）：
  A. family_a_openai   /v1/images/{generations|edits}  Bearer  prompt  multipart edits  size: 1024x1024
  B. family_b_doubao   /v1/responses  Authorization:<key>(无Bearer)  input:string  image=URL/base64  size: 2K/2048x2048
  C. family_c_qwen     /v1/responses(同B端点,提示词嵌套)  input.messages[].content[].text  size: 宽*高(星号!)
  D. family_d_gemini   /v1beta/models/<m>:generateContent  x-goog-api-key  contents[].parts[]  size: imageConfig
"""
