import re
import unicodedata


def to_safe_path_segment(text: str, fallback: str = 'untitled', max_bytes: int = 255) -> str:
    """将任意字符串处理为标准UNIX路径名."""
    if text is None:
        text = ''

    s = unicodedata.normalize('NFKC', text)  # 量把全角/兼容字符收敛
    s = s.lower()  # 转小写
    s = re.sub(r'\s+', '-', s)  # 空白折叠为 -
    s = re.sub(r'[^a-z0-9\u4e00-\u9fff._-]', '-', s)  # 只保留 a-z 0-9 . _ - 中文(CJK Unified Ideographs)
    s = re.sub(r'-{2,}', '-', s)  # 合并连续的 -
    s = s.strip('._-')  # 去掉首尾的 . _ -

    # 避免特殊目录名
    if not s or s in {'.', '..'}:
        s = fallback

    # 避免以 - 开头, 防止命令行歧义
    if s.startswith('-'):
        s = '_' + s[1:]

    # 再清洗一次 fallback 场景
    s = s.strip('._-')
    if not s or s in {'.', '..'}:
        s = 'untitled'

    # 限制为 max_bytes 字节, 避免截断到非法 utf-8
    encoded = s.encode('utf-8')
    if len(encoded) > max_bytes:
        encoded = encoded[:max_bytes]
        while True:
            try:
                s = encoded.decode('utf-8')
                break
            except UnicodeDecodeError:
                encoded = encoded[:-1]

        s = s.strip('._-')
        if not s:
            s = 'untitled'

    return s
