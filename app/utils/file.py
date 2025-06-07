import subprocess


def format_size(size_bytes: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == 'TB':
            return f'{size:.2f} {unit}'
        size /= 1024
    return ''


def get_folder_size_du(path: str) -> str:
    result = subprocess.run(['du', '-sb', path], stdout=subprocess.PIPE, text=True)
    size_in_bytes = int(result.stdout.split()[0])
    return format_size(size_in_bytes)
