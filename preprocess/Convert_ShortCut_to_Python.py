import argparse
import os
import shortcuts
from shortcuts.utils import download_shortcut, is_shortcut_url
from tqdm import tqdm
import pandas as pd
from utils import *


def list_all_controls(folder_path, use_url=True, excel_path=None):
    set_controls = set()

    if use_url and excel_path:
        df = pd.read_excel(excel_path)
        for url in tqdm(df['iCloud Links']):
            shortcut = download_shortcut(url)
            shortcut = plistlib.loads(shortcut)

            for action in shortcut['WFWorkflowActions']:
                if 'GroupingIdentifier' in action['WFWorkflowActionParameters'] or 'WFControlFlowMode' in action['WFWorkflowActionParameters']:
                    set_controls.add(action['WFWorkflowActionIdentifier'])
            print(set_controls)
    else:
        items = os.listdir(folder_path)

        for file_name in items:
            if file_name.endswith(('.plist', '.shortcut')):
                with open(os.path.join(folder_path, file_name), 'rb') as f:
                    shortcut = f.read()
                try:
                    shortcut = plistlib.loads(shortcut)

                    for action in shortcut['WFWorkflowActions']:
                        if 'GroupingIdentifier' in action['WFWorkflowActionParameters'] or 'WFControlFlowMode' in action['WFWorkflowActionParameters']:
                            set_controls.add(action['WFWorkflowActionIdentifier'])
                    print(set_controls)
                except Exception as e:
                    print(f'Error at {file_name}: {e}')


def _get_format(filepath: str) -> str:
    '''
    Args:
        filepath: path for a file which format needs to be determined

    Returns:
        file format (shortcut, toml or url)
    '''
    if type(filepath).__name__ == 'dict':
        return 'dict'

    if is_shortcut_url(filepath):
        return 'url'

    if filepath == '':
        return shortcuts.FMT_JSON

    _, ext = os.path.splitext(filepath)
    ext = ext.strip('.')
    if ext in (shortcuts.FMT_SHORTCUT, 'plist'):
        return shortcuts.FMT_SHORTCUT
    elif ext == 'toml':
        return shortcuts.FMT_TOML
    elif ext == 'json':
        return shortcuts.FMT_JSON
    raise RuntimeError(f'Unsupported file format: {filepath}: "{ext}"')


# Replace UUID with variable names
def replace_uuids(code, replace_dict):
    for uuid, variable in replace_dict.items():
        code = re.sub(r'\b{}\b'.format(re.escape(uuid)), variable, code)
    return code


# Check if there are extra UUIDs
def find_extra_uuids(code):
    uuid_pattern = re.compile(r'\b[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}\b')
    found_uuids = set(re.findall(uuid_pattern, code))

    return len(found_uuids) != 0


def convert_shortcut(input_filepath: str, title: str,) -> str:
    input_format = _get_format(input_filepath)

    if input_format == 'dict':
        sc_data = input_filepath
        shortcut_dict = {
            'WFWorkflowActions': [sc_data]
        }
    else:
        if input_format == 'url':
            sc_data = download_shortcut(input_filepath)
        else:
            with open(input_filepath, 'rb') as f:
                sc_data = f.read()

        if isinstance(sc_data, str):
            sc_data = sc_data.encode('utf-8')

        shortcut_dict = plistlib.loads(sc_data)

    # Parse the shortcut script to AST
    parser = ShortcutParser(shortcut_dict['WFWorkflowActions'])
    try:
        parser.parse()
    except Exception as e:
        print(f'Input Error at {input_filepath}: {e}')
        return None

    parser.add_necessary_UUID(parser.root)
    parser.handle_contrl_UUID(parser.root)
    parser.add_empty_child(parser.root)
    parser.mark_multi_repeat(parser.root)

    python_code = parser.to_python()

    return python_code


def main():
    parser = argparse.ArgumentParser(description='Shortcuts')
    parser.add_argument('file', nargs='?', help='Input file: *.(|shortcut|itunes url)', default='*')
    parser.add_argument('--folder_path', help='Path to the folder containing shortcut files', default='../code/data/Merged Shortcuts Dataset')
    parser.add_argument('--excel_path', help='Path to the Excel file containing iCloud links', default=None)

    args = parser.parse_args()

    if args.file != '*':
        title = os.path.basename(args.file).split('.')[0]
        code = convert_shortcut(args.file, title)
        if code:
            print(code)
    else:
        folder_path = args.folder_path

        title2code = defaultdict(dict)

        for root, dirs, files in os.walk(folder_path):
            print(len(title2code))

            for file_name in files:
                if file_name.endswith(('.plist', '.shortcut')):
                    title = os.path.splitext(file_name)[0]
                    print(os.path.join(root, file_name))

                    with open(os.path.join(root, file_name), 'rb') as f:
                        shortcut = f.read()
                    try:
                        shortcut = plistlib.loads(shortcut)
                    except Exception as e:
                        print(f'Error loading {file_name}: {e}')
                        continue

                    code = convert_shortcut(os.path.join(root, file_name), title)

                    if code is not None:
                        title2code[title]['shortcut'] = shortcut
                        title2code[title]['code'] = code

        with open('case_study.pkl', 'wb') as file:
            pickle.dump(title2code, file)
        print('Final:', len(title2code))


if __name__ == '__main__':
    main()
