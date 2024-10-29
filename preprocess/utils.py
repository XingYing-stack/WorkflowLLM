import copy
from datetime import datetime, timedelta
import re
from collections import defaultdict
import plistlib
import pickle
import json
import os.path
import uuid
from typing import Dict
from urllib.parse import urlparse
from urllib.request import urlopen

from shortcuts import exceptions

last_uuid = None

with open('identifier2keyparams.pkl', 'rb') as file:
    identifier2keyparams = pickle.load(file)

with open('identifier2output.pkl', 'rb') as file:
    identifier2output = pickle.load(file)



class FixedUUIDGenerator:
    def __init__(self):
        self.index = 0

    def generate_uuid(self):
        self.index += 1

        part5 = self.index & 0xFFFFFFFFFFFF

        part4 = (self.index >> 48) & 0xFFFF

        part3 = (self.index >> 64) & 0xFFFF

        part2 = (self.index >> 80) & 0xFFFF

        part1 = (self.index >> 96) & 0xFFFFFFFF

        part1_str = format(part1, '08X')
        part2_str = format(part2, '04X')
        part3_str = format(part3, '04X')
        part4_str = format(part4, '04X')
        part5_str = format(part5, '012X')

        return f'{part1_str}-{part2_str}-{part3_str}-{part4_str}-{part5_str}'


class Node:
    def __init__(self, action, params, index_id=None, children=None):
        self.action = action
        self.params = params
        self.index_id = index_id
        self.children = children if children is not None else []
        self.max_len = 384

    def __repr__(self):
        return f"Node(action={self.action}, index_id={self.index_id}, children={self.children})"


    def process_value(self, value):
        if isinstance(value, str):
            if re.match(r'[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}', value):
                return value
            if value.startswith("'''") and value.endswith("'''"):
                return value.strip("'")

            if "f'" in value or len(value) == 0 or value[0] in ('"', "'", '[', '{'):
                value = value
            else:
                value = "'''" + escape_single_quotes(value) + "'''"
            return value
        elif isinstance(value, list):
            return '[' + ', '.join(self.process_value(v) for v in value) + ']'
        elif isinstance(value, dict):
            return '{' + ', '.join(f'"{k}": {self.process_value(v)}' for k, v in value.items()) + '}'
        else:
            return repr(value)

    def add_quotation(self, value, uuid_map):
        pattern = r'[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}'
        value_repr = value
        is_uuid = False

        if isinstance(value_repr, str):
            matches = re.findall(pattern, value_repr)
            if len(matches) > 0:
                is_uuid = True

            if value_repr in uuid_map:
                is_uuid = True

        if (is_uuid or isinstance(value_repr, (int, float, bool)) or value_repr is None or value_repr == "" or
                (isinstance(value_repr, str) and ("f'" in value_repr or value_repr[0] in ('"', "'", "[", "{"))) or
                (isinstance(value, dict) and value.get('Value', {}).get('Type', {}) == 'Variable')):
            value_repr = value_repr
        else:
            value_repr = self.process_value(value_repr)
        return value_repr


    def action_to_function_call(self, action_identifier, parameters, uuid_map):

        action_identifier = action_identifier.replace('.', '_')
        function_call = f"{action_identifier}("

        for key, value in parameters.items():

            value_repr = self.get_variable(value)
            value_repr = self.add_quotation(value_repr, uuid_map)

            if key == 'UUID':
                continue
            if type(value_repr).__name__ == 'str' and value_repr == '':
                continue

            function_call += f" {key}={value_repr},"

        if function_call[-1] == ',':
            function_call = function_call[:-1] + ")"
        else:
            function_call = function_call + ")"

        if 'UUID' in parameters:
            function_call = parameters['UUID'] + ' = ' + function_call

        return function_call

    def to_python(self, level=-1, uuid_map=None):
        global last_uuid
        if uuid_map is None:
            uuid_map = {}

        indent = "    " * level
        code_lines = []

        if self.action == 'root':
            pass
        elif self.action == 'pass' or self.action == 'is.workflow.actions.nothing':
            code_lines.append(f"{indent}pass")

        elif self.action == 'is.workflow.actions.comment':
            comment_text = self.get_variable(self.params.get('WFCommentActionText', '')).replace('\n',
                                                                                                 '\n' + indent + '# ')
            code_lines.append(f"{indent}# {comment_text}")


        elif self.action == 'is.workflow.actions.conditional':
            if self.params['WFControlFlowMode'] == 0:
                condition = self.params.get('WFCondition', '')
                value = self.get_comparison_value(self.params)
                if not value or value == 'None':
                    value = 1
                input_var = self.get_input_var(self.params, uuid_map)

                operator_map = {
                    '0': '<',  # datetime earlier than or less than
                    '1': '<=',  #
                    '2': '>',  # Larger than
                    '3': '>=',  # Larger than or equals
                    '4': '==',  # Equals
                    '5': '!=',  # Does Not Equal
                    '8': 'start with',  # end with
                    '9': 'end with',  # end with
                    '99': 'contains',  # Contains
                    '100': 'exists',  # Exists (our new interpretation)
                    '101': 'not exists',  # Not Exists
                    '999': 'not contains',  # Not Contains
                    '1000': 'future',  # future time
                    '1001': 'recent',  # last time
                    '1002': 'is today',
                    '1003': 'between',  # Between
                    'Contains': 'contains',
                    'Is Greater Than': '>',
                    'Is Less Than': '<',
                    'Equals': '==',
                    '': 'contains'
                }

                operator = operator_map.get(str(condition))

                if operator == 'contains':
                    code_lines.append(f"{indent}if {value} in {input_var}:")
                elif operator == 'not contains':
                    code_lines.append(f"{indent}if {value} not in {input_var}:")
                elif operator == 'exists':
                    code_lines.append(f"{indent}if {input_var}:")
                elif operator == 'not exists':
                    code_lines.append(f"{indent}if not {input_var}:")
                elif operator in ['==', '!=', '>', '>=', '<', '<=']:
                    code_lines.append(f"{indent}if {input_var} {operator} {value}:")
                elif operator == 'between':
                    if 'WFNumberValue' in self.params and 'WFAnotherNumber' in self.params:
                        start_value = self.get_variable(self.params.get('WFNumberValue'))
                        end_value = self.get_variable(self.params.get('WFAnotherNumber'))
                    elif 'WFDate' in self.params and 'WFAnotherDate' in self.params:
                        start_value = self.get_variable(self.params.get('WFDate'))
                        end_value = self.get_variable(self.params.get('WFAnotherDate'))
                    elif 'WFMeasurement' in self.params and 'WFAnotherMeasurement' in self.params:  # todo: bug here
                        start_value = self.get_variable(self.params['WFMeasurement']['Value']['Magnitude']) + ' ' + \
                                      self.params['WFMeasurement']['Value']['Unit']
                        end_value = self.get_variable(self.params['WFAnotherMeasurement']['Value']['Magnitude']) + ' ' + \
                                    self.params['WFAnotherMeasurement']['Value']['Unit']
                    elif 'WFDuration' in self.params and 'WFAnotherDuration' in self.params:
                        start_value = self.get_variable(self.params['WFDuration']['Value']['Magnitude']) + ' ' + \
                                      self.params['WFDuration']['Value']['Unit']
                        end_value = self.get_variable(self.params['WFAnotherDuration']['Value']['Magnitude']) + ' ' + \
                                    self.params['WFAnotherDuration']['Value']['Unit']

                    else:
                        raise Exception
                    code_lines.append(f"{indent}if {start_value} <= {input_var} <= {end_value}:")
                elif operator == 'start with':
                    code_lines.append(f"{indent}if {input_var}.startswith({value}):")
                elif operator == 'end with':
                    code_lines.append(f"{indent}if {input_var}.endswith({value}):")
                elif operator == 'is today':
                    today = "datetime.now().date()"
                    code_lines.append(f"{indent}if {input_var}.date() == {today}:")
                elif operator in ['future', 'recent']:
                    duration = self.params['WFDuration']

                    magnitude = self.get_variable(duration['Value']['Magnitude'])

                    unit = duration['Value']['Unit']

                    if operator == 'future':
                        #
                        target_time = 'get_future_time(' + f"{magnitude}" + ',' + f"'{unit}'" + ')'
                        code_lines.append(f"{indent}if {input_var} <= {target_time}:")
                    elif operator == 'recent':
                        target_time = 'get_recent_time(' + f"{magnitude}" + ',' + f"'{unit}'" + ')'
                        code_lines.append(f"{indent}if {input_var} >= {target_time}:")
                else:
                    raise Exception('NOT IMPLEMENTED')
            else:
                code_lines.append(f"{indent}else:")

        elif self.action == 'is.workflow.actions.setvariable' or self.action == 'is.workflow.actions.appendvariable':
            var_name = self.params.get('WFVariableName', '')
            var_name = re.sub(r'[^a-zA-Z0-9]', '_', var_name)

            value = self.get_input_var(self.params, uuid_map)
            if value != False and  not value:
                value = 'None'

            if var_name:
                uuid_map[var_name] = value
                code_lines.append(f"{indent}{var_name} = {value}")
            else:
                code_lines.append(f"{value}")

        elif self.action == 'is.workflow.actions.repeat.each':
            if self.params['repeats_num'] == 1:
                index_str = ""
            else:
                index_str = f"_{self.params['repeats_num']}"

            input_var = self.get_input_var(self.params, uuid_map)

            code_lines.append(
                f"{indent}for Repeat_Index{index_str}, Repeat_Item{index_str} in enumerate({input_var}, start=1):")

        elif self.action == 'is.workflow.actions.repeat.count':
            if self.params['repeats_num'] == 1:
                index_str = ""
            else:
                index_str = f"_{self.params['repeats_num']}"
            count = self.get_variable(self.params.get('WFRepeatCount', 1))
            code_lines.append(f"{indent}for Repeat_Index{index_str} in range(int({count})):")

        elif self.action == 'is.workflow.actions.choosefrommenu':
            if self.params['WFControlFlowMode'] == 0:
                prompt = self.params.get('WFMenuPrompt', '')
                if prompt != '':
                    prompt = self.get_variable(self.params.get('WFMenuPrompt', ''))
                if "f'" in prompt or len(prompt) == 0 or prompt[0] in ('"', "'", '[', '{'):
                    prompt = prompt
                else:
                    prompt = "'''" + escape_single_quotes(prompt) + "'''"
                if len(prompt) != 0:
                    code_lines.append(f"{indent}match input(prompt={prompt}):")
                else:
                    code_lines.append(f"{indent}match input():")
            elif self.params['WFControlFlowMode'] == 1:
                Title = self.params.get('WFMenuItemTitle', '')

                code_lines.append(f'{indent}case "{Title}":')

        elif self.action == 'is.workflow.actions.gettext':

            text = self.params.get('WFTextActionText', '')
            if text != '':
                text = self.get_variable(text)
            else:
                text = '""'
            if "f'" in text or len(text) == 0 or text[0] in ('"', "'", '[', '{'):
                text = text
            else:
                text = "'''" + escape_single_quotes(text) + "'''"

            UUID = self.params.get('UUID', '')
            if UUID == '':
                code_lines.append(f"{indent}{escape_single_quotes(text)}")
            else:
                code_lines.append(f"{indent}{UUID} = {escape_single_quotes(text)}")

        elif 'list' in self.action:
            code_lines.append(f"{indent}{self.action_to_function_call(self.action, self.params, uuid_map)}")

        elif self.action == 'is.workflow.actions.dictionary':
            UUID = self.params.get('UUID')

            value_repr_new = '{'
            items = self.params.get('WFItems', {}).get('Value', {}).get('WFDictionaryFieldValueItems', [])
            for i, item in enumerate(items):
                key, val = item["WFKey"], item['WFValue']

                key, val = self.handle_dict_value(key, uuid_map), self.handle_dict_value(val, uuid_map)
                if not val:
                    val = 'None'

                value_repr_new += f'{key}: {val}'
                if i != len(items) - 1:
                    value_repr_new += ', '
            value_repr_new += '}'
            if UUID:
                code_lines.append(f"{indent}{UUID} = {value_repr_new}")
            else:
                code_lines.append(f"{indent}{value_repr_new}")


        elif self.action == 'is.workflow.actions.getvalueforkey':
            UUID = self.params.get('UUID')

            if UUID:
                variable = self.get_input_var(self.params, uuid_map)
                if self.params.get('WFGetDictionaryValueType', '') in {'All Keys',
                                                                       'All Values'} or 'WFDictionaryKey' not in self.params:
                    code_lines.append(f"{indent}{UUID} = {variable}")
                else:
                    key = self.get_variable(self.params['WFDictionaryKey'])
                    code_lines.append(f"{indent}{UUID} = {variable}[{key}]")
            else:
                variable = self.get_input_var(self.params, uuid_map)
                if self.params.get('WFGetDictionaryValueType', '') in {'All Keys',
                                                                       'All Values'} or 'WFDictionaryKey' not in self.params:
                    code_lines.append(f"{indent}{variable}")
                else:
                    key = self.get_variable(self.params['WFDictionaryKey'])
                    code_lines.append(f"{indent}{variable}[{key}]")
        elif self.action == 'is.workflow.actions.ask':
            prompt = self.params.get('WFAskActionPrompt', '')
            if prompt != '':
                prompt = self.get_variable(prompt)
            if "f'" in prompt or len(prompt) == 0 or prompt[0] in ('"', "'", '[', '{'):
                prompt = prompt
            else:
                prompt = "'''" + escape_single_quotes(prompt) + "'''"

            UUID = self.params.get('UUID', '')
            if UUID == '':
                code_lines.append(f"{indent}input({escape_single_quotes(prompt)})")
            else:
                code_lines.append(f"{indent}{UUID} = input({escape_single_quotes(prompt)})")

        elif 'WFDict' in str(self.action) or 'WFDict' in str(self.params):
            code_lines.append(f"{indent}{self.action_to_function_call(self.action, self.params, uuid_map)}")


        else:
            code_lines.append(f"{indent}{self.action_to_function_call(self.action, self.params, uuid_map)}")

        # ---- set last uuid --------
        if 'UUID' in self.params:
            last_uuid = self.params['UUID']
        else:
            last_uuid = None

        # -----------------
        for child in self.children:
            child_code = child.to_python(level + 1, uuid_map)
            code_lines.extend(child_code)

        return code_lines

    def handle_magic_value(self, value):
        base_string = value['string']
        attachments = value['attachmentsByRange']
        sorted_attachments = sorted(attachments.items(), key=lambda x: int(x[0].strip('{}').split(',')[0]))
        offset = 0

        for range_key, attachment in sorted_attachments:
            start, length = map(int, range_key.strip('{}').split(','))
            if attachment['Type'] == 'Variable':
                ans = attachment['VariableName']
                ans = re.sub(r'[^a-zA-Z0-9]', '_', ans)
                aggrandizements = attachment.get('Aggrandizements', [])
                variable_value = self.handle_aggrandizements(ans, aggrandizements)
                variable_value = "{" + variable_value + "}"
            elif attachment['Type'] == 'ActionOutput':
                ans = attachment.get('OutputUUID')
                aggrandizements = attachment.get('Aggrandizements', [])
                variable_value = self.handle_aggrandizements(ans, aggrandizements)
                variable_value = "{" + variable_value + "}"

            elif attachment['Type'] == 'CurrentDate':
                variable_value = 'datetime.datetime.now()'
            elif attachment['Type'] == 'ExtensionInput':
                variable_value = 'input("Please enter the value: ")'
            elif attachment['Type'] == 'Clipboard':
                variable_value = 'is_workflow_actions_getclipboard()'
            elif attachment['Type'] == 'DeviceDetails':
                properties = [_["PropertyName"] for _ in value.get("Aggrandizements", [])]
                if len(properties):
                    variable_value = f'is_workflow_actions_getdevicedetails({[_["PropertyName"] for _ in value["Aggrandizements"]]})'
                else:
                    variable_value = f'is_workflow_actions_getdevicedetails()'

            elif attachment['Type'] == 'Ask' or attachment['Type'] == 'Input':
                variable_value = "f'{" + 'input("Please enter the value:")' + "}'"
            else:
                raise Exception('Not Implemented!')
            base_string = base_string[:start + offset] + variable_value + base_string[start + offset + length:]
            offset += len(variable_value) - length
        if len(sorted_attachments) > 0:
            return "f'''" + escape_single_quotes(base_string) + "'''"
        else:
            return escape_single_quotes(base_string)

    def get_comparison_value(self, params):
        if 'WFConditionalActionString' in params:
            value = params['WFConditionalActionString']
            if isinstance(value, dict) and 'Value' in value and 'string' in value['Value'] and 'attachmentsByRange' in \
                    value['Value']:
                return self.handle_magic_value(value['Value'])

            if (isinstance(value, (int, float, bool)) or value is None or value == "" or
                    (isinstance(value, str) and ("f'" in value or value[0] in ('"', "'", "[", "{"))) or
                    (isinstance(value, dict) and value.get('Value', {}).get('Type', {}) == 'Variable')):
                value = value
            else:
                value = self.process_value(value)

            return value
        elif 'WFNumberValue' in params:
            try:
                return f"{self.get_variable(params['WFNumberValue'])}"
            except:
                return self.get_variable(params['WFNumberValue'])
        elif 'WFDate' in params:
            return self.get_variable(params['WFDate'])
        elif 'WFMeasurement' in params:
            measurement = params['WFMeasurement']['Value']
            magnitude = self.get_variable(measurement.get('Magnitude', 0))
            unit = measurement['Unit']
            return f"{magnitude}{unit}"
        elif 'WFEnumeration' in params:
            return self.get_variable(params['WFEnumeration'])
        return ''

    def handle_aggrandizements(self, init_value, aggrandizements):
        value = init_value
        for aggrandizement in aggrandizements:
            aggrandizement_type = aggrandizement['Type']
            if aggrandizement_type == 'WFPropertyVariableAggrandizement':
                property_name = aggrandizement['PropertyName']
                value = f"{value}.{property_name}"
            elif aggrandizement_type == 'WFUnitVariableAggrandizement':
                unit_type = aggrandizement['WFMeasurementUnitType']
                unit_symbol = aggrandizement['WFUnitSymbol']
                value = f'unit_variable_transform(value={value}, unit_type="{unit_type}", unit_symbol="{unit_symbol}")'
            elif aggrandizement_type == 'WFCoercionVariableAggrandizement':
                coercion_class = aggrandizement['CoercionItemClass']
                if coercion_class == 'WFStringContentItem':
                    value = f'str({value})'
                elif coercion_class == 'WFDictionaryValueVariableAggrandizement':
                    value = f'dict({value})'
                elif coercion_class == 'WFNumberContentItem':
                    value = f'float({value})'

                else:
                    value = f'coerce_variable(value={value}, coercion_class="{coercion_class}")'
            elif aggrandizement_type == 'WFDateFormatVariableAggrandizement':
                date_format = aggrandizement.get('WFDateFormatStyle', None)
                time_format = aggrandizement.get('WFTimeFormatStyle', None)
                iso_include_time = aggrandizement.get('WFISO8601IncludeTime', None)

                value = "format_date(value={value}"

                if date_format:
                    value += f", date_format='{date_format}'"
                if time_format:
                    value += f", time_format='{time_format}'"
                if iso_include_time is not None:
                    value += f", iso_include_time={iso_include_time}"

                value += ")"

            elif aggrandizement_type == 'WFDictionaryValueVariableAggrandizement':
                dictionary_key = aggrandizement['DictionaryKey']
                value = f'{value}["{dictionary_key}"]'
            else:
                raise ValueError(f"Unsupported aggrandizement type: {aggrandizement_type}")
        # value = "{" + value + "}"
        return value

    def get_variable(self, value):
        if type(value).__name__ == 'str':


            if (isinstance(value, (int, float, bool)) or value is None or value == "" or
                    (isinstance(value, str) and ("f'" in value or value[0] in ('"', "'", "[", "{"))) or
                    (isinstance(value, dict) and value.get('Value', {}).get('Type', {}) == 'Variable')):
                value = value
            else:
                value = self.process_value(value)

            return value
        elif type(value).__name__ == 'bytes':
            try:
                return escape_single_quotes(plistlib.loads(value))
            except:
                return escape_single_quotes(value.decode(errors='ignore'))
        elif type(value).__name__ == 'int' or type(value).__name__ == 'float' or type(
                value).__name__ == 'double' or type(value).__name__ == 'datetime' or type(value).__name__ == 'bool':
            return value
        # elif type(value).__name__ == 'dict' and 'WFSerializationType' in value:
        elif isinstance(value, dict) and value.get('WFSerializationType') == 'WFDictionaryFieldValue':
            items = value.get('Value', {}).get('Value', {}).get('WFDictionaryFieldValueItems', [])
            nested_repr = '{'
            for i, item in enumerate(items):
                key, val = item["WFKey"], item['WFValue']
                key, val = self.get_variable(key), self.get_variable(val)
                nested_repr += f'{key}: {val}'
                if i != len(items) - 1:
                    nested_repr += ', '
            nested_repr += '}'
            return nested_repr
        elif isinstance(value, dict) and value.get('WFSerializationType') == 'WFArrayParameterState':
            items = value['Value']
            nested_repr = '['
            for i, item in enumerate(items):
                if isinstance(item, dict) and 'WFItemType' in item and 'WFValue' in item:
                    try:
                        item_repr = self.get_variable(item['WFValue'])
                    except:
                        print('Error handling item')
                        item_repr = 'error'
                else:
                    item_repr = str(item)
                nested_repr += f"{item_repr}"
                if i != len(items) - 1:
                    nested_repr += ', '
            nested_repr += ']'
            return nested_repr

        elif type(value).__name__ == 'dict' and 'WFValue' in value:
            value = value['WFValue']
            return self.get_variable(value)
        elif type(value).__name__ == 'dict' and 'Value' in value:
            value = value['Value']
            return self.get_variable(value)
        elif type(value).__name__ == 'list':
            return [self.get_variable(v) for v in value]
        elif value is None:
            return 'None'

        if 'attachmentsByRange' in value:
            return self.handle_magic_value(value)

        if 'Type' in value:
            if value['Type'] == 'ExtensionInput' or value['Type'] == 'Ask' or value['Type'] == 'Input':
                return "f'{" + 'input("Please enter the value:")' + "}'"

            elif value['Type'] == 'ActionOutput':
                ans = value.get('OutputUUID')
                aggrandizements = value.get('Aggrandizements', [])
                return self.handle_aggrandizements(ans, aggrandizements)

            elif value['Type'] == 'Variable':
                ans = value.get('VariableName', '')
                ans = re.sub(r'[^a-zA-Z0-9]', '_', ans)

                if len(ans)> 0:
                    aggrandizements = value.get('Aggrandizements', [])
                    return self.handle_aggrandizements(ans, aggrandizements)
                elif 'Variable' in value:
                    return self.get_variable(value['Variable'])

            elif value['Type'] == 'Clipboard':
                return 'is_workflow_actions_getclipboard()'
            elif value['Type'] == 'CurrentDate':
                return 'datetime.datetime.now()'
            elif value['Type'] == 'HomeAccessory':
                home_service = value.get('HomeService', {})
                return f'is_workflow_actions_gethomeaccessorystate({home_service})'
            elif value['Type'] == 'DeviceDetails':
                properties = [_.get("PropertyName", None) for _ in
                              value.get("Aggrandizements", [])]
                properties = [_ for _ in properties if _ is not None]
                if len(properties):
                    return f'is_workflow_actions_getdevicedetails({properties})'
                else:
                    return f'is_workflow_actions_getdevicedetails()'
            else:
                print(value)
                raise Exception('Not Implemented!!')
        elif type(value).__name__ == 'dict':
            return self.process_value({k: self.get_variable(v) for k, v in value.items()})

    def get_input_var(self, params, uuid_map):
        if 'WFInput' not in params:
            return last_uuid
        wf_input = params['WFInput']
        if 'Variable' in wf_input:
            wf_input = wf_input['Variable']
        if 'WFSerializationType' in wf_input:
            value = wf_input['Value']
        else:
            value = wf_input
        value = self.get_variable(value)
        if value:
            return value
        if value == None:
            return 'None'
        return 1

    #
    def handle_dict_value(self, dict_value, uuid_map):
        return self.add_quotation(self.get_variable(dict_value), uuid_map)


def class_to_string(obj, level=0):
    result = ""
    indent = "  " * level
    if isinstance(obj, dict):
        for k, v in obj.items():
            result += f"{indent}{k}: {class_to_string(v, level + 1)}\n"
    elif isinstance(obj, list):
        for item in obj:
            result += f"{indent}- {class_to_string(item, level + 1)}\n"
    elif hasattr(obj, '__dict__'):
        for k, v in obj.__dict__.items():
            result += f"{indent}{k}: {class_to_string(v, level + 1)}\n"
    else:
        result += f"{indent}{obj}"
    return result

class ShortcutParser:
    def __init__(self, shortcut_json):
        self.shortcut_json = shortcut_json
        self.node_map = {}
        self.node2parent = {}
        self.root = Node("root", {}, index_id=-1)
        self.node_map[-1] = self.root
        self.current_node = self.root
        self.action_handlers = {
            'is.workflow.actions.conditional': self.handle_conditional,
            'is.workflow.actions.repeat.each': self.handle_repeat_each,
            'is.workflow.actions.repeat.count': self.handle_repeat_count,
            'is.workflow.actions.choosefrommenu': self.handle_choosefrommenu
        }

        self.uuid_generator = FixedUUIDGenerator()


    def parse(self):
        for index_id, action in enumerate(self.shortcut_json):
            action_id = action['WFWorkflowActionIdentifier']
            params = action.get('WFWorkflowActionParameters', {})
            handler = self.action_handlers.get(action_id, self.handle_default)
            handler(action_id, params, index_id)


    def cal_max_length(self, node):
        if len(node.children) == 0:
            return 1
        return max([self.cal_max_length(child_node) for child_node in node.children]) + 1

    def add_empty_child(self, node):
        if node.action in (
        'is.workflow.actions.conditional', 'is.workflow.actions.choosefrommenu', 'is.workflow.actions.repeat.each',
        'is.workflow.actions.repeat.count') and node.children == []:
            node.children.append(Node("pass", {}, index_id=-1))
            return
        for child in node.children:
            self.add_empty_child(child)

    def add_necessary_UUID(self, node):
        if len(node.children):
            now = node.children[0]

            now_key_paras = identifier2keyparams[now.action.replace('.', '_')]
            if len(now_key_paras) > 0 and now_key_paras not in now.params:
                parent = self.node_map[self.node2parent[now.index_id]]
                if node.action == 'is.workflow.actions.conditional' and 'WFInput' in parent.params:
                    now.params[now_key_paras] = copy.deepcopy(parent.params['WFInput'])

        most_frequent_uuid = None
        if len(node.children) > 0 and 'UUID' in node.children[0].params:
            most_frequent_uuid = node.children[0].params['UUID']
        for index in range(1, len(node.children), 1):
            now = node.children[index]
            last = node.children[index - 1]

            now_key_paras = identifier2keyparams[now.action.replace('.', '_')]

            if len(now_key_paras) > 0 and now_key_paras not in now.params:
                if 'UUID' in last.params:
                    now.params[now_key_paras] = {'Value': {'OutputName': '_', 'OutputUUID': last.params['UUID'],
                                                           'Type': 'ActionOutput'},
                                                 'WFSerializationType': 'WFTextTokenAttachment'}
                elif last.action.replace('.', '_') in identifier2output or (last.action == 'is.workflow.actions.conditional' and last.params['WFControlFlowMode'] == 2):
                    last.params['UUID'] = self.uuid_generator.generate_uuid()
                    now.params[now_key_paras] = {'Value': {'OutputName': '_', 'OutputUUID': last.params['UUID'],
                                                           'Type': 'ActionOutput'},
                                                 'WFSerializationType': 'WFTextTokenAttachment'}

                    most_frequent_uuid = last.params['UUID']

                else:
                    now.params[now_key_paras] = {'Value': {'OutputName': '_', 'OutputUUID': most_frequent_uuid,
                                                           'Type': 'ActionOutput'},
                                                 'WFSerializationType': 'WFTextTokenAttachment'}

            if 'UUID' in now.params:
                most_frequent_uuid = now.params['UUID']

        for child in node.children:
            self.add_necessary_UUID(child)
    def handle_contrl_UUID(self, node):
        for index, child in enumerate(node.children):
            if 'is.workflow.actions.conditional' in child.action and child.params['WFControlFlowMode'] == 2:
                ENDIF_UUID = child.params.get('UUID')
                if ENDIF_UUID is not None:
                    if index > 0 and len(node.children[index - 1].children):
                        node_to_change = node.children[index - 1].children[-1]
                        if node_to_change.action.replace('.', '_') in identifier2output or ('is.workflow.actions.conditional' in node_to_change.action and node_to_change.params['WFControlFlowMode'] == 2):
                            node_to_change.params['UUID'] = ENDIF_UUID

                    if index > 1 and len(node.children[index - 2].children):
                        node_to_change = node.children[index - 2].children[-1]
                        if node_to_change.action.replace('.', '_') in identifier2output or ('is.workflow.actions.conditional' in node_to_change.action and node_to_change.params['WFControlFlowMode'] == 2):
                            node_to_change.params['UUID'] = ENDIF_UUID

            elif 'is.workflow.actions.choosefrommenu' in child.action and child.params['WFControlFlowMode'] == 0:
                if child.params.get('UUID'):
                    ENDMENU_UUID = child.params.get('UUID', None)
                else:
                    if len(child.children) > 0:
                        ENDMENU_UUID = child.children[-1].params.get('UUID', None)
                    else:
                        ENDMENU_UUID = None
                if ENDMENU_UUID is not None:
                    for subindex, subchild in enumerate(child.children[:-1]):
                        if len(subchild.children) > 0:
                            node_to_change = subchild.children[-1]
                            if node_to_change.action.replace('.', '_') in identifier2output or ('is.workflow.actions.conditional' in node_to_change.action and node_to_change.params['WFControlFlowMode'] == 2):
                                node_to_change.params['UUID'] = ENDMENU_UUID
                child.children = child.children[:-1]
            elif child.action in ('is.workflow.actions.repeat.count', 'is.workflow.actions.repeat.each') and child.params['WFControlFlowMode'] == 2:
                ENDFOR_UUID = child.params.get('UUID')
                if ENDFOR_UUID is not None:
                    if index > 0 and len(node.children[index - 1].children):
                        node_to_change = node.children[index - 1].children[-1]
                        if node_to_change.action.replace('.', '_') in identifier2output or ('is.workflow.actions.conditional' in node_to_change.action and node_to_change.params['WFControlFlowMode'] == 2):
                            node_to_change.params['UUID'] = ENDFOR_UUID


        node.children = [child for child in node.children if not
        (child.action in ('is.workflow.actions.conditional','is.workflow.actions.repeat.count', 'is.workflow.actions.repeat.each' ) and child.params['WFControlFlowMode'] == 2)]

        for child in node.children:
            self.handle_contrl_UUID(child)

    def mark_multi_repeat(self, node):
        if 'is.workflow.actions.repeat' in node.action:
            repeats_num = 0
            temp_node = node
            while (temp_node.index_id != -1):
                if 'is.workflow.actions.repeat' in temp_node.action:
                    repeats_num += 1
                temp_node = self.node_map[self.node2parent[temp_node.index_id]]

            node.params['repeats_num'] = repeats_num
        for child in node.children:
            self.mark_multi_repeat(child)

    def handle_conditional(self, action_id, params, index_id):
        mode = params.get('WFControlFlowMode')
        if mode == 0:  # if statement
            self.add_node(action_id, params, index_id)
            self.current_node = self.node_map[index_id]
        elif mode == 1:  # else statement
            self.current_node = self.node_map[
                self.node2parent[self.current_node.index_id]]  # Go back to the parent of the if node
            self.add_node(action_id, params, index_id)
            self.current_node = self.node_map[index_id]
        elif mode == 2:  # end if/else statement
            self.current_node = self.node_map[self.node2parent[self.current_node.index_id]]

            self.add_node(action_id, params, index_id)
    def handle_choosefrommenu(self, action_id, params, index_id):
        mode = params.get('WFControlFlowMode')
        if mode == 0:
            self.add_node(action_id, params, index_id)
            self.current_node = self.node_map[index_id]
        elif mode == 1:  #
            if self.current_node.action == 'is.workflow.actions.choosefrommenu' and self.current_node.params[
                'WFControlFlowMode'] == 0:
                self.add_node(action_id, params, index_id)
                self.current_node = self.node_map[index_id]
            elif self.current_node.action == 'is.workflow.actions.choosefrommenu' and self.current_node.params[
                'WFControlFlowMode'] == 1:
                self.current_node = self.node_map[
                    self.node2parent[self.current_node.index_id]]
                self.add_node(action_id, params, index_id)
                self.current_node = self.node_map[index_id]
            else:
                raise Exception('MATCH ERROR')
        elif mode == 2:
            self.current_node = self.node_map[self.node2parent[self.current_node.index_id]]
            self.add_node(action_id, params, index_id)
            self.current_node = self.node_map[self.node2parent[self.current_node.index_id ]]

    def handle_repeat_each(self, action_id, params, index_id):
        mode = params.get('WFControlFlowMode')
        if mode == 0:
            self.add_node(action_id, params, index_id)
            self.current_node = self.node_map[index_id]
        elif mode == 2:
            self.current_node = self.node_map[self.node2parent[self.current_node.index_id]]

            self.add_node(action_id, params, index_id)


    def handle_repeat_count(self, action_id, params, index_id):
        mode = params.get('WFControlFlowMode')
        if mode == 0:
            self.add_node(action_id, params, index_id)
            self.current_node = self.node_map[index_id]
        elif mode == 2:
            self.current_node = self.node_map[self.node2parent[self.current_node.index_id]]
            self.add_node(action_id, params, index_id)

    def handle_default(self, action_id, params, index_id):
        self.add_node(action_id, params, index_id)

    def add_node(self, action_id, params, index_id):
        node = Node(action_id, copy.deepcopy(params), index_id)
        self.current_node.children.append(node)
        self.node2parent[index_id] = self.current_node.index_id
        self.node_map[index_id] = node

    def to_python(self):
        return "\n".join(self.root.to_python())


def download_shortcut(url: str):
    '''Downloads shortcut file if possible and returns a string with plist'''
    shortcut_id = _get_shortcut_uuid(url)
    shortcut_info = _get_shortcut_info(shortcut_id)
    download_url = shortcut_info['fields']['shortcut']['value']['downloadURL']
    response = _make_request(download_url)

    return response.read()


def _get_shortcut_uuid(url: str) -> str:
    '''
    Returns uuid from shortcut's public URL.
    Public url example: https://www.icloud.com/shortcuts/{uuid}/

    Raises:
        property_list_style.exceptions.InvalidShortcutURLError if the "url" parameter is not valid
    '''

    if not is_shortcut_url(url):
        raise exceptions.InvalidShortcutURLError('Not a shortcut URL!')

    parsed_url = urlparse(url)
    splitted_path = os.path.split(parsed_url.path)

    if splitted_path:
        shortcut_id = splitted_path[-1]
        try:
            uuid.UUID(
                shortcut_id
            )  # just for validation, raises an error if it's not a valid UUID
        except ValueError:
            raise exceptions.InvalidShortcutURLError(
                f'Can not find shortcut id in "{url}"'
            )

        return shortcut_id


def is_shortcut_url(url: str) -> bool:
    '''Determines is it a shortcut URL or not'''
    parsed_url = urlparse(url)

    if parsed_url.netloc not in ('www.icloud.com', 'icloud.com'):
        return False

    if not parsed_url.path.startswith('/property_list_style/'):
        return False

    return True


def _get_shortcut_info(shortcut_id: str) -> Dict:
    '''
    Downloads shortcut information from a public (and not official) API

    Returns:
        dictioanry with shortcut information
    '''
    url = f'https://www.icloud.com/shortcuts/api/records/{shortcut_id}/'
    response = _make_request(url)

    return json.loads(response.read())


def _make_request(url: str):
    '''
    Makes HTTP request

    Raises:
        RuntimeError if response.status != 200
    '''
    response = urlopen(url)

    if response.status != 200:  # type: ignore
        raise RuntimeError(
            f'Can not get shortcut information from API: response code {response.status}'  # type: ignore
        )

    return response


def escape_single_quotes(input_string, max_len=384):
    if type(input_string).__name__ == 'list':
        return str([escape_single_quotes(_) for _ in input_string])

    # First, replace instances of three single quotes with a placeholder
    placeholder = "<triple_quotes_placeholder>"
    input_string = input_string.replace("'''", placeholder)

    # Then, replace single quotes with escaped single quotes
    input_string = input_string.replace("'", "\\'")

    # Finally, restore the triple single quotes
    input_string = input_string.replace(placeholder, "'''")

    if isinstance(input_string, str) and len(input_string) > max_len:
        input_string = input_string[:max_len // 2] + input_string[-max_len // 2:]

    return input_string


param_type2python = defaultdict(
    lambda: 'object',
    {
        'WFEnumerationParameter': 'str',
        'WFSwitchParameter': 'bool',
        'WFContentArrayParameter': 'list',
        'WFStepperParameter': 'int',
        'WFChooseFromMenuArrayParameter': 'list',
        'WFIntentAppPickerParameter': 'dict',
        'WFTextInputParameter': 'str',
        'WFNumberFieldParameter': 'float',
        'WFOSAScriptEditorParameter': 'str',
        'WFMediaRoutePickerParameter': 'str',
        'WFInputSurfaceParameter': 'str',
        'WFImageConvertFormatPickerParameter': 'str',
        'WFSliderParameter': 'float',
        'WFFaceTimeTypePickerParameter': 'str',
        'WFGetWiFiDetailEnumerationParameter': 'str',
        'WFDynamicEnumerationParameter': 'str',
        'WFFlipImageDirectionPickerParameter': 'str',
        'WFUnitQuantityFieldParameter': 'float',
        'WFUnitTypePickerParameter': 'str',
        'WFTodoistProjectPickerParameter': 'str',
        'WFMapsAppPickerParameter': 'str',
        'WFSpeakTextLanguagePickerParameter': 'str',
        'WFSpeakTextVoicePickerParameter': 'str',
        'WFWorkoutGoalQuantityFieldParameter': 'float',
        'WFFileSizePickerParameter': 'float',
        'WFHealthCategoryAdditionalPickerParameter': 'str',
        'WFVPNPickerParameter': 'str',
        'WFFocusModesPickerParameter': 'str',
        'WFHomeServicePickerParameter': 'str',
        'WFColorPickerParameter': 'str',
        'WFInputTypeParameter': 'str',
        'WFDateFieldParameter': 'datetime.date',
        'WFTagFieldParameter': 'str',
        'WFVariablePickerParameter': 'str',
        'WFEvernoteTagsTagFieldParameter': 'str',
        'WFWorkflowPickerParameter': 'str',
        'WFSSHKeyParameter': 'str',
        'WFMakeImageFromPDFPageColorspaceParameter': 'str',
        'WFVariableFieldParameter': 'str',
        'WFURLParameter': 'str',
        'WFMediaPickerParameter': 'str',
        'WFContactFieldParameter': 'str',
        'WFAirDropVisibilityParameter': 'str',
        'WFPodcastPickerParameter': 'str',
        'WFConditionalOperatorParameter': 'str',
        'WFTranslateTextLanguagePickerParameter': 'str',
        'WFHealthQuantityFieldParameter': 'float',
        'WFMakeImageFromPDFPageImageFormatParameter': 'str',
        'WFRemindersListPickerParameter': 'str',
        'WFTrelloListPickerParameter': 'str',
        'WFMailSenderPickerParameter': 'str',
        'WFWorkflowFolderPickerParameter': 'str',
        'WFSearchLocalBusinessesRadiusParameter': 'float',
        'WFCountryFieldParameter': 'str',
        'WFHomeAreaPickerParameter': 'str',
        'WFTumblrBlogPickerParameter': 'str',
        'WFListeningModePickerParameter': 'str',
        'WFDynamicTagFieldParameter': 'str',
        'WFAccountPickerParameter': 'str',
        'WFConditionalSubjectParameter': 'str',
        'WFPosterPickerParameter': 'str',
        'WFLightroomPresetPickerParameter': 'str',
        'WFTimeZonePickerParameter': 'str',
        'WFWorkoutTypePickerParameter': 'str',
        'WFHealthActionStartDateFieldParameter': 'datetime.date',
        'WFHealthQuantityAdditionalPickerParameter': 'str',
        'WFDisplayPickerParameter': 'str',
        'WFLocationAccuracyParameter': 'float',
        'WFDictateTextLanguagePickerParameter': 'str',
        'WFHealthQuantityAdditionalFieldParameter': 'float',
        'WFLocationParameter': 'object',
        'WFHealthCategoryPickerParameter': 'str',
        'WFPaymentMethodParameter': 'str',
        'WFFitnessWorkoutTypePickerParameter': 'str',
        'WFNetworkPickerParameter': 'str',
        'WFFilePickerParameter': 'str',
        'WFDictionaryParameter': 'dict',
        'WFCalendarPickerParameter': 'str',
        'WFNumericDynamicEnumerationParameter': 'float',
        'WFUIRecordingEventParameter': 'str',
        'WFCurrencyQuantityFieldParameter': 'float',
        'WFEvernoteNotebookPickerParameter': 'str',
        'WFiTunesStoreCountryPickerParameter': 'str',
        'WFDatePickerParameter': 'datetime.date',
        'WFFileLabelColorPickerParameter': 'str',
        'WFLinkDynamicEnumerationParameter': 'str',
        'WFDurationQuantityFieldParameter': 'float',
        'WFContactHandleFieldParameter': 'str',
        'WFPlaylistPickerParameter': 'str',
        'WFRideOptionParameter': 'str',
        'WFHomeAccessoryPickerParameter': 'str',
        'WFQuantityTypePickerParameter': 'str',
        'WFCustomDateFormatParameter': 'str',
        'WFMeasurementUnitPickerParameter': 'str',
        'WFFontPickerParameter': 'str',
        'WFExpandingParameter': 'object',
        'WFAppPickerParameter': 'str',
        'WFArchiveFormatParameter': 'str',
        'WFTumblrComposeInAppParameter': 'str',
        'WFHomeCharacteristicPickerParameter': 'str',
        'WFTrelloBoardPickerParameter': 'str',
        'WFPhoneNumberFieldParameter': 'str',
        'WFHealthActionEndDateFieldParameter': 'datetime.date',
        'WFEmailAddressFieldParameter': 'str',
        'WFGetDistanceUnitPickerParameter': 'str',
        'WFPhotoAlbumPickerParameter': 'str',
        'WFSlackChannelPickerParameter': 'str',
        'WFLocalePickerParameter': 'str',
        'public.data': 'object',
        'WFTrelloBoard': 'object',
        'WFPhoneNumber': 'str',
        'MKMapItem': 'object',
        'public.html': 'str',
        'WFContact': 'object',
        'WFTripInfo': 'object',
        'WFShazamMedia': 'object',
        'WFImageContentItem': 'object',
        'WFAppStoreAppContentItem': 'object',
        'AVAsset': 'object',
        'NSAttributedString': 'str',
        'WFSafariWebPage': 'str',
        'WFPodcastShowContentItem': 'object',
        'NSMeasurement': 'object',
        'WFParkedCarContentItem': 'object',
        'WFDictionaryContentItem': 'object',
        'INRideStatus': 'object',
        'WFDateContentItem': 'datetime.date',
        'float': 'float',
        'NSDecimalNumber': 'float',
        'WFArticleContentItem': 'object',
        'WFBooleanContentItem': 'bool',
        'WFGiphyObject': 'object',
        'WFArticle': 'object',
        'NSNumber': 'float',
        'bool': 'bool',
        'WFHKWorkoutContentItem': 'object',
        'WFPhotoMediaContentItem': 'object',
        'int': 'int',
        'NSDictionary': 'dict',
        'REMReminder': 'object',
        'CLLocation': 'object',
        'WFPDFContentItem': 'object',
        'WFWeatherData': 'object',
        'WFiTunesProductContentItem': 'object',
        'WFContactContentItem': 'object',
        'com.apple.quicktime-movie': 'object',
        'ENNoteRef': 'object',
        'NSString': 'str',
        'EKEvent': 'object',
        'WFAppContentItem': 'object',
        'WFPhoneNumberContentItem': 'str',
        'com.apple.m4a-audio': 'object',
        'WFStringContentItem': 'str',
        'WFLocationContentItem': 'object',
        'MPMediaItem': 'object',
        'WFTrelloList': 'object',
        'WFNumberContentItem': 'float',
        'NSDateComponents': 'datetime.date',
        'WFEmailAddressContentItem': 'str',
        'com.apple.coreaudio-format': 'object',
        'WFContentItem': 'object',
        'WFTimeInterval': 'float',
        'WFTrelloCard': 'object',
        'dict': 'dict',
        'WFHKSampleContentItem': 'object',
        'WFURLContentItem': 'str',
        'WFWorkflowReference': 'object',
        'WFPodcastEpisodeContentItem': 'object',
        'public.folder': 'object',
        'WFMachineReadableCode': 'str',
        'WFEmailAddress': 'str',
        'WFGenericFileContentItem': 'object',
        'WFStreetAddress': 'object',
        'public.mpeg-4': 'object',
        'com.compuserve.gif': 'object',
        'com.adobe.pdf': 'object',
        'list': 'list',
        'WFFocusModeContentItem': 'object',
        'public.plain-text': 'str',
        'PHAsset': 'object',
        'WFPosterRepresentation': 'object',
        'str': 'str',
        'public.image': 'object',
        'WFImage': 'object',
        'NSURL': 'str',
        'NSDate': 'datetime.date'
    }
)
