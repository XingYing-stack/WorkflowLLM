from utils import *
from collections import defaultdict
import pickle

def generate_function_signature_from_WFactions(name, config, identifier2python, identifier2keyparams, identifier2output):
    name = name.replace(".", "_")
    description = config.get('Description', {}).get('DescriptionSummary', 'No description available.')
    if config.get('ActionKeywords', []):
        description += '\nKey Words: ' + ', '.join(config.get('ActionKeywords', []))

    parameters = config.get('Parameters', [])
    output_config = config.get('Output', None)
    output_type = 'object'

    if output_config:
        output_types = output_config.get('Types', [])
        if output_types:
            output_type = param_type2python.get(output_types[0], 'object')
        output_name = output_config.get('OutputName', 'Output')
        identifier2output[name].append(output_type)
    else:
        output_type = 'None'
        output_name = 'Output'


    params_with_default = []
    params_without_default = []

    for param in parameters:
        param_class = param.get('Class', 'object')
        param_type = param_type2python.get(param_class, 'object')
        param_key = param.get("Key")
        param_label = param.get("Label", "")
        param_prompt = param_label + '\n' + param.get("Prompt", "")
        param_default = param.get("DefaultValue", param.get('Placeholder', None))
        param_items = param.get("Items", [])

        param_detail = f'- {param_key.replace(" ", "_")} ({param_type}): {param_prompt}'

        if param_default is not None:
            param_default = f'"{param_default}"' if param_type == 'str' else param_default
            params_with_default.append((f'{param_key.replace(" ", "_")}: {param_type} = {param_default}', param_detail + f' Default is {param_default}.'))
        else:
            params_without_default.append((f'{param_key.replace(" ", "_")}: {param_type}', param_detail))

        if param_items:
            param_detail += f' Possible values are {param_items}.'

    param_list = [param[0] for param in params_without_default + params_with_default]
    param_docs = [param[1] for param in params_without_default + params_with_default]

    if len(param_docs) == 0:
        param_docs = ['- None']

    function_signature = f'def {name}(\n    ' + ',\n    '.join(param_list) + f'\n) -> {output_type}:'

    docstring = f'"""\n\n{description}\n\nParameters:\n' + '\n'.join(param_docs) + '\n\nReturns:\n'
    if output_type == 'None':
        docstring += '- None\n'
    else:
        docstring += f'- {output_type}: {output_name}.\n'

    docstring += '\n"""'

    function_body = """
        pass
        """

    python_code = f"""{function_signature}\n    {docstring}\n{function_body}"""

    identifier2python[name] = python_code

    key_param = None
    input = config.get('Input', {})
    if input.get('Required', True) and (input.get('ParameterKey', None) is not None):
        identifier2keyparams[name] = input.get('ParameterKey', None)
    return python_code



def generate_function_signature_from_DOTactions(config, name, name_template, in_enums, in_entities, identifier2python, identifier2keyparams, identifier2output):
    action_name = name_template.format(name).replace('.', '_')
    action_title = config.get('title', {}).get('key', 'No title available')
    action_description = config.get('descriptionMetadata', {}).get('descriptionText', {}).get('key', '')
    parameters = config.get('parameters', [])

    param_docs = {}
    param_type2python = {'0': 'str', '1': 'bool', '2': 'int', '7': 'float', '8': 'datetime', '9': 'datetime', '12': 'object'}

    enum_type2values = {enum['identifier']: {val['identifier']: idx for idx, val in enumerate(enum['cases'])} for enum in in_enums}
    used_enums = {}
    entity_type2props = {entity_name: entity['properties'] for entity_name, entity in in_entities.items()}
    used_entities = {}

    required_params = []
    optional_params = []

    key_parameters = []
    for param in parameters:
        param_name = param.get('name', 'param')
        param_display_name = param.get('title', {}).get('key', param_name)
        value_type = param.get('valueType', {})
        primitive_type = value_type.get('primitive', {}).get('wrapper', {}).get('typeIdentifier')
        enum_type = value_type.get('linkEnumeration', {}).get('wrapper', {}).get('identifier')
        entity_type = value_type.get('entity', {}).get('wrapper', {}).get('typeName')
        type_specific_metadata = param.get('typeSpecificMetadata', [])

        is_optional = param.get('isOptional', True)
        if is_optional == False:
            key_parameters.append(param_name)

        default_value = None
        _default_list = []
        for i in range(0, len(type_specific_metadata), 2):
            if type_specific_metadata[i] == 'LNValueTypeSpecificMetadataKeyDefaultValue':
                _default_list = [type_specific_metadata[i+1]]

        for item in _default_list:
            if isinstance(item, list) and 'LNValueTypeSpecificMetadataKeyDefaultValue' in item:
                default_value = item[1]
                break
            elif isinstance(item, dict) and 'int' in item:
                default_value = item['int'].get('wrapper')
                break
            elif isinstance(item, dict) and 'string' in item:
                default_value = item['string'].get('wrapper')
                break
            elif isinstance(item, dict) and 'bool' in item:
                default_value = item['bool'].get('wrapper')
                break
            elif isinstance(item, dict) and 'double' in item:
                default_value = item['double'].get('wrapper')
                break
            elif isinstance(item, dict) and 'array' in item:
                default_value = item['array'].get('elements')
                break
            else:
                raise Exception

        metadata_details = []
        for item in type_specific_metadata:
            if isinstance(item, dict):
                for k, v in item.items():
                    metadata_details.append(f'{k}: {v}')
            if isinstance(item, str):
                metadata_details.append(f'{item}')

        if metadata_details:
            metadata_doc = ' (' + ', '.join(metadata_details) + ')'
        else:
            metadata_doc = ''

        if enum_type and enum_type in enum_type2values:
            used_enums[enum_type] = enum_type2values[enum_type]
            enum_class_name = ''.join([word.capitalize() for word in enum_type.split('_')])
            if default_value and default_value in enum_type2values[enum_type]:
                default_value_str = f'= {enum_class_name}.{default_value}'
            else:
                default_value_str = ''
            param_type_str = 'int'
            if len(default_value_str) != 0: # optional
                optional_params.append((param_name, f'{param_name}: {param_type_str} {default_value_str}'))
                param_docs[param_name] = f'- {param_name} ({param_type_str}): {param_display_name}. Must be one of the properties in class {enum_type}.'
                if len(default_value_str) != 0:
                    param_docs[param_name] += f'Default is {default_value_str}.{metadata_doc}'
            else:
                required_params.append((param_name, f'{param_name}: {param_type_str}'))
                param_docs[param_name] = f'- {param_name} ({param_type_str}): {param_display_name}. Must be one of the properties in class {enum_type}.{metadata_doc}'
        elif entity_type and entity_type in entity_type2props:
            used_entities[entity_type] = entity_type2props[entity_type]
            param_type_str = entity_type
            default_value_str = f' = {default_value}' if default_value else ''
            if  len(default_value_str) != 0:
                optional_params.append((param_name, f'{param_name}: {param_type_str}{default_value_str}'))
                param_docs[param_name] = f'- {param_name} ({param_type_str}): {param_display_name}. Default is {default_value}.{metadata_doc}'
            else:
                required_params.append((param_name, f'{param_name}: {param_type_str}{default_value_str}'))
                param_docs[param_name] = f'- {param_name} ({param_type_str}): {param_display_name}.{metadata_doc}'
        elif 'array' in value_type:
            member_type = value_type['array']['wrapper']['memberValueType']
            if 'primitive' in member_type:
                param_type_str = f'List[{param_type2python.get(str(member_type["primitive"]["wrapper"]["typeIdentifier"]), "object")}]'
            elif 'entity' in member_type:
                param_type_str = f'List[{member_type["entity"]["wrapper"]["typeName"]}]'
                used_entities[member_type["entity"]["wrapper"]["typeName"]] = entity_type2props[member_type["entity"]["wrapper"]["typeName"]]
            elif 'linkEnumeration' in member_type:
                param_type_str = 'List[int]'
                used_enums[member_type['linkEnumeration']['wrapper']['identifier']] = enum_type2values[member_type['linkEnumeration']['wrapper']['identifier']]
            else:
                param_type_str = 'List[object]'
            default_value_str = f' = {default_value}' if default_value else ''
            if  len(default_value_str) != 0:
                optional_params.append((param_name, f'{param_name}: {param_type_str}{default_value_str}'))
                param_docs[param_name] = f'- {param_name} ({param_type_str}): {param_display_name}. Default is {default_value}.{metadata_doc}'
            else:
                required_params.append((param_name, f'{param_name}: {param_type_str}{default_value_str}'))
                param_docs[param_name] = f'- {param_name} ({param_type_str}): {param_display_name}.{metadata_doc}'
        else:
            param_type_str = param_type2python.get(str(primitive_type), 'object')
            if param_type_str == 'str' and str(default_value).lower() != 'none':
                default_value = f'"{default_value}"'
            default_value_str = f' = {default_value}' if default_value is not None else ''
            if  default_value != None:
                optional_params.append((param_name, f'{param_name}: {param_type_str}{default_value_str}'))
                param_docs[param_name] = f'- {param_name} ({param_type_str}): {param_display_name}. Default is {default_value}.{metadata_doc}'
            else:
                required_params.append((param_name, f'{param_name}: {param_type_str}{default_value_str}'))
                param_docs[param_name] = f'- {param_name} ({param_type_str}): {param_display_name}.{metadata_doc}'

    output_type = config.get('outputType', {})
    if 'primitive' in output_type:
        return_type = param_type2python.get(str(output_type['primitive']['wrapper']['typeIdentifier']), 'str')
    elif 'entity' in output_type:
        return_type = output_type['entity']['wrapper']['typeName']
        used_entities[output_type['entity']['wrapper']['typeName']] = entity_type2props[output_type['entity']['wrapper']['typeName']]
    elif 'array' in output_type:
        member_type = output_type['array']['wrapper']['memberValueType']
        if 'primitive' in member_type:
            return_type = f'List[{str(param_type2python.get(member_type["primitive"]["wrapper"]["typeIdentifier"], "object"))}]'
        elif 'entity' in member_type:
            return_type = f'List[{member_type["entity"]["wrapper"]["typeName"]}]'
            used_entities[member_type["entity"]["wrapper"]["typeName"]] = entity_type2props[member_type["entity"]["wrapper"]["typeName"]]
        elif 'linkEnumeration' in member_type:
            return_type = 'List[int]'
            used_enums[member_type['linkEnumeration']['wrapper']['identifier']] = enum_type2values[member_type['linkEnumeration']['wrapper']['identifier']]
        else:
            return_type = 'List[object]'
    else:
        return_type = 'None'
    if return_type != 'None':
        identifier2output[action_name].append(return_type)

    all_params = required_params + optional_params
    all_params_str = [param[1] for param in all_params]
    function_signature = f'def {action_name}(\n    ' + ',\n    '.join(all_params_str) + f'\n) -> {return_type}:'

    if all_params:
        param_docs_str = '\n'.join([param_docs[_[0]] for _ in all_params])
    else:
        param_docs_str = '- None'

    docstring = f'"""\n{action_title}\n\n{action_description}\n\nParameters:\n' + param_docs_str + '\n\nReturns:\n- ' + return_type

    docstring += '\n"""'

    function_body = """
    pass
    """

    enum_definitions = []
    for enum_name, enum_values in used_enums.items():
        enum_class_name = ''.join([word.capitalize() for word in enum_name.split('_')])
        enum_class_definition = f'class {enum_class_name}:\n'
        for val_name, val_index in enum_values.items():
            enum_class_definition += f'    {val_name} = {val_index}\n'
        enum_definitions.append(enum_class_definition)

    # 生成实体类
    entity_definitions = []
    for entity_name, properties in used_entities.items():
        entity_class_name = ''.join([word.capitalize() for word in entity_name.split('_')])
        entity_class_definition = f'class {entity_class_name}:\n'
        for prop in properties:
            prop_name = prop['identifier']
            prop_type = param_type2python.get(str(prop['valueType'].get('primitive', {}).get('wrapper', {}).get('typeIdentifier')), 'object')
            entity_class_definition += f'    {prop_name}: {prop_type}\n'
        entity_definitions.append(entity_class_definition)

    python_code = '\n\n'.join(enum_definitions + entity_definitions) + f'\n\n{function_signature}\n    {docstring}\n{function_body}\n\n'
    identifier2python[action_name] = python_code

    if len(key_parameters):
        identifier2keyparams[action_name] = key_parameters

    return python_code



def stat_actions(actions):
    types_set = set()

    classes_set = set()
    classes2types = defaultdict()
    for action in actions.values():
        for para in action.get('Parameters', []):
            classes_set.add(para['Class'])
            if 'DefaultValue' in para:
                classes2types[para['Class']] = type(para['DefaultValue']).__name__
                types_set.add(type(para['DefaultValue']).__name__)
        output = action.get('Output', defaultdict(list))
        for class_type in output.get('Types', []):
            types_set.add(class_type)


def generate_function_signature_from_intent(intent_config, name_template, in_enums, identifier2python, identifier2keyparams, identifier2output):
    intent_name = intent_config.get('INIntentName', 'generated_function')
    intent_name = name_template.format(intent_name).replace('.', '_')
    intent_title = intent_config.get('INIntentTitle', 'No title available')
    intent_description = intent_config.get('INIntentDescription', 'No description available')
    parameters = intent_config.get('INIntentParameters', [])
    response = intent_config.get('INIntentResponse', {})
    response_params = response.get('INIntentResponseParameters', [])
    key_parameter = intent_config.get('INIntentKeyParameter', 'None')
    param_combinations = intent_config.get('INIntentManagedParameterCombinations', {})

    param_docs = {}
    return_types = []
    param_type2python = {'Integer': 'int', 'Decimal': 'float', 'Boolean': 'bool', 'File': 'list', 'String': 'str', 'Object': 'object'}

    enum_type2values = {enum['INEnumName']: {val['INEnumValueName']: val.get('INEnumValueIndex', 0) for val in enum['INEnumValues']} for enum in in_enums}
    used_enums = {}
    required_params = []
    optional_params = []

    for param in parameters:
        param_name = param.get('INIntentParameterName', 'param')
        param_display_name = param.get('INIntentParameterDisplayName', param_name)
        param_type = param.get('INIntentParameterType', 'String')
        metadata = param.get('INIntentParameterMetadata', {})
        default_value = metadata.get("INIntentParameterMetadataDefaultValue", None)

        enum_type = param.get('INIntentParameterEnumType')
        if enum_type and enum_type in enum_type2values:
            used_enums[enum_type] = enum_type2values[enum_type]
            enum_values = enum_type2values[enum_type]
            if default_value is not None:
                optional_params.append((param_name, f'{param_name}: int = {enum_type}.{default_value}'))
                param_docs[param_name] = f'- {param_name} (int): {param_display_name}. Must be one of properties in class {enum_type}. Default is {enum_type}.{default_value}.'
            else:
                required_params.append((param_name, f'{param_name}: int'))
                param_docs[param_name] = f'- {param_name} (int): {param_display_name}. Must be one of properties in class {enum_type}.'
        else:
            param_type_str = param_type2python.get(param_type, 'object')
            if default_value is not None:
                default_value = f'"{default_value}"' if param_type_str == 'str' else default_value
                optional_params.append((param_name, f'{param_name}: {param_type_str} = {default_value}'))
                param_docs[param_name] = f'- {param_name} ({param_type_str}): {param_display_name}. Default is {default_value}.'
            else:
                required_params.append((param_name, f'{param_name}: {param_type_str}'))
                param_docs[param_name] = f'- {param_name} ({param_type_str}): {param_display_name}.'

    response_docs = []
    for response_param in response_params:
        param_name = response_param.get('INIntentResponseParameterName')
        param_display_name = response_param.get('INIntentResponseParameterDisplayName', param_name)
        if 'error' in param_name.lower():
            has_error = True
            continue
        param_type = param_type2python.get(response_param.get('INIntentResponseParameterType'), 'str')
        response_docs.append(f'- {param_name} ({param_type}): {param_display_name}.')
        return_types.append(param_type)
    if not response_docs:
        response_docs.append('- None')

    if not return_types:
        function_return_type = 'None'
    elif len(return_types) == 1:
        function_return_type = return_types[0]
    else:
        function_return_type = 'dict'

    if function_return_type != 'None':
        identifier2output[intent_name] = [function_return_type]

    all_params = required_params + optional_params
    all_params_str = [param[1] for param in all_params]
    function_signature = f'def {intent_name}(\n    ' + ',\n    '.join(all_params_str) + f'\n) -> {function_return_type}:'

    if all_params:
        param_docs_str = '\n'.join([param_docs[_[0]] for _ in all_params])
    else:
        param_docs_str = '- None'

    docstring = f'"""\n{intent_title}\n\n{intent_description}\n\nParameters:\n' + param_docs_str + '\n\nReturns:\n' + '\n'.join(response_docs)
    docstring += '\n"""'

    function_body = """
    pass
    """

    enum_definitions = []
    for enum_name, enum_values in used_enums.items():
        enum_class_name = ''.join([word.capitalize() for word in enum_name.split('_')])
        enum_class_definition = f'class {enum_class_name}:\n'
        for val_name, val_index in enum_values.items():
            enum_class_definition += f'    {val_name} = {val_index}\n'
        enum_definitions.append(enum_class_definition)

    python_code = '\n\n'.join(enum_definitions) + f'\n\n{function_signature}\n    {docstring}\n{function_body}\n\n'
    identifier2python[intent_name] = python_code

    if key_parameter != 'None':
        identifier2keyparams[intent_name] = [key_parameter]


    return python_code




if __name__ == "__main__":
    identifier2python = defaultdict(str)
    identifier2keyparams = defaultdict(str)
    identifier2output = defaultdict(list)


    # # -----Internal API START---------
    actions = 'WFActions.plist'
    with open(actions, 'rb') as f:
        actions = f.read()
    actions = plistlib.loads(actions)
    actions = {k:v for k, v in actions.items() if 'WFHandleCustomIntentAction' not in str(v) }
    for name, action in actions.items():
        generate_function_signature_from_WFactions(name, action, identifier2python, identifier2keyparams, identifier2output)
    # # -----Internal APi END---------

    # -----Outer API START-------

    with open('4_api_json_filter.json', 'r') as fp:
        contents = json.load(fp)


    for content in contents:
        app_name = content['AppName']

        for key, value in content.items():
            if '.actionsdata' in key:
                Enums = value['enums']
                Entities = value['entities']
                name_template = f"{app_name}" + ".{}"
                for action_name, action in value['actions'].items():
                    print('=' * 50)
                    print(generate_function_signature_from_DOTactions(action ,action_name, name_template, Enums, Entities, identifier2python, {},identifier2output ))# NOTE: 不考虑keyparams
                    print('=' * 50)

            if '.intent' in key:
                intent_config = value
                Enum = intent_config.get('INEnums', [])
                name_templates = f"{app_name}" + ".{}Intent"
                for intent in intent_config['INIntents']:
                    print(generate_function_signature_from_intent(intent, name_templates, Enum, identifier2python , {}, identifier2output)) # NOTE: 不考虑keyparams
                    print('=' * 50)
    # -----Outer API END------

    with open('../data/identifier2python.pkl', 'wb') as file:
        pickle.dump(identifier2python, file)
    print(len(identifier2python))