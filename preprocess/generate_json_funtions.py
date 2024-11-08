from utils import *
from collections import defaultdict
import pickle
import json


def generate_json_signature_from_WFactions(name, config, identifier2json):
    description = config.get('Description', {}).get('DescriptionSummary', 'No description available.')

    parameters = config.get('Parameters', [])
    output_config = config.get('Output', None)

    required_parameters = []
    optional_parameters = []

    for param in parameters:
        param_type = param.get('Class', 'object')
        param_key = param.get("Key", "")
        param_description = param.get("Placeholder", param.get("Label", ""))

        param_entry = {
            'name': param_key,
            'type': param_type,
            'description': param_description
        }

        if param.get("DefaultValue", param.get('Placeholder', None)) is not None:
            optional_parameters.append(param_entry)
        else:
            required_parameters.append(param_entry)

    return_data = []
    if output_config:
        output_name = output_config.get('OutputName', 'Output')
        output_description = output_config.get('DescriptionSummary', 'Output data.')
        return_data.append({'name': output_name, 'description': output_description})

    function_signature = [
        [name, {
            'name': name,
            'description': description,
            'required_parameters': required_parameters,
            'optional_parameters': optional_parameters,
            'return_data': return_data
        }]
    ]

    identifier2json[name] = function_signature

    return function_signature


def generate_json_signature_from_DOTactions(action, action_name, name_template, in_enums, in_entities, identifier2json):
    action_identifier = name_template.format(action_name)

    action_title = action.get('title', {}).get('key', 'No title available')
    action_description = action.get('descriptionMetadata', {}).get('descriptionText', {}).get('key', '')

    enum_type2values = {enum['identifier']: {val['identifier']: idx for idx, val in enumerate(enum['cases'])} for enum in in_enums}
    used_enums = {}

    entity_type2props = {entity_name: entity['properties'] for entity_name, entity in in_entities.items()}
    used_entities = {}

    parameters = action.get('parameters', [])
    required_parameters = []
    optional_parameters = []

    for param in parameters:
        param_name = param.get('name', 'param')
        param_display_name = param.get('title', {}).get('key', param_name)
        value_type = param.get('valueType', {})
        primitive_type = value_type.get('primitive', {}).get('wrapper', {}).get('typeIdentifier')
        enum_type = value_type.get('linkEnumeration', {}).get('wrapper', {}).get('identifier')
        entity_type = value_type.get('entity', {}).get('wrapper', {}).get('typeName')
        is_optional = param.get('isOptional', True)

        param_entry = {
            'name': param_name,
            'type': 'object',
            'description': param_display_name
        }

        if enum_type and enum_type in enum_type2values:
            used_enums[enum_type] = enum_type2values[enum_type]
            param_entry['type'] = 'ENUM'
            param_entry['enum_class'] = enum_type

        elif entity_type and entity_type in entity_type2props:
            used_entities[entity_type] = entity_type2props[entity_type]
            param_entry['type'] = 'ENTITY'
            param_entry['entity_class'] = entity_type

        elif primitive_type is not None:
            param_type_map = {'0': 'STRING', '1': 'BOOLEAN', '2': 'INTEGER', '7': 'FLOAT', '8': 'DATETIME', '9': 'DATETIME'}
            param_entry['type'] = param_type_map.get(str(primitive_type), 'STRING')

        if is_optional:
            optional_parameters.append(param_entry)
        else:
            required_parameters.append(param_entry)

    return_data = []
    output_type = action.get('outputType', {})

    if 'primitive' in output_type:
        return_data.append({
            'name': 'result',
            'description': f"The result in {output_type['primitive']['wrapper']['typeIdentifier']} format."
        })

    elif 'entity' in output_type:
        return_data.append({
            'name': output_type['entity']['wrapper']['typeName'],
            'description': f"The {output_type['entity']['wrapper']['typeName']} entity returned by the action."
        })

    elif 'linkEnumeration' in output_type:
        enum_type = output_type['linkEnumeration']['wrapper']['identifier']
        if enum_type in enum_type2values:
            used_enums[enum_type] = enum_type2values[enum_type]
            return_data.append({
                'name': 'result',
                'description': f"The result as an ENUM of type {enum_type}."
            })


    elif 'intents' in output_type:
        return_data.append({
            'name': 'result',
            'description': "The result is an intent of type identifier 12."
        })

    elif 'array' in output_type:
        member_value_type = output_type['array']['wrapper']['memberValueType']

        # 数组的元素是实体类型
        if 'entity' in member_value_type:
            entity_name = member_value_type['entity']['wrapper']['typeName']
            return_data.append({
                'name': 'result',
                'description': f"The result is an array of {entity_name} entities."
            })
            used_entities[entity_name] = entity_type2props[entity_name]

        elif 'intents' in member_value_type:
            return_data.append({
                'name': 'result',
                'description': "The result is an array of intents with type identifier 12."
            })

        elif 'primitive' in member_value_type:
            primitive_type = member_value_type['primitive']['wrapper']['typeIdentifier']
            return_data.append({
                'name': 'result',
                'description': f"The result is an array of {primitive_type} types."
            })
    elif 'measurement' in output_type:
        unit_type = output_type['measurement']['wrapper']['unitType']
        return_data.append({
            'name': 'result',
            'description': f"The result is a measurement with unit type {unit_type}."
        })

    elif len(output_type) == 0:
        pass

    else:
        print(output_type)

    action_signature = [
        action_identifier, {
            'name': action_identifier,
            'description': action_description,
            'required_parameters': required_parameters,
            'optional_parameters': optional_parameters,
            'return_data': return_data
        }
    ]

    identifier2json[action_identifier] = action_signature

    return action_signature


def generate_json_signature_from_intent(intent_config, name_template, in_enums, identifier2json):
    intent_name = intent_config.get('INIntentName', 'generated_function')
    intent_name = name_template.format(intent_name)
    intent_description = intent_config.get('INIntentDescription', 'No description available')
    parameters = intent_config.get('INIntentParameters', [])
    response = intent_config.get('INIntentResponse', {})
    response_params = response.get('INIntentResponseParameters', [])

    param_type2json = {
        'Integer': 'NUMBER',
        'Decimal': 'NUMBER',
        'Boolean': 'BOOLEAN',
        'File': 'FILE',
        'String': 'STRING',
        'Object': 'OBJECT'
    }

    enum_type2values = {
        enum['INEnumName']: {val['INEnumValueName']: idx for idx, val in enumerate(enum['INEnumValues'])}
        for enum in in_enums
    }
    used_enums = {}

    required_params = []
    optional_params = []

    # 处理参数
    for param in parameters:
        param_name = param.get('INIntentParameterName', 'param')
        param_display_name = param.get('INIntentParameterDisplayName', param_name)
        param_type = param.get('INIntentParameterType', 'String')
        metadata = param.get('INIntentParameterMetadata', {})
        default_value = metadata.get("INIntentParameterMetadataDefaultValue", None)

        enum_type = param.get('INIntentParameterObjectType')
        if enum_type and enum_type in enum_type2values:
            used_enums[enum_type] = enum_type2values[enum_type]
            param_entry = {
                'name': param_name,
                'type': 'ENUM',
                'description': param_display_name,
                'enum_class': enum_type
            }
            if default_value is not None:
                optional_params.append(param_entry)
            else:
                required_params.append(param_entry)
        else:
            param_entry = {
                'name': param_name,
                'type': param_type2json.get(param_type, 'OBJECT'),
                'description': param_display_name
            }
            if default_value is not None:
                optional_params.append(param_entry)
            else:
                required_params.append(param_entry)

    return_data = []
    if response_params:
        for response_param in response_params:
            param_name = response_param.get('INIntentResponseParameterName')
            param_display_name = response_param.get('INIntentResponseParameterDisplayName', param_name)
            param_type = param_type2json.get(response_param.get('INIntentResponseParameterType'), 'STRING')
            return_data.append({
                'name': param_name,
                'description': f'{param_display_name}'
            })

    action_signature = [
        intent_name, {
            'name': intent_name,
            'description': intent_description,
            'required_parameters': required_params,
            'optional_parameters': optional_params,
            'return_data': return_data
        }
    ]

    identifier2json[intent_name] = action_signature

    return action_signature


if __name__ == "__main__":
    identifier2json = defaultdict(str)


    actions = 'WFActions.plist'
    with open(actions, 'rb') as f:
        actions = f.read()
    actions = plistlib.loads(actions)


    actions = {k:v for k, v in actions.items() if 'WFHandleCustomIntentAction' not in str(v) }
    for name, action in actions.items():
        generate_json_signature_from_WFactions(name, action, identifier2json)

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
                    print(generate_json_signature_from_DOTactions(action ,action_name, name_template, Enums, Entities, identifier2json))
                    print('=' * 50)

            if '.intent' in key:
                intent_config = value
                Enum = intent_config.get('INEnums', [])
                name_templates = f"{app_name}" + ".{}Intent"
                for intent in intent_config['INIntents']:
                    print(generate_json_signature_from_intent(intent, name_templates, Enum, identifier2json))
                    print('=' * 50)
    with open('../data/identifier2json.pkl', 'wb') as file:
        pickle.dump(identifier2json, file)
    print(len(identifier2json))
