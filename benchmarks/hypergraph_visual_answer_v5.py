"""Common answer rules with query-conditioned choice of at most twelve images.

The historical answerer stays frozen. Prompts, normalizer, validation and model
settings below are the same; only image selection is an opt-in V5 operation.
"""
import base64
import json
from pathlib import Path

from benchmarks.answer_normalization import select_any_target
from benchmarks.findingdory_mem0 import answer_request
from benchmarks.findingdory_pilot import goal_indices
from benchmarks.findingdory_visual_answer import VISUAL_ANSWER_SYSTEM, answer_errors
from benchmarks.hypergraph_recall_v5 import select_images_v5
from memory_agent.clients import ModelError


def decode_visual(client, question, evidence, observations, data, episode, resolution=None):
    if not evidence:
        return {'goal_frames': [], 'evidence_ids': [], 'visual_frames': []}
    indices = select_images_v5(evidence, observations, resolution)
    state = answer_request(question, evidence, resolution) | {'attached_visual_frames': indices}
    catalog = {}
    for frame in observations:
        if frame['frame_index'] in indices:
            for entity in frame.get('entities', []):
                catalog.setdefault(entity['id'], entity)
    state['observed_entity_catalog'] = list(catalog.values())
    content = [{'type': 'text', 'text': json.dumps(state, ensure_ascii=False)}]
    for index in indices:
        path = Path(data)/'frames'/episode/f'{index:03d}.jpg'
        content.extend([{'type': 'text', 'text': f'Original evidence image: frame_index={index}'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+
                base64.b64encode(path.read_bytes()).decode(), 'detail': 'high'}}])
    model = client.settings.answer_model or client.settings.llm_model
    repairs, normalizations = [], []
    for attempt in range(2):
        request_content = content
        if repairs:
            request_content = content + [{'type': 'text', 'text': json.dumps({
                'repair_required': repairs[-1]['errors'], 'previous_output': repairs[-1]['output'],
                'instruction': 'Recheck the same evidence. Return a complete supported set; do not invent missing goals or identities.'})}]
        response = client.post(client.settings.llm_url.rstrip('/')+'/chat/completions', client.settings.llm_key,
            {'model': model, 'temperature': 0, 'max_tokens': 3072, 'response_format': {'type': 'json_object'},
             **client.chat_options(model), 'messages': [{'role': 'system', 'content': VISUAL_ANSWER_SYSTEM},
                                                       {'role': 'user', 'content': request_content}]}, 'answer')
        try:
            output = json.loads(response['choices'][0]['message']['content'])
            if not isinstance(output, dict):
                raise ValueError()
        except (ValueError, KeyError, IndexError, TypeError):
            raise ModelError('answer: invalid visual structured answer') from None
        output, normalization = select_any_target(output, question, evidence)
        if normalization is not None:
            normalizations.append(normalization)
        errors = answer_errors(output, evidence, question, resolution, set(catalog))
        if not errors:
            break
        repairs.append({'errors': errors, 'output': output})
    predicted = goal_indices(json.dumps(output))
    ids = output.get('evidence_ids', [])
    known = {item['id']: item for item in evidence}
    valid = (predicted and isinstance(ids, list) and ids and all(isinstance(i,str) and i in known for i in ids)
             and set(predicted) <= {n for i in ids for n in known[i]['frame_indices']})
    if not valid or errors:
        return {'goal_frames': [], 'evidence_ids': [], 'visual_frames': indices,
                'validation': errors or ['no_supported_answer'], 'raw_output': output,
                'repairs': repairs, 'normalizations': normalizations}
    return {'goal_frames': predicted, 'evidence_ids': ids, 'visual_frames': indices,
            'target_bindings': output.get('target_bindings', []), 'repairs': repairs, 'normalizations': normalizations}
