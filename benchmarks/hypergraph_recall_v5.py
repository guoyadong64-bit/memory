"""Query-conditioned incidence retrieval over original, frame-bound witnesses.

No task IDs, categories, evaluation labels or previous answers enter this module.
Read views collect observations; they never certify new identities or events.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import re

from benchmarks.comparison_methods import HypergraphMethod
from benchmarks.findingdory_memory import history_query
from benchmarks.hypergraph_evidence_subgraph import pack_subgraph, record_frames
from benchmarks.hypergraph_recall_v3 import _namespace
from memory_agent.recall import ORDINALS, parse_plan, words

LIMITS = {'candidate_frames': 32, 'evidence_records': 20, 'evidence_bytes': 40000,
          'answer_images': 12, 'incidence_hops': 2}
STOP = set('navigate revisit robot goal yesterday previously interacted interaction ending '
           'with that which from picked placed before after immediately took time '
           'current location longest shortest order specific second first third fourth fifth '
           'all any not did have had to on in is of and them it an a the '
           'farthest nearest closest rearranged moved ending visited was were been'.split())


def query_terms(question):
    # Keep category/appearance nouns but remove instruction grammar. Instance
    # names are taken from observed text, never a benchmark category lookup.
    return words(question) - {w.rstrip('s') for w in STOP} - set(ORDINALS) - {
        '', 'receptacle', 'object', 'room', 'interact', 'rearrange', 'end', 'pick', 'place'}


class WitnessIndex:
    def __init__(self, records):
        self.records = {r.id: r for r in records}
        spaces = {_namespace(r) for r in records} - {None}
        if len(spaces) != 1:
            raise ValueError('V5 witness index requires exactly one episode namespace')
        self.namespace = next(iter(spaces))
        self.frames, self.originals = {}, {}
        self.appearances = defaultdict(set)
        self.roles = defaultdict(lambda: defaultdict(set))
        self.entity_terms = defaultdict(set)
        self.events = []
        for record in sorted(records, key=lambda r: (bool(r.evolution), r.id)):
            for entity in record.entities:
                # Role participants may be supported in another original frame
                # of the event. They remain indexed even when absent in the
                # particular query viewpoint; this does not assert visibility.
                self.entity_terms[entity.id].update(words(entity.name))
                for label in entity.labels:
                    if label.confidence >= .8:
                        self.entity_terms[entity.id].update(words(label.text))
            for obs in record.observations:
                try:
                    frame = json.loads(obs.value)
                except (ValueError, TypeError):
                    continue
                if not isinstance(frame, dict) or not isinstance(frame.get('frame_index'), int):
                    continue
                n = frame['frame_index']
                if n not in self.frames:
                    self.frames[n] = frame
                    self.originals[n] = (obs, record)
                for entity in frame.get('entities', []):
                    if entity.get('kind') != 'agent' and entity.get('attributes', {}).get('role') != 'robot_component':
                        self.appearances[entity['id']].add(n)
            observation_ids = {o.id: n for n, (o, _) in self.originals.items()}
            for role in record.role_bindings:
                if role.role in {'object', 'source', 'destination'} and role.confidence >= .8:
                    self.roles[role.role][role.entity_id].update(
                        observation_ids[oid] for oid in role.observation_ids if oid in observation_ids)
            if record.evolution:
                self.events.append(record)
        self.events.sort(key=lambda r: (min(record_frames(r), default=0), r.id))
        self.tokens, self.visible, self.features = {}, {}, {}
        self.clock_incidence = defaultdict(set)
        for n, frame in self.frames.items():
            if frame.get('clock'):
                self.clock_incidence[frame['clock']].add(n)
            self.visible[n] = {e['id'] for e in frame.get('entities', [])
                               if e['id'] in self.appearances}
            for entity in frame.get('entities', []):
                text = entity['name'] + ' ' + ' '.join(str(v) for v in entity.get('attributes', {}).values())
                text += ' ' + ' '.join(label['text'] for label in entity.get('labels', [])
                                      if label.get('confidence', 0) >= .8)
                self.entity_terms[entity['id']].update(words(text))
            self.tokens[n] = words(frame.get('description', '')+' '+str(frame.get('clock') or '')) | set().union(
                *(self.entity_terms[e] for e in self.visible[n]))
            action = frame.get('interaction') or {}
            features = {'entity:'+e for e in self.visible[n]}
            # Role-labelled incidences remain specific to the original frame.
            for field in ('object_id', 'source_id', 'target_id'):
                if action.get(field):
                    features.add(field+':'+action[field])
            if action.get('object_id'):
                features.add('phase:'+action['object_id']+':'+str(action.get('phase')))
            self.features[n] = features
        frequencies = Counter(t for tokens in self.tokens.values() for t in tokens)
        self.idf = {t: math.log(1+(len(self.frames)+1)/(count+1)) for t, count in frequencies.items()}

    def rank(self, seed_ids, question, resolution=None):
        plan = parse_plan(question)
        terms = query_terms(question)
        named = {identity for identity, tokens in self.entity_terms.items()
                 if terms and len(terms & tokens)/len(terms) >= .5}
        # The requested answer and the conditioning role differ in "the object
        # picked from a cabinet". Keep the source as a constraint, not the target.
        object_target = bool(re.search(r'(?:navigate to|revisit)\s+(?:(?:the|an?|any|all)\s+)*objects?\b', question, re.I))
        wanted = 'object' if object_target else plan.goal if plan.goal in {'source', 'destination'} else 'object'
        relevant_events = []
        connected = set()
        for event in self.events:
            participants = {b.entity_id for b in event.role_bindings if b.confidence >= .8}
            if participants & named or not terms:
                relevant_events.append(event)
                connected.update(b.entity_id for b in event.role_bindings
                                 if b.role == wanted and b.confidence >= .8)
        if plan.anchor_hint:
            anchor_terms = words(plan.anchor_hint)
            anchors = [i for i,event in enumerate(self.events) if any(
                b.role == 'object' and anchor_terms and
                len(anchor_terms & self.entity_terms.get(b.entity_id,set()))/len(anchor_terms) >= .5
                for b in event.role_bindings)]
            direction = -1 if plan.relation == 'before' else 1
            for i in anchors:
                for delta in {max(1,plan.offset-1),plan.offset,plan.offset+1}:
                    neighbor = i+direction*delta
                    if 0 <= neighbor < len(self.events):
                        connected.update(b.entity_id for b in self.events[neighbor].role_bindings
                                         if b.role == wanted and b.confidence >= .8)
        seeds = defaultdict(float)
        for rank, identity in enumerate(seed_ids):
            for n in record_frames(self.records[identity]) if identity in self.records else ():
                seeds[n] = max(seeds[n], 1/(1+.12*rank))
        protected = {t['frame_index'] for t in (resolution or {}).get('targets', [])
                     if t.get('frame_index') in self.frames}
        if object_target and plan.goal in {'source','destination'}:
            protected = set()
        event_frames = defaultdict(set)
        for event in self.events:
            for n in record_frames(event):
                if n in self.frames:
                    event_frames[n].add(event.id)
        temporal = bool(plan.anchor_hint or plan.ordinal or plan.visit_order or plan.all_targets
                        or re.search(r'\b(?:longest|shortest|duration)\b', question, re.I))
        weights = {}
        requested_clocks = {f'{int(hour):02d}:{minute}' for hour,minute in
                            re.findall(r'\b([01]?\d|2[0-3]):([0-5]\d)\b',question)}
        clock_matches = set().union(*(self.clock_incidence[t] for t in requested_clocks)) if requested_clocks else set()
        for n, frame in self.frames.items():
            overlap = sum(self.idf.get(t, 1) for t in terms & self.tokens[n])
            denominator = sum(self.idf.get(t, 1) for t in terms) or 1
            lexical = overlap/denominator
            action = frame.get('interaction') or {}
            role_match = any(n in ns for e, ns in self.roles[wanted].items()
                             if not named or e in named or e in connected)
            named_visible = len(self.visible[n] & named)
            connected_visible = len(self.visible[n] & connected)
            score = (1.25 if plan.anchor_hint else 2.2)*lexical + .65*seeds[n] + .65*min(named_visible, 2)
            score += .65*role_match + .45*min(connected_visible, 2)
            score += 1.5*(n in clock_matches)
            if named_visible and re.search(r'close[- ]up|closer view|prominent|clearly visible',
                                          frame.get('description',''),re.I):
                score += .2  # Visual clarity cue, never a metric-distance assertion.
            if temporal:
                score += .5*bool(event_frames[n]) + .25*bool(action)
            if action.get('phase') in {'approach', 'grasp', 'release', 'settled'}:
                score += .15
            if plan.goal == 'uninteracted':
                # Absence of an observed action is uncertainty, not a proof.
                acted = set(self.roles['object']) | set(self.roles['source']) | set(self.roles['destination'])
                score += .4*bool((self.visible[n] & named)-acted)
                if self.visible[n] & named and not ((self.visible[n] & named)-acted):
                    score -= .6
            weights[n] = score + 4*(n in protected)
        selected = []
        covered = set()
        visited_events = set()
        remaining = set(self.frames)
        focus = named | connected
        focus_seen = set()
        near_penalty = defaultdict(float)
        duplicate_penalty = defaultdict(float)
        while remaining and len(selected) < LIMITS['candidate_frames']:
            def utility(n):
                features = self.features[n]
                fresh = len(features-covered)/max(1, len(features))
                event_gain = len(event_frames[n]-visited_events)
                # Late positions receive no implicit preference. Distant
                # viewpoints can supply genuinely different spatial witnesses.
                intent_gain = min(2,len((self.visible[n] & focus)-focus_seen))
                return weights[n] + .65*fresh + .6*min(event_gain, 1) + .9*intent_gain - near_penalty[n] - duplicate_penalty[n], -n
            n = max(remaining, key=utility)
            selected.append(n)
            remaining.remove(n)
            covered.update(self.features[n])
            focus_seen.update(self.visible[n] & focus)
            visited_events.update(event_frames[n])
            for other in remaining:
                near_penalty[other] = max(near_penalty[other], .8/(1+abs(other-n)))
                if abs(other-n) <= 3:
                    shared = len(self.visible[other]&self.visible[n])/max(1,len(self.visible[other]|self.visible[n]))
                    duplicate_penalty[other] = max(duplicate_penalty[other], .6*shared)
        return selected, {'query_terms': sorted(terms), 'answer_role': wanted,
                          'condition_role': plan.goal, 'matched_entity_ids': sorted(named),
                          'linked_role_entity_ids': sorted(connected), 'protected_frames': sorted(protected),
                          'frame_priority': selected, 'weights': {str(n): round(weights[n],4) for n in selected}}

    def project(self, seed_ids, question, resolution=None):
        indices, details = self.rank(seed_ids, question, resolution)
        views = []
        frame_to_view = {}
        # Each view is a collection of independent original witnesses, not a
        # newly asserted event. Two rows per view respect the record budget.
        for offset in range(0, len(indices), 2):
            chosen = indices[offset:offset+2]
            observations = [self.originals[n][0] for n in chosen]
            obs_ids = {o.id for o in observations}
            entities = {}
            roles = []
            for n in chosen:
                source = self.originals[n][1]
                for entity in source.entities:
                    labels = [label.model_copy(update={'observation_ids': [oid for oid in label.observation_ids if oid in obs_ids]})
                              for label in entity.labels if set(label.observation_ids)&obs_ids]
                    if entity.id not in entities:
                        entities[entity.id] = entity.model_copy(update={'labels': labels})
                for role in source.role_bindings:
                    kept = [oid for oid in role.observation_ids if oid in obs_ids]
                    if kept:
                        roles.append(role.model_copy(update={'observation_ids': kept, 'stage_id': None}))
            root = self.originals[chosen[0]][1]
            identity = 'query_witness_'+hashlib.sha256((self.namespace+str(chosen)).encode()).hexdigest()[:20]
            view = root.model_copy(update={'id': identity, 'observations': observations,
                'entities': list(entities.values()), 'role_bindings': roles, 'evolution': None,
                'answer': '{}', 'source': 'V5 query-time original witness collection; independent observations'})
            views.append(view)
            frame_to_view.update({n: identity for n in chosen})
        revised = dict(resolution) if resolution is not None else None
        if revised is not None:
            if details['answer_role'] == 'object' and details['condition_role'] in {'source','destination'}:
                revised.update(complete=False, targets=[], warnings=[*revised.get('warnings',[]),
                    'Location is a conditioning role; the requested target is the object, not the location.'])
            original_targets = list(revised.get('targets', []))
            revised['targets'] = [t | {'record_id': frame_to_view[t['frame_index']]}
                                  for t in revised.get('targets', []) if t.get('frame_index') in frame_to_view]
            if len(revised['targets']) != len(original_targets):
                revised.update(complete=False, warnings=[*revised.get('warnings',[]),
                    'Candidate budget omitted one or more proposed target witnesses.'])
            for key in ('record_ids', 'target_record_ids', 'anchor_record_ids'):
                revised[key] = list(dict.fromkeys(frame_to_view[n] for old in revised.get(key, []) if old in self.records
                    for n in record_frames(self.records[old]) if n in frame_to_view))
        fragment_info = [{
            'source_record_id': event.id,
            'phases': list(dict.fromkeys(s.phase for s in event.evolution.stages)),
            'observed_frame_range': [min(record_frames(event)), max(record_frames(event))],
            'roles': [{'role': role, 'entity_id': entity, 'basis': basis} for role,entity,basis in sorted({
                (b.role,b.entity_id,b.basis) for b in event.role_bindings if b.role in {'object','source','destination'}})],
            'retained_witness_frames': sorted(set(record_frames(event)) & set(indices)),
            'complete_history': False,
        } for event in self.events if set(record_frames(event)) & set(indices)]
        reserve = len(json.dumps(fragment_info,ensure_ascii=False).encode()) + len(str(indices))*2 + 800
        evidence, revised, audit = pack_subgraph(views, revised, max_bytes=max(2,40000-reserve))
        allowed = {n for item in evidence for n in item['frame_indices']}
        if evidence:
            evidence[0]['content']['query_frame_priority'] = [n for n in indices if n in allowed]
            evidence[0]['content']['projection_caveat'] = (
                'Independent witnessed frames; collection membership is not event membership. '
                'Shared names do not merge physical IDs; frame gaps do not establish durations.')
            evidence[0]['content']['observed_event_fragments'] = [f | {
                'retained_witness_frames': [n for n in f['retained_witness_frames'] if n in allowed]}
                for f in fragment_info if set(f['retained_witness_frames']) & allowed]
        size = len(json.dumps(evidence, ensure_ascii=False).encode())
        if size > LIMITS['evidence_bytes'] or len(evidence) > LIMITS['evidence_records']:
            raise ValueError('V5 evidence budget exceeded')
        audit.update(version='query-conditioned-incidence-v5', bytes=size, limits=LIMITS,
                     candidate_frames=len(allowed), image_priority=[n for n in indices if n in allowed],
                     query=details, certified_event_membership=False)
        return evidence, revised, audit


def project_retrieval(records, seed_ids, question, resolution=None):
    namespaces = {_namespace(r) for r in records if r.id in set(seed_ids)} - {None}
    if len(namespaces) != 1:
        raise ValueError('V5 retrieval seeds must identify exactly one namespace')
    namespace = next(iter(namespaces))
    return WitnessIndex([r for r in records if _namespace(r) == namespace]).project(seed_ids, question, resolution)


class HypergraphRecallV5(HypergraphMethod):
    def build(self):
        super().build()
        self.witness_indices = {ep: WitnessIndex(records) for ep, records in self.records.items()}

    def retrieve(self, episode, question):
        settings = self.client.settings
        query = history_query(question, episode, self.records[episode])
        query.recall_plan = parse_plan(question)
        selected, retrieval = self.graphs[episode].search(query, 20, strategy='hypergraph', embedder=self.client.embed,
            embedding_namespace=(settings.embedding_url or settings.llm_url).rstrip('/')+'|'+settings.embedding_model,
            minimum_similarity=settings.semantic_min_similarity)
        if retrieval['semantic_backend'] != 'embedding':
            raise ValueError('Hypergraph embedding fallback; comparison stopped')
        evidence, resolution, audit = self.witness_indices[episode].project(
            [r.id for r in selected[:20]], question, retrieval.get('plan_evidence'))
        retrieval['read_projection_v5'] = audit
        return evidence, resolution, retrieval


def select_images_v5(evidence, observations, resolution=None, limit=12):
    from benchmarks.findingdory_visual_answer import select_images
    if limit < 1:
        return []
    allowed = {n for item in evidence for n in item['frame_indices']}
    priority = next((item['content'].get('query_frame_priority', []) for item in evidence
                     if isinstance(item.get('content'), dict) and item['content'].get('query_frame_priority')), [])
    result = list(dict.fromkeys(n for n in priority if n in allowed))[:limit]
    if len(result) < limit:
        result.extend(n for n in select_images(evidence, observations, resolution, limit) if n not in result)
    return result[:limit]
