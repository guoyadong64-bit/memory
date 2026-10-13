"""Evidence-local name enrichment; never merge physical entity IDs by similarity."""
import re


def tokens(text):
    return re.findall(r"[a-z]+", text.lower().replace("_", " "))


def contextual_labels(entities, description):
    """Copy uniquely attributable noun phrases, not scene-wide semantic guesses.

    A phrase must contain a declared noun head and at least two identity/category
    words. Competing entities with equal evidence leave the phrase unassigned.
    Returned labels refer to the same observation, not independent visual truth.
    """
    anchors = {}
    for entity in entities:
        names = [entity.name, *[l.text for l in entity.labels if l.confidence >= .8]]
        terms = set(tokens(" ".join(names)))
        heads = {parts[-1] for name in names if (parts := tokens(name))}
        anchors[entity.id] = (terms, heads)
    boundary = {"a", "an", "the", "this", "that", "and", "or", "with", "of",
                "on", "in", "inside", "above", "below", "near", "over", "under",
                "from", "to", "by", "is", "are", "was", "were", "has", "have",
                "holding", "lifting", "grasping", "carrying", "placing", "releasing"}
    result = []
    for sentence in re.split(r"[.,;:!?()]", description.lower()):
        text = tokens(sentence)
        for end, word in enumerate(text):
            if not any(word in heads for _, heads in anchors.values()):
                continue
            start = max(0, end - 5)
            for index in range(start, end):
                if text[index] in boundary:
                    start = index + 1
            phrase_words = text[start:end + 1]
            if len(phrase_words) < 2:
                continue
            phrase = " ".join(phrase_words)
            candidates = [(len(set(phrase_words) & terms), entity.id)
                          for entity in entities
                          for terms, heads in [anchors[entity.id]] if word in heads
                          and len(set(phrase_words) & terms) >= 2]
            if not candidates:
                continue
            best = max(score for score, _ in candidates)
            identities = [identity for score, identity in candidates if score == best]
            if len(identities) != 1:
                continue
            entity = next(e for e in entities if e.id == identities[0])
            # Confidence stays bounded by an existing category assertion.
            confidence = max((l.confidence for l in entity.labels
                              if l.confidence >= .8 and word in tokens(l.text)), default=0)
            if confidence >= .8:
                result.append((entity.id, phrase, confidence))
    return list(dict.fromkeys(result))


def direct_lift(frame):
    """A local lift is supported even if the eventual placement is unobserved.

    Require held_by plus an affirmative actor-lift-object clause. An approach,
    attempted lift, or background use of the word 'lift' does not establish it.
    """
    interaction = frame.get("interaction") or {}
    subject, actor = interaction.get("object_id"), interaction.get("actor_id")
    if not subject or not actor or interaction.get("confidence", 0) < .8:
        return False
    if not any(r.get("source") == subject and r.get("target") == actor and
               r.get("predicate") == "held_by" and r.get("confidence", 0) >= .8
               for r in frame.get("relations", [])):
        return False
    entities = {e["id"]: e for e in frame.get("entities", [])}
    if subject not in entities or actor not in entities:
        return False
    obj = entities[subject]
    obj_words = set(tokens(obj["name"])) | {
        w for label in obj.get("labels", []) if label.get("confidence", 0) >= .8
        for w in tokens(label["text"])}
    actor_names = {actor.replace("_", " "), entities[actor]["name"].replace("_", " "),
                   "robotic arm", "robot arm", "robotic gripper", "gripper", "robot"}
    for clause in re.split(r"[.;!?]", frame.get("description", "").lower()):
        # Negation or modal/attempt language in the clause prevents certification.
        if re.search(r"\b(?:not|never|no|may|might|could|attempt\w*|try\w*|trying|before)\b", clause):
            continue
        match = re.search(r"\b(?:" + "|".join(re.escape(s) for s in sorted(actor_names, key=len, reverse=True)) +
            r")\b\s+(?:(?:is|was|has|now|actively|currently|just|successfully)\s+)*"
            r"(?:lift(?:s|ed|ing)?|pick(?:s|ed|ing)?\s+up)\s+(.+)", clause)
        if not match:
            held_objects = {r.get("source") for r in frame.get("relations", [])
                            if r.get("predicate") == "held_by" and r.get("target") == actor
                            and r.get("confidence", 0) >= .8}
            actor_pattern = "|".join(re.escape(s) for s in sorted(actor_names, key=len, reverse=True))
            if held_objects == {subject} and re.search(
                r"\b(?:" + actor_pattern + r")\b\s+has\s+closed\s+around\s+the\s+object,\s*"
                r"lifting\s+it\s+from\b", clause):
                return True
            continue
        phrase = re.split(r"\b(?:from|above|over|with|while|and)\b", match.group(1))[0]
        matched = set(tokens(phrase)) & obj_words
        # Require an object noun rather than its color alone.
        heads = {parts[-1] for name in [obj["name"], *[
            l["text"] for l in obj.get("labels", []) if l.get("confidence", 0) >= .8]]
                 if (parts := tokens(name))}
        if len(matched) >= 2 and matched & heads:
            return True
    return False
