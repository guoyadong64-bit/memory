"""Explicit, configurable normalization of JSON rows; no inferred units or identities."""
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import Observation


class ObservationMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    modality: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    input_unit: str = Field(min_length=1)
    output_unit: str = Field(min_length=1)
    fields: dict[str,str]
    entity_aliases: dict[str,str] = Field(default_factory=dict)
    scale: float = Field(default=1,allow_inf_nan=False)
    offset: float = Field(default=0,allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_fields(self):
        required = {"id","entity","value","observed_at","provenance"}
        if not required.issubset(self.fields) or set(self.fields)-required-{"unit","quality"}:
            raise ValueError("Mapping requires id/entity/value/observed_at/provenance fields.")
        if any(not value.strip() for value in self.fields.values()):
            raise ValueError("Mapped field names must be nonempty.")
        if self.input_unit != self.output_unit and self.scale == 1 and self.offset == 0:
            raise ValueError("Unit conversion requires an explicit scale or offset.")
        return self


def normalize_rows(rows: list[dict], mapping: ObservationMapping) -> list[Observation]:
    result = []
    for row in rows:
        if not isinstance(row,dict):
            raise ValueError("Each raw observation must be a JSON object.")
        try:
            data = {target:row[source] for target,source in mapping.fields.items()}
        except KeyError:
            raise ValueError("Raw observation is missing a mapped field.") from None
        if data.pop("unit",mapping.input_unit) != mapping.input_unit:
            raise ValueError("Input unit does not match the declared mapping.")
        value = data["value"]
        if isinstance(value,bool):
            raise ValueError("Boolean values require explicit categorical normalization.")
        if mapping.scale != 1 or mapping.offset != 0:
            if not isinstance(value,(int,float)):
                raise ValueError("Unit conversion requires a numeric value.")
            data["value"] = value*mapping.scale+mapping.offset
        entity = str(data["entity"])
        data["entity"] = mapping.entity_aliases.get(entity,entity)
        data["id"] = mapping.source+":"+str(data["id"])
        result.append(Observation(**data,source=mapping.source,modality=mapping.modality,
                                  namespace=mapping.namespace,metric=mapping.metric,unit=mapping.output_unit))
    if len({o.id for o in result}) != len(result):
        raise ValueError("Normalized observation IDs must be unique.")
    return result
