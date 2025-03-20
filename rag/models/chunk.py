from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import json

@dataclass
class Chunk:
    id: str
    doc_id: str = ""
    kb_id: str = ""
    create_time: str = ""
    create_timestamp_flt: float = 0.0
    img_id: str = ""
    docnm_kwd: str = ""
    title_tks: str = ""
    title_sm_tks: str = ""
    name_kwd: str = ""
    important_kwd: List[str] = field(default_factory=list)
    tag_kwd: List[str] = field(default_factory=list)
    important_tks: str = ""
    question_kwd: List[str] = field(default_factory=list)
    question_tks: str = ""
    content_with_weight: str = ""
    content_ltks: str = ""
    content_sm_ltks: str = ""
    authors_tks: str = ""
    authors_sm_tks: str = ""
    page_num_int: List[int] = field(default_factory=list)
    top_int: List[int] = field(default_factory=list)
    position_int: List[List[int]] = field(default_factory=list)
    weight_int: int = 0
    weight_flt: float = 0.0
    rank_int: int = 0
    rank_flt: float = 0
    available_int: int = 1
    knowledge_graph_kwd: str = ""
    entities_kwd: List[str] = field(default_factory=list)
    pagerank_fea: int = 0
    tag_feas: Dict = field(default_factory=dict)
    vector_embeddings: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for field_name, field_value in self.__dict__.items():
            if field_name == "vector_embeddings":
                d.update(field_value)
                continue
            
            if field_name in ["important_kwd", "question_kwd", "entities_kwd", "tag_kwd"]:
                d[field_name] = "###".join(field_value)
            elif field_name == "position_int":
                arr = [num for row in field_value for num in row]
                d[field_name] = "_".join(f"{num:08x}" for num in arr)
            elif field_name in ["page_num_int", "top_int"]:
                d[field_name] = "_".join(f"{num:08x}" for num in field_value)
            elif field_name == "tag_feas":
                d[field_name] = json.dumps(field_value)
            else:
                d[field_name] = field_value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        vector_embeddings = {}
        chunk_dict = {}
        
        for k, v in d.items():
            if k.endswith("_vec"):
                vector_embeddings[k] = v
                continue
                
            if k in ["important_kwd", "question_kwd", "entities_kwd", "tag_kwd"]:
                chunk_dict[k] = [x for x in v.split("###") if x] if v else []
            elif k == "position_int":
                if v:
                    arr = [int(x, 16) for x in v.split("_")]
                    chunk_dict[k] = [arr[i:i+5] for i in range(0, len(arr), 5)]
                else:
                    chunk_dict[k] = []
            elif k in ["page_num_int", "top_int"]:
                chunk_dict[k] = [int(x, 16) for x in v.split("_")] if v else []
            elif k == "tag_feas":
                chunk_dict[k] = json.loads(v) if v else {}
            else:
                chunk_dict[k] = v
                
        chunk_dict["vector_embeddings"] = vector_embeddings
        return cls(**chunk_dict)
