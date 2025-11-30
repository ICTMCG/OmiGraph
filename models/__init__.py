from .sem_graph import SemGraph
from .bert import BertModel
from .int_graph import IntGraph

def build_model(args):
    model_name = args.model.lower()
    if model_name == "semgraph":
        print("Using SemGraph model")
        return SemGraph(args)
    elif model_name == "intgraph":
        print("Using IntGraph model")
        return IntGraph(args)
    elif model_name == "bert":
        print("Using BertModel")
        return BertModel(args)
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented.")