from torchoptim.core.model_interface import ModelSignature

def test_signature_dataclass():
    sig = ModelSignature(input_schema={"p": "str"}, output_schema={"r": "str"})
    assert sig.input_schema["p"] == "str"