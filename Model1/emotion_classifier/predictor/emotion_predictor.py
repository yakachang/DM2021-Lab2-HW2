from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors import Predictor
from typing import List
from overrides import overrides

MULTI_LABEL_TO_INDEX = {
    'anger': 0, 
    'anticipation': 1, 
    'disgust': 2, 
    'fear': 3, 
    'sadness': 4, 
    'surprise': 5, 
    'trust': 6, 
    'joy': 7
}

@Predictor.register('emotion_predictor')
class EmotionPredictor(Predictor):
    @overrides
    def predict_json(self, inputs: JsonDict) -> str:
        probs = self.predict_probs(inputs)
        return {'text': inputs['text'], 'probs': probs}

    def predict_probs(self, inputs: JsonDict):
        
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        probs = output_dict['probs']
        return probs

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[str]:
        instances = self._batch_json_to_instances(inputs)
        output_dicts = self.predict_batch_instance(instances)
        results = []
        for inp, od in zip(inputs, output_dicts):
            results.append(
                {'text': inp['text'], 'probs': od['probs']})
        return results

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict['text']
        return self._dataset_reader.example_to_instance(text=text, label=None)