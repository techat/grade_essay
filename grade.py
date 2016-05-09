import essay_set
import logging
import numpy as np
import util_functions
import math
import feature_extractor
import sklearn.ensemble

logging.basicConfig()
log = logging.getLogger(__name__)

class Grader():
    def __init__(self, model_data):
        self.model_data = model_data

    def grade(self, submission):
        """
        Grades a specified submission using specified models
        model_data - A dictionary:
        {
            'model' : trained model,
            'extractor' : trained feature extractor,
            'prompt' : prompt for the question,
            'algorithm' : algorithm for the question,
        }
        submission - The student submission (string)
        """
        #Initialize result dictionary
        results = {'errors': [],'tests': [],'score': 0, 'feedback' : "",
                   'success' : False, 'confidence' : 0}
        has_error=False

        grader_set = essay_set.EssaySet(essaytype="test")
        feedback = {}

        model, extractor = self.get_classifier_and_ext(self.model_data)

        try:
            #Try to add essay to essay set object
            grader_set.add_essay(str(submission),0)
            grader_set.update_prompt(str(self.model_data['prompt']))
        except Exception:
            error_message = "Essay could not be added to essay set:{0}".format(submission)
            log.exception(error_message)
            results['errors'].append(error_message)
            has_error=True

        #Try to extract features from submission and assign score via the model
        try:
            grader_feats=extractor.gen_feats(grader_set)
            feedback=extractor.gen_feedback(grader_set,grader_feats)[0]
            #print "*************feedback is: *****************"
            #print feedback
            results['score']=int(model.predict(grader_feats)[0])
        except Exception:
            error_message = "Could not extract features and score essay."
            log.exception(error_message)
            results['errors'].append(error_message)
            has_error=True

        #Try to determine confidence level
        try:
            results['confidence'] = self.get_confidence_value(self.model_data['algorithm'], model, grader_feats,
                                                              results['score'],
self.model_data['score'])
        except Exception:
            #If there is an error getting confidence, it is not a show-stopper, so just log
            log.exception("Problem generating confidence value")

        if not has_error:

            #If the essay is just a copy of the prompt, return a 0 as the score
            if('too_similar_to_prompt' in feedback and feedback['too_similar_to_prompt']):
                results['score']=0
                results['correct']=False

            results['success']=True

            #Generate short form output--number of problem areas identified in feedback

            #Add feedback to results if available
            results['feedback'] = {}
            if 'topicality' in feedback and 'prompt_overlap' in feedback:
                results['feedback'].update({
                    'topicality' : feedback['topicality'],
                    'prompt-overlap' : feedback['prompt_overlap'],
                    })

            results['feedback'].update(
                 {
                    'spelling' : feedback['spelling'],
                    'grammar' : feedback['grammar'],
                    'markup-text' : feedback['markup_text'],
                    }
            )
        else:
            #If error, success is False.
            results['success']=False
        return results

    def get_confidence_value(self, algorithm,model,grader_feats,score, scores):
        """
        Determines a confidence in a certain score, given proper input parameters
        algorithm- from util_functions.AlgorithmTypes
        model - a trained model
        grader_feats - a row of features used by the model for classification/regression
        score - The score assigned to the submission by a prior model
        """
        min_score=min(np.asarray(scores))
        max_score=max(np.asarray(scores))
        if algorithm == util_functions.AlgorithmTypes.classification and hasattr(model, "predict_proba"):
            #If classification, predict with probability, which gives you a matrix of confidences per score point
            raw_confidence=model.predict_proba(grader_feats)[0,(float(score)-float(min_score))]
            #TODO: Normalize confidence somehow here
            confidence=raw_confidence
        elif hasattr(model, "predict"):
            raw_confidence = model.predict(grader_feats)[0]
            confidence = max(float(raw_confidence) - math.floor(float(raw_confidence)), math.ceil(float(raw_confidence)) - float(raw_confidence))
        else:
            confidence = 0

        return confidence

    def get_classifier_and_ext(self, model_data):
        if 'classifier' in model_data:
            model = model_data['classifier']
        else:
            raise Exception("Cannot find a valid model.")

        if 'feature_ext' in model_data:
            extractor = model_data['feature_ext']
        else:
            raise Exception("Cannot find the extractor")

        return model, extractor

