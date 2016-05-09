import logging
import util_functions
import essay_set
import feature_extractor
import numpy as np
import sklearn

logging.basicConfig()
log = logging.getLogger(__name__)

class ModelCreator():
    def __init__(self, scores, text, prompt):
        self.scores = scores
        self.text = text
        self.prompt = prompt

    def create_model(self):
        return self.create(self.text, self.scores, self.prompt)

    def create(self, text,score,prompt_string):
        """
        Creates a machine learning model from input text, associated scores, a prompt, and a path to the model
        text - A list of strings containing the text of the essays
        score - a list of integers containing score values
        prompt_string - the common prompt for the set of essays
        """
        algorithm = self.select_algorithm(score)
        #Initialize a results dictionary to return
        results = {'errors': [],'success' : False, 'cv_kappa' : 0,
                   'cv_mean_absolute_error': 0, 'feature_ext' : "", 
                   'classifier' : "", 'algorithm' : algorithm,
                   'score' : score, 'text' : text, 'prompt' : prompt_string
                  }
        print ("The length of text and score lists are respectively: " + str(len(text)) + ' ' + str(len(score)))

        if len(text)!=len(score):
            msg = "Target and text lists must be same length."
            results['errors'].append(msg)
            log.exception(msg)
            return results

        try:
            #Create an essay set object that encapsulates all the essays and alternate representations (tokens, etc)
            e_set = self.create_essay_set(text, score, prompt_string)
        except:
            msg = "essay set creation failed."
            results['errors'].append(msg)
            log.exception(msg)
        try:
            #Gets features from the essay set and computes error
            feature_ext, classifier, cv_error_results = self.extract_features_and_generate_model(e_set, algorithm)
            results['cv_kappa']=cv_error_results['kappa']
            results['cv_mean_absolute_error']=cv_error_results['mae']
            results['feature_ext']=feature_ext
            results['classifier']=classifier
            results['algorithm'] = algorithm
            results['success']=True
        except:
            msg = "feature extraction and model creation failed."
            results['errors'].append(msg)
            log.exception(msg)

        return results

    def select_algorithm(self, score_list):
        #Decide what algorithm to use (regression or classification)
        try:
            #Count the number of unique score points in the score list
            if len(util_functions.uniq_elem(list(score_list)))>5:
                algorithm = util_functions.AlgorithmTypes.regression
            else:
                algorithm = util_functions.AlgorithmTypes.classification
        except:
            algorithm = util_functions.AlgorithmTypes.regression

        return algorithm

    def create_essay_set(self, text, score, prompt_string, generate_additional=True):
        """
        Creates an essay set from given data.
        Text should be a list of strings corresponding to essay text.
        Score should be a list of scores where score[n] corresponds to text[n]
        Prompt string is just a string containing the essay prompt.
        Generate_additional indicates whether to generate additional essays 
        at the minimum score point or not.
        """
        e_set = essay_set.EssaySet()
        for i in xrange(0, len(text)):
            e_set.add_essay(text[i], score[i])

        e_set.update_prompt(prompt_string)

        return e_set

    def extract_features_and_generate_model(self, essays, algorithm=util_functions.AlgorithmTypes.regression):
        """
        Feed in an essay set to get feature vector and classifier
        essays must be an essay set object
        additional array is an optional argument that can specify
        a numpy array of values to add in
        returns a trained FeatureExtractor object and a trained classifier
        """
        f = feature_extractor.FeatureExtractor()
        f.initialize_dictionaries(essays)
        print "Essays are: "
        print essays._type
        print essays._score
        #print essays._id
        #print essays._tokens
        #print essays._pos
        #print essays._generated
        print essays._prompt
        #print essays._spelling_errors

        train_feats = f.gen_feats(essays)
        set_score = np.asarray(essays._score, dtype=np.int)
        clf,clf2 = self.get_algorithms(algorithm)
        cv_error_results = self.get_cv_error(clf2,train_feats,essays._score)

        try:
            clf.fit(train_feats, set_score)
        except ValueError:
            log.exception("Not enough classes (0,1,etc) in sample.")
            set_score[0]=1
            set_score[1]=0
            clf.fit(train_feats, set_score)

        return f, clf, cv_error_results

    def get_algorithms(self, algorithm):
        """
        Gets two classifiers for each type of algorithm, and returns them.
        First for predicting, second for cv error. type - one of
        util_functions.AlgorithmTypes
        """
        if algorithm == util_functions.AlgorithmTypes.classification:
            clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100,
                  learning_rate=.05, max_depth=4, random_state=1,min_samples_leaf=3)
            clf2= sklearn.ensemble.GradientBoostingClassifier(n_estimators=100,
                  learning_rate=.05, max_depth=4, random_state=1,min_samples_leaf=3)
        else:
            print "This is a regression problem*******************************************"
            clf = sklearn.ensemble.GradientBoostingRegressor(n_estimators=500, max_features = None,
                  learning_rate=.05, random_state=1, min_samples_split=1)
            clf2= sklearn.ensemble.GradientBoostingRegressor(n_estimators=500, max_features = None,
                  learning_rate=.05, random_state=1, min_samples_split=1)
            #clf = sklearn.ensemble.RandomForestRegressor(n_estimators=100, min_samples_split=1, random_state = 1)
            #clf2 = sklearn.ensemble.RandomForestRegressor(n_estimators=100, min_samples_split=1, random_state = 1)
        return clf, clf2

    def get_cv_error(self, clf,feats,scores):
        """
        Gets cross validated error for a given classifier, set of features, and scores
        clf - classifier
        feats - features to feed into the classified and cross validate over
        scores - scores associated with the features -- feature row 1 associates with score 1, etc.
        """
        results={'success' : False, 'kappa' : 0, 'mae' : 0}
        try:
            cv_preds=util_functions.gen_cv_preds(clf,feats,scores)
            err=np.mean(np.abs(np.array(cv_preds)-scores))
            kappa=util_functions.quadratic_weighted_kappa(list(cv_preds),scores)
            results['mae']=err
            results['kappa']=kappa
            results['success']=True
        except ValueError as ex:
            # If this is hit, everything is fine.  It is hard to explain why the
            # error occurs, but it isn't a big deal.
            msg = u"Not enough classes (0,1,etc) in each cross validation fold: {ex}".format(ex=ex)
            log.debug(msg)
        except:
            log.exception("Error getting cv error estimates.")

        return results
