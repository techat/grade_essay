import os
import logging
import random 
import sys
from model_creator import ModelCreator
from grade import Grader

logging.basicConfig()
log = logging.getLogger(__name__)

CHARACTER_LIMIT = 1000000
TRAINING_LIMIT = 20000
QUICK_TEST_LIMIT = 20000

class HewlettLoader():
    def __init__(self, _pathname, _filename):
        self.pathname = _pathname
        self.filename = os.path.join(_pathname, _filename) 
           
    def normalize_score(self, essay_set, score):
        if essay_set == "1":
            return (score - 2) / (12.0 - 2.0) * 100
        elif essay_set == "2_1":          
            return (score - 1) / (6.0 - 1.0) * 100
        elif essay_set == "2_2":
            return (score - 1) / (4.0 - 1.0) * 100
        elif essay_set == "7":
            return (score - 0) / (30.0 - 0.0) * 100
        elif essay_set == "8":
            return (score - 0) / (60.0 - 0.0) * 100

    def add_essay_training(self, data, essay_set, essay, score):
        if essay_set not in data:
            data[essay_set] = {"essay":[],"score":[]}
        score = self.normalize_score(essay_set, score)
        data[essay_set]["essay"].append(essay)
        data[essay_set]["score"].append(score)
     
    def read_training_data(self):
        f = open(self.filename)
        f.readline()

        training_data = {}
        for row in f:
            row = row.strip().split("\t")
            essay_set = row[1]
            # Skip essay sets 3-6 as they are source_response type essays
            if essay_set in ['3', '4', '5', '6']:
                continue
            essay = row[2]
            domain1_score = int(row[6])
            if essay_set == "2":
                essay_set = "2_1"
            self.add_essay_training(training_data, essay_set, essay, domain1_score)
            
            if essay_set == "2_1":
                essay_set = "2_2"
                domain2_score = int(row[9])
                self.add_essay_training(training_data, essay_set, essay, domain2_score)

        return training_data

def main():
    pathname = "/home/kungangli/aes/data/hewlett/"
    filename = "training_set_rel3.tsv"
    dataloader = HewlettLoader(pathname, filename)
    training_data = dataloader.read_training_data()
    
    #scores, text = training_data["2"]['score'], training_data["2"]['essay']

    scores, text = [], []
    for i in training_data.keys():
        scores += training_data[i]['score']
        text += training_data[i]['essay']

    # Shuffle to mix up the classes, set seed to make it repeatable
    #random.seed(1)
    #shuffled_scores = []
    #shuffled_text = []
    #indices = [i for i in xrange(0,len(scores))]
    #random.shuffle(indices)
    #for i in indices:
    #    shuffled_scores.append(scores[i])
    #    shuffled_text.append(text[i])

    #text = shuffled_text[:TRAINING_LIMIT]
    #scores = shuffled_scores[:TRAINING_LIMIT]

    # Model creation and grading
    score_subset = scores[:QUICK_TEST_LIMIT]
    text_subset = text[:QUICK_TEST_LIMIT]
    print "The score subset is: "
    print score_subset
    prompt1 = "More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you."
    prompt2 = "Censorship in the Libraries. All of us can think of a book that we hope none of our children or any other children have taken off the shelf. But if I have the right to remove that book from the shelf -- that work I abhor --then you also have exactly the same right and so does everyone else. And then we have no books left on the shelf for any of us. -- Katherine Paterson, Author. Write a persuasive essay to a newspaper reflecting your vies on censorship in libraries. Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive? Support your position with convincing arguments from your own experience, observations, and/or reading"
    prompt7 = "Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining. Do only one of the following: write a story about a time when you were patient OR write a story about a time when someone you know was patient OR write a story in your own way about patience."
    prompt8 = "We all understand the benefits of laughter. For example, someone once said, Laughter is the shortest distance between two people. Many other people believe that laughter is an important part of any relationship. Tell a true story in which laughter was one element or part."
    prompt = prompt1 + " " + prompt2 + " " + prompt7 + " " + prompt8
    model_creator = ModelCreator(score_subset, text_subset, prompt)
    results = model_creator.create_model()
    print "model creator results: "
    print results["cv_kappa"]
    print results["cv_mean_absolute_error"]

    test_essay_testdataset = "I believe that computers have a positive effect on people. They help you stay in touch with family in a couple different ways they excercise your mind and hands and help you learn and make things easier. Computer's help you keep in touch with people. Say you live in @LOCATION1 and you miss your @CAPS1. You can just send an e-mail and talk all you want. If you don't just want to limit it to words you can add pictures so they can see how much you've grown or if you are well. Even if you're just e-mailing someone down the block it is just as effective as getting up and walking over there. You can also use a computer to make a scrap book card or slide show to show how much you love the person you give them to. Computers @MONTH1 not excercise you whole body but it excersises you mind and hands. You could play solitaire on the computer and come away @PERCENT1 smarter than before. You can play other games of strategy like checkers and chess while still sitting at home being comfortable. Your hands always play a big role while you're on the computer. They need to move the mouse and press the keys on a keyboard. Your hands learn all the keys from memorization. It's like the computer's teaching handi-coordination and studying habit for the future. Computers make human lives easier. Not only do they help kids turn in a nice neatly printed piece or paper for home work but they also help the average person. Teachers use it to keep peoples grades in order and others use it to write reports for various jobs. The @CAPS2 probably uses one to write a speech or to just keep his day in order. Computers make it easier to learn certain topics like the @LOCATION2 history. You can type something into a searcher site and have ton's of websites for one person with, who knows how much imformation. Instead of flipping through all the pages in a dictionary you can look for an online dictionary, type in the word and you have the definition. Computers have positive effects on people because they help you keep close to your family, they challenge your mind to be greater and excercise your hands and they make life easier for kids and the average person. This is why, I think computers have good effects on society"
    test_essay_traindataset10 = "Dear @CAPS1, @CAPS2 you imagine a world where technology controls human interaction? Its not such a far fetched idea. Some people @MONTH1 see this as a great advancement. Not me. I think that computers detach people from reality, create false personas, and has no physical benefit. This is what i think the computers effects have on people. First of all, Computers detach people from reality. Its a lot easier for someone to sit on the computer, browsing aimlessly on some social networking website, than go out and actually meet new people. Sure you @CAPS2 send all the friend requests you want, but theres nothing that compares to a first impression that you get from meeting someone in person. The computer can also have an affect on family members and the time spent with them. Secondly, the computer creates false personas. If i had a facebook or @CAPS3 account, it would be very easy to say something false about myself, or someone else. The computer dosn't know better. This allows people to gain, ""friendships"" with others that dont know the real you. I find this as a real problem because eventually lead to shaky or unsteady relationshops. Lastly, I think that the computer offers no physical gain, This nation is in a severe heath crisis. With things easier to get now a days like fast food, and drugs, I don't think that computers help. People need discipline and theres a time and place for everything, including computers. However those that aren't disciplined might use them more than they need to and that isn't good. So @CAPS2 you imagine a world controlled by computers? I think that little by little, thats whats happening, slowly taking away the human element. I think the effects will be severe, detaching reality, creating false personas, and offering no physical gain. I don't think its right, but I hope I've persuaded you to think my way."
    test_essay_offtopic = "To Sherlock Holmes she is always THE woman. I have seldom heard him mention her under any other name. In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind. He was, I take it, the most perfect reasoning and observing machine that the world has seen, but as a lover he would have placed himself in a false position. He never spoke of the softer passions, save with a gibe and a sneer. They were admirable things for the observer--excellent for drawing the veil from men's motives and actions. But for the trained reasoner to admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which might throw a doubt upon all his mental results. Grit in a sensitive instrument, or a crack in one of his own high-power lenses, would not be more disturbing than a strong emotion in a nature such as his. And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory"
    test_essay_pupil = "The first reason is my family. Over half of my family lives in New Jersey. When I visit, my cousins and I laugh and play all day and night. My uncles and aunts take me to the boardwalk where we ride roller coasters. We devour juicy caramel-covered apples and foot-long hot dogs. My family is fun to be with. The second reason for New Jersey being my favorite place is the weather. Instead of being hot and sweaty, it's always cool and moist. When I think about my visits, I can just feel the crisp fall breeze in my hair. I can just see the white, fluffy winter snow. I can just hear the soft spring trickles of rain splashing on the sidewalks. I can just feel the warm summer sun on my face. The weather is great! The third reason for New Jersey being my favorite place is crabbing. If it's crab season, we crab. We keep the blue crabs and the snow crabs, and we let the others go. Sometimes we catch crabs on hooks, and sometimes we lower crab cages into the bay. Then we pull them out later. One time my brother caught a crab so big that it got stuck in the crab cage! The crab finally got out, but it hurt one of its legs and broke the cage trying. Poor crab!"
    test_essay_badessay = "I live in a house that every body in it came from acting. I remember my mom telling me this it you in find your self bad situation, don't forget your smile with 'you'. I think she ment that what ever is the difficulty think always positive. For an example, I grow up in place that full with bad poeple and one time some body try to convinse me to smoke. And smoking it very bad thing. So I started to tell joukes on people that canser and after 2 minutes I change the subject. Or that every time I am getting sick and fill not so good. I am trying to see comedy movies as much as I can. Because I have been told that comedy is the best cure. I think that as an actor on the stage you need to be always ready for something rong, and if you ready and prepard. It will be good and life for your self in you all life and not only there. This experience importent for your benfits, always a positive person and people will love you and get along with you. This mark it the best."
    test_essay_randomhigh = "Kentish lyricised constringent gullibly farmerville dynamited lamaist bush. Sour agitate chemotherapist astrateia quorum olecranon filippino episcopize. Painfulness phenolic labyrinthodont assurbanipal denominatively lenses dreg. Innoxiously unsentineled intervaginal thorez separator hilum republication quinquagesima. Jam assafetida nim evincing dilatancy fellowship fasciately messy. Harmonised pout pedicellate froghopper reconform riskier demo bram. Sine racketlike homophonic cravenness halidome excursionary benny transequatorial. Necrotising prereversed waitress paperwork elbowroom ultramodern cauterization unsaturated. Unflaking greely erzerum hypochilia oppugnant recondemnation underspread noncoincident. Unaccepted pyrope laurelled magnific office foehn smoothhound absolutism. Explanator conker surrogated inframarginal hopscotch supremum excel flocci. Saintliness unfabricated remittence chink senecan conjectured beep prototypically."
    test_essay_randomlow = "Add you viewing ten equally believe put. Separate families my on drawings do oh offended strictly elegance. Perceive jointure be mistress by jennings properly. An admiration at he discovered difficulty continuing. We in building removing possible suitable friendly on. Nay middleton him admitting consulted and behaviour son household. Recurred advanced he oh together entrance speedily suitable. Ready tried gay state fat could boy its among shall. Continual delighted as elsewhere am convinced unfeeling. Introduced stimulated attachment no by projection. To loud lady whom my mile sold four. Need miss all four case fine age tell. He families my pleasant speaking it bringing it thoughts. View busy dine oh in knew if even. Boy these along far own other equal old fanny charm. Difficulty invitation put introduced see middletons nor preference"

    grader = Grader(results)
    #results = grader.grade(text[10])
    results = grader.grade(prompt)
    print "model grader results prompt: "
    print results
    results = grader.grade(test_essay_offtopic)
    print "model grader results test_essay_offtopic: "
    print results
    results = grader.grade(test_essay_testdataset)
    print "model grader results test_essay_testdataset: "
    print results
    results = grader.grade(test_essay_traindataset10)
    print "model grader results test_essay_traindataset10: "
    print results
    results = grader.grade(test_essay_pupil)
    print "model grader results test_essay_pupil:" 
    print results
    results = grader.grade(test_essay_badessay)
    print "model grader results test_essay_badessay: "
    print results
    results = grader.grade(test_essay_randomhigh)
    print "model grader results test_essay_randomhigh: "
    print results
    results = grader.grade(test_essay_randomlow)
    print "model grader results test_essay_randomlow: "
    print results

    #for i in xrange(len(score_subset)):
    #    print i + 1, score_subset[i], grader.grade(text_subset[i])["score"]

if __name__ == '__main__':
    main()    
