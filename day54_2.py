# Comment Analysis between +ve and -ve Comment
import pickle
def check_review(textData):
    recreated_model = pickle.load(open('model_data.pkl', 'rb'))
    myvocab = pickle.load(open('features_data.pkl', 'rb'))
    from sklearn.feature_extraction.text import CountVectorizer
    vocab = CountVectorizer(vocabulary= myvocab, decode_error='replace')
    from sklearn.feature_extraction.text import TfidfTransformer
    cvtotf = TfidfTransformer()
    return recreated_model.predict(cvtotf.fit_transform(vocab.transform([textData])))

mycomment = input("Enter the Comment to do analysis: ")
if (check_review(mycomment)):
    print("Yeah!! you have a Positive Comment")
else:
    print('Sorry!! The Comment by User is Negative')
