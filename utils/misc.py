import cv2
import numpy as np
from tensorflow.keras.metrics import Metric
import tensorflow as tf

def labels_to_text(preds, actual_genres, genre_names, threshold=0.3):
    '''
    convert the predictions and actual labels to their respective string genres
    '''
    text_preds = []
    for pred in preds:
        text_pred = []
        for i, val in enumerate(pred):
            if val >= threshold:
                text_pred.append(genre_names[i])
        text_preds.append(text_pred)
    text_actuals = []
    for actual in actual_genres:
        text_actual = []
        for i, val in enumerate(actual):
            if val == 1:
                text_actual.append(genre_names[i])
        text_actuals.append(text_actual)
    return text_preds, text_actuals

def show_poster_and_genres(img_path, text_preds, text_actual=None, save_img=False, model=None):
    '''
    display a window that shows the model's predictions and actual labels, if specified

    Parameters
    ==========
    `img_path`:
        absolute or relative path to poster image

    `text_preds`:
        list of the predicted genres for the poster

    Keyword Args
    ==========
    `text_actual`:
        list of the actual genres for the poster. Default None

    `save_img`:
        if True, iamges is saved to "figures/xception_preds/{img_id}.png"

    `model`:
        if `save_img` is True, the name of the model
    '''
    img = cv2.imread(img_path)
    img = show_labels_and_predictions(img, text_preds, text_actual)
    img_id = img_path.split("/")[-1][:-4]
    print(model, img_id)
    cv2.imshow(img_id, img)
    cv2.waitKey(0)
    if save_img:
        cv2.imwrite(f"figures/{model}-predictions-{img_id}.png", img)

def show_labels_and_predictions(img, preds, actual=None,
                                font=cv2.FONT_HERSHEY_COMPLEX,
                                color=(0, 0, 0),
                                thickness=1,
                                font_scale=0.5,
                                desired_size=(300, 450)):
    '''
    read image and draw text to show the predictions and actual labels to the right of the image
    '''
    img = cv2.resize(img, desired_size)
    
    # add a blank (white) space to the right of the image
    img = np.concatenate((img, np.full((len(img), img.shape[0]//3, 3), 255, dtype=np.uint8)), axis=1)
    
    # print 'Predictions' and 'Actual' on the figure
    img = cv2.putText(img, "Prediction(s):", (img.shape[1]-img.shape[0]//3, 40), font, font_scale, color, thickness, cv2.LINE_AA)
    # loop through the predictions and actual labels and print them on the image
    for i, pred in enumerate(preds):
        img = cv2.putText(img, pred, (img.shape[1]-img.shape[0]//3, 40+(1+i)*30), font, font_scale, color, thickness, cv2.LINE_AA)
    if actual:
        img = cv2.putText(
            img,
            "Actual Label(s):",
            (img.shape[1]-img.shape[0]//3, img.shape[0]//2),
            font, font_scale, color, thickness, cv2.LINE_AA)

        for i, actual_label in enumerate(actual):
            img = cv2.putText(
                img,
                actual_label,
                (img.shape[1]-img.shape[0]//3, img.shape[0]//2+(1+i)*30),
                font, font_scale, color, thickness, cv2.LINE_AA)

    # plt.figure()
    # plt.imshow(img)
    # plt.show()
    return img

def get_genres():
    '''
    return the full list of strings with the names of the genres
    
    Return
    ==========
    genres
    '''
     # All the genres in the posters-and-genres.csv metadata file
    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror',
              'Music', 'Musical', 'Mystery', 'N/A', 'News', 'Reality-TV', 'Romance',
              'Sci-Fi', 'Short', ' Sport', 'Thriller', 'War', 'Western']

    return genres

class LabelsPerfect(Metric):
    #TODO: fix the inaccurate logic of this class
    def __init__(self, num_classes, threshold=0.3, name='labels_perfect', **kwargs):
        super(LabelsPerfect, self).__init__(name=name, **kwargs)
        self.perfect_labels = self.add_weight(name='labelsPerfect', initializer='zeros')
        self.num_classes = num_classes
        self.size = None
        self.threshold = threshold
        self.to_1 = tf.constant(1, shape=num_classes, dtype=tf.float32)
        self.to_0 = tf.constant(0, shape=num_classes, dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # convert values above threshold to 1, and those below to 0
        y_pred = tf.where(tf.greater(y_pred, self.threshold), self.to_1, self.to_0)
        # convert to boolean array
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        # create masks for boolean comparison
        num_samples = y_pred.shape[0]
        trues_to_0 = tf.constant(1, shape=(num_samples,), dtype=tf.float32)
        false_to_1 = tf.constant(0, shape=(num_samples,), dtype=tf.float32)

        values = tf.math.equal(y_true, y_pred) # find element-wise equality
        values = tf.cast(values, self.dtype) # cast to original data type
        counts = tf.reduce_sum(values, 1) # get sum of true values in each row
        # if count in the row is equal to all classes, then it is a perfect prediction
        perfect_samples = tf.where(counts==self.num_classes, trues_to_0, false_to_1)
        num_perfect = tf.reduce_sum(perfect_samples) # count the number of correct samples
        self.perfect_labels.assign_add(num_perfect)

    def result(self):
        return self.perfect_labels
