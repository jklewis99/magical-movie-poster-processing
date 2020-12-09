import cv2
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

def show_labels_and_predictions(img, preds, actual,
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
    img = cv2.putText(img, "Actual Label(s):", (img.shape[1]-img.shape[0]//3, img.shape[0]//2), font, font_scale, color, thickness, cv2.LINE_AA)

    # loop through the predictions and actual labels and print them on the image
    for i, pred in enumerate(preds):
        img = cv2.putText(img, pred, (img.shape[1]-img.shape[0]//3, 40+(1+i)*40), font, font_scale, color, thickness, cv2.LINE_AA)
    for i, actual_label in enumerate(actual):
        img = cv2.putText(img, actual_label, (img.shape[1]-img.shape[0]//3, img.shape[0]//2+(1+i)*40), font, font_scale, color, thickness, cv2.LINE_AA)
    plt.figure()
    plt.imshow(img)
    plt.show()
    return img

class LabelsPerfect(Metric):
    #TODO: fix the inaccurate logic of this class
    def __init__(self, num_classes, name='labels_perfect', **kwargs):
        super(LabelsPerfect, self).__init__(name=name, **kwargs)
        self.perfect_labels = self.add_weight(name='labelsPerfect', initializer='zeros')
        self.num_classes = num_classes
        self.size = None

    def update_state(self, y_true, y_pred, sample_weight=None):
        # convert to boolean array
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        self.size = y_true.shape[0]

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.perfect_labels.assign_add(tf.reduce_sum(values))

    def result(self):
        return tf.divide(tf.divide(self.perfect_labels, self.num_classes), self.size)