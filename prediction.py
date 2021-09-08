import pandas as pd
from typing import List
from tensorflow.python.keras.engine.functional import Functional
import os


def apply_model(row_list: List, img_file: str, len_boxes: int, classifier: Functional) -> List:
    main_folder = "./images_data"
    w = {}
    for n in range(len_boxes):
        a = str(n + 1)
        b = "train_converted" + a + ".csv"
        x_pred = pd.read_csv(os.path.join(main_folder, img_file.replace(".jpg", ''), b))
        x_pred = x_pred.iloc[:, :].values.astype('float32')
        x_pred = x_pred.reshape(-1, 32, 32, 1)
        predictions = classifier.predict(x_pred)
        x = predictions[4]
        c = "predictions" + a
        w.update({c: x.argmax()})

    row = [img_file, str(w.get("predictions1")) +
           str(w.get("predictions2")) +
           str(w.get("predictions3")) +
           str(w.get("predictions4")) +
           str(w.get("predictions5")) +
           str(w.get("predictions6"))]
    row_list.append(row)
    print(img_file + " " + "SAYAÇ MİKTAR: " +
          str(w.get("predictions1")) +
          str(w.get("predictions2")) +
          str(w.get("predictions3")) +
          str(w.get("predictions4")) +
          str(w.get("predictions5")) +
          str(w.get("predictions6"))
          )

    return row_list
